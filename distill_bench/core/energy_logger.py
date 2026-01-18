import os
import csv
import json
import time
import torch
import pynvml
import threading
import atexit
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path

from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from .environment import save_environment
from .utils import _safe_filename, _write_json


def _infer_nvml_device_indices(device_count: int) -> List[int]:
    """
    Infer which NVML GPU indices to poll based on environment variables.
    Handles common Slurm patterns + cgroup remapping.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return list(range(device_count))

    tokens = [t.strip() for t in cvd.split(",") if t.strip()]

    # CUDA_VISIBLE_DEVICES can be UUID-form ("GPU-...") on some systems.
    # If UUIDs are present, safest fallback is to poll visible indices only.
    if any(t.startswith("GPU-") or t.startswith("MIG-") for t in tokens):
        return list(range(device_count))

    # Numeric form
    try:
        inds = [int(t) for t in tokens]
    except ValueError:
        return list(range(device_count))

    # If cgroups/remapping expose only N GPUs, NVML count may be N, but CVD might still contain
    # physical indices (rare) or logical ones. If any index is out of range, treat as remapped.
    if not inds:
        return list(range(device_count))
    if max(inds) >= device_count:
        return list(range(device_count))

    # Otherwise treat as physical indices
    return inds


@dataclass
class StageMetrics:
    """Metrics collected for a single stage of the pipeline."""

    stage_id: str
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    # Token counts
    tokens_processed: int = 0

    # GPU metrics (from NVML polling)
    gpu_energy_joules: float = 0.0
    gpu_avg_power_watts: float = 0.0
    gpu_peak_power_watts: float = 0.0
    gpu_power_samples: List[float] = field(default_factory=list)

    # CodeCarbon metrics
    total_codecarbon_energy_kwh: float = 0.0
    codecarbon_emissions_kg: float = 0.0

    # CPU + total
    cpu_energy_joules: float = 0.0
    total_energy_joules: float = 0.0
    total_energy_kwh: float = 0.0

    # Derived metrics
    joules_per_token: float = 0.0
    kwh_total: float = 0.0
    tokens_per_second: float = 0.0

    def compute_derived_metrics(self, total_energy_policy: str = "measured"):
        """Compute derived metrics after stage completion."""
        self.duration_seconds = self.end_time - self.start_time

        if self.duration_seconds > 0 and self.tokens_processed > 0:
            self.tokens_per_second = self.tokens_processed / self.duration_seconds

        gpu_j = self.gpu_energy_joules
        cpu_j = self.cpu_energy_joules
        cc_kwh = self.total_codecarbon_energy_kwh
        cc_j = cc_kwh * 3_600_000 if cc_kwh > 0 else 0.0

        # Total energy selection policy
        if total_energy_policy == "codecarbon" and cc_j > 0:
            self.total_energy_joules = cc_j
        elif total_energy_policy == "gpu_only" and gpu_j > 0:
            self.total_energy_joules = gpu_j
        else:
            # "measured": prefer measured GPU+CPU; else fall back to CodeCarbon
            if gpu_j > 0 or cpu_j > 0:
                self.total_energy_joules = gpu_j + cpu_j
            elif cc_j > 0:
                self.total_energy_joules = cc_j
            else:
                self.total_energy_joules = 0.0

        self.total_energy_kwh = self.total_energy_joules / 3_600_000 if self.total_energy_joules > 0 else 0.0

        # Backwards-compatible kWh field:
        # use measured total if we have it; else fall back to CodeCarbon or GPU-only.
        if self.total_energy_kwh > 0:
            self.kwh_total = self.total_energy_kwh
        elif cc_kwh > 0:
            self.kwh_total = cc_kwh
        elif gpu_j > 0:
            self.kwh_total = gpu_j / 3_600_000
        else:
            self.kwh_total = 0.0

        if self.tokens_processed > 0 and self.total_energy_joules > 0:
            self.joules_per_token = self.total_energy_joules / self.tokens_processed
        else:
            self.joules_per_token = 0.0

    def to_dict(self, include_power_samples: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_power_samples:
            data.pop("gpu_power_samples", None)
        return data


class RAPLReader:
    """
    Minimal RAPL wrapper to read package (and optional dram) energy via powercap.
    """

    def __init__(self, rapl_root: Path, domains: List[str] = ["package"]):
        self.domains = domains
        self.rapl_root = rapl_root
        self.start_uj: Dict[str, int] = {}
        self.end_uj: Dict[str, int] = {}
        self.max_range_uj: Dict[str, int] = {}
        self.available = self.rapl_root.exists()

    def _read_int(self, p: Path) -> int:
        try:
            return int(p.read_text().strip())
        except PermissionError as e:
            raise PermissionError(f"RAPL permission denied reading {p}") from e

    def _iter_zones(self):
        # top-level zones: intel-rapl:0, intel-rapl:1, ...
        for zone in self.rapl_root.glob("intel-rapl:*"):
            # ignore subzones here
            if zone.name.count(":") != 1:
                continue
            yield zone

    def _collect_raw(self) -> Dict[str, int]:
        """
        Return energy_uj sums per domain across packages.
        Keys: "package", "dram" if requested and present.
        """
        out = {d: 0 for d in self.domains}
        if not self.available:
            return out

        for zone in self._iter_zones():
            # package energy
            if "package" in self.domains:
                e_uj = self._read_int(zone / "energy_uj")
                out["package"] += e_uj
                self.max_range_uj.setdefault("package", self._read_int(zone / "max_energy_range_uj"))

            # dram subzones (optional)
            if "dram" in self.domains:
                for sub in zone.glob("intel-rapl:*:*"):
                    name_file = sub / "name"
                    try:
                        name = name_file.read_text().strip().lower()
                    except Exception:
                        continue
                    if "dram" in name:
                        e_uj = self._read_int(sub / "energy_uj")
                        out["dram"] += e_uj
                        self.max_range_uj.setdefault("dram", self._read_int(sub / "max_energy_range_uj"))
        return out

    def start(self):
        if not self.available:
            return
        self.start_uj = self._collect_raw()

    def stop(self) -> Dict[str, float]:
        """
        Returns energy deltas in Joules per domain.
        """
        if not self.available:
            return {d: 0.0 for d in self.domains}
        self.end_uj = self._collect_raw()
        deltas_j: Dict[str, float] = {}
        for d in self.domains:
            s = self.start_uj.get(d, 0)
            e = self.end_uj.get(d, 0)
            mr = self.max_range_uj.get(d, 0)
            if s == 0 and e == 0:
                deltas_j[d] = 0.0
                continue
            if mr > 0 and e < s:
                # wrapped
                delta_uj = e + (mr - s)
            else:
                delta_uj = e - s
            deltas_j[d] = delta_uj / 1e6  # microjoules -> joules
        return deltas_j


class NVMLPoller:
    """Background thread for polling GPU power via NVML."""

    def __init__(self, poll_interval_ms: int = 500, device_indices: Optional[List[int]] = None):
        self.poll_interval_sec = poll_interval_ms / 1000.0
        self.device_indices = device_indices
        self.power_readings: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handles: List[Any] = []
        self._initialized = False
        self._lock = threading.Lock()

    def start(self):
        """Initialize NVML and start polling thread."""
        try:
            pynvml.nvmlInit()
            self._initialized = True

            # Get device handles
            device_count = pynvml.nvmlDeviceGetCount()
            if self.device_indices is None:
                self.device_indices = _infer_nvml_device_indices(device_count)

            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_indices]

            try:
                uuids = []
                for h in self._handles:
                    uuids.append(pynvml.nvmlDeviceGetUUID(h))
                print(f"[NVMLPoller] Polling GPUs: indices={self.device_indices}, uuids={uuids}")
            except Exception:
                pass

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

        except pynvml.NVMLError as e:
            print(f"Warning: NVML initialization failed: {e}")
            self._initialized = False

    def _poll_loop(self):
        """Polling loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                timestamp = time.time()
                powers = {}
                total_power = 0.0

                for idx, handle in zip(self.device_indices, self._handles):
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    powers[f"gpu_{idx}"] = power_w
                    total_power += power_w

                powers["timestamp"] = timestamp
                powers["total_power_w"] = total_power
                with self._lock:
                    self.power_readings.append(powers)

            except pynvml.NVMLError:
                pass  # Skip failed readings

            self._stop_event.wait(self.poll_interval_sec)

    def stop(self) -> List[Dict[str, float]]:
        """Stop polling and return collected readings."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            readings = self.power_readings.copy()
            self.power_readings.clear()
        return readings

    def get_current_readings(self) -> List[Dict[str, float]]:
        """Get readings collected so far without stopping."""
        with self._lock:
            return self.power_readings.copy()


class EnergyTracker:
    """
    Unified energy tracker for distillation experiments.

    Wraps CodeCarbon, NVML polling, and optional RAPL CPU readings for
    energy measurement with stage-wise accounting.

    Usage:
        tracker = EnergyTracker(run_dir, config=config)

        tracker.start_stage("teacher_forward")
        # ... do work ...
        tracker.end_stage(tokens_processed=1000)

        tracker.start_stage("student_train")
        # ... do work ...
        tracker.end_stage(tokens_processed=5000)

        tracker.save_summary()
    """

    def __init__(
        self,
        run_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[Any] = None,
        nvml_poll_interval_ms: Optional[int] = None,
        track_cpu: Optional[bool] = None,
        country_iso_code: Optional[str] = None,
        offline_mode: Optional[bool] = None,
        rapl_root: Optional[str] = None,
    ):
        self.config = config
        self._stage_counts: Dict[str, int] = {}

        def _cfg(explicit: Any, attr_name: str, dotted: str, default: Any) -> Any:
            if explicit is not None:
                return explicit
            if self.config is not None:
                if hasattr(self.config, attr_name):
                    val = getattr(self.config, attr_name)
                    if val is not None:
                        return val
                if hasattr(self.config, "get"):
                    nested_val = self.config.get(dotted, None)
                    if nested_val is not None:
                        return nested_val
            return default

        self.experiment_name = _cfg(experiment_name, "experiment_name", "experiment.name", "experiment")
        self.nvml_poll_interval_ms = _cfg(nvml_poll_interval_ms, "energy_nvml_poll_ms", "energy.nvml_poll_interval_ms", 500)
        self.country_iso_code = _cfg(country_iso_code, "energy_country_iso", "energy.country_iso_code", "USA")
        self.offline_mode = _cfg(offline_mode, "energy_offline_mode", "energy.offline_mode", True)
        self.track_cpu = _cfg(track_cpu, "energy_track_cpu", "energy.track_cpu", True)
        rapl_root_val = _cfg(rapl_root, "energy_rapl_root", "energy.rapl_root", "/sys/class/powercap/intel-rapl")
        self.rapl_root = Path(rapl_root_val)
        self.total_energy_policy = _cfg(None, "energy_total_policy", "energy.total_energy_policy", "measured")

        self.config = config

        # run directories
        resolved_run_dir = _cfg(run_dir, "run_dir", "output.run_dir", None)
        if not resolved_run_dir:
            raise ValueError("EnergyTracker requires run_dir (preferred) or output_dir.")
        self.run_dir = Path(resolved_run_dir)

        self.energy_root = self.run_dir
        self.stages_dir = self.energy_root / "stages"
        self.codecarbon_dir = self.energy_root / "codecarbon"
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.codecarbon_dir.mkdir(parents=True, exist_ok=True)

        # Stage tracking
        self.stages: Dict[str, StageMetrics] = {}
        self.current_stage: Optional[str] = None
        self._stage_dirs: Dict[str, Path] = {}

        # Tool instances (created per-stage)
        self._codecarbon_tracker: Optional[EmissionsTracker] = None
        self._nvml_poller: Optional[NVMLPoller] = None
        self._rapl_reader = RAPLReader(rapl_root=self.rapl_root, domains=["package"]) if self.track_cpu else None

        # Experiment metadata
        self.start_time = datetime.now().isoformat()
        self.experiment_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Register cleanup
        atexit.register(self._cleanup)

    def start_stage(self, stage_name: str):
        """Start tracking a new stage."""
        if self.current_stage is not None:
            print(f"Warning: Stage '{self.current_stage}' still running. Ending it first.")
            self.end_stage()

        # Include experiment name so stage labels stay unique across configs
        stage_full_name = f"{stage_name}_{self.experiment_name}" if self.experiment_name else stage_name

        n = self._stage_counts.get(stage_full_name, 0) + 1
        self._stage_counts[stage_full_name] = n
        stage_id = stage_full_name if n == 1 else f"{stage_full_name}__{n}"

        self.current_stage = stage_id
        stage_metrics = StageMetrics(stage_id=stage_id, stage_name=stage_full_name)
        self.stages[stage_id] = stage_metrics

        # Prepare per-stage folder under stages/
        safe_stage_name = _safe_filename(stage_full_name)
        safe_stage_id = _safe_filename(stage_id)
        stage_dir = self.stages_dir / safe_stage_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._stage_dirs[stage_id] = stage_dir

        # Snapshot environment and config for this stage
        try:
            env_filename = f"environment_{safe_stage_name}__{safe_stage_id}.json"
            save_environment(output_dir=str(stage_dir), filename=env_filename)
        except Exception as e:
            print(f"[EnergyTracker] Warning: failed to save environment for stage '{stage_id}': {e}")

        if self.config is not None:
            try:
                cfg_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
                cfg_payload = {
                    "stage_name": stage_full_name,
                    "stage_id": stage_id,
                    "config": cfg_dict,
                }
                cfg_filename = f"config_{safe_stage_name}__{safe_stage_id}.json"
                _write_json(stage_dir / cfg_filename, cfg_payload)
            except Exception as e:
                print(f"[EnergyTracker] Warning: failed to save config for stage '{stage_id}': {e}")

        # RAPL CPU energy: use it if available, otherwise fall back to CodeCarbon
        if self.track_cpu and self._rapl_reader is not None:
            if not self._rapl_reader.available:
                print("[EnergyTracker] RAPL not available; will rely on CodeCarbon CPU estimates.")
                self._rapl_reader = None
            else:
                try:
                    self._rapl_reader.start()
                except PermissionError as e:
                    print(f"[EnergyTracker] RAPL permission denied; will rely on CodeCarbon CPU estimates. ({e})")
                    self._rapl_reader = None
                except Exception as e:
                    print(f"[EnergyTracker] RAPL start failed; will rely on CodeCarbon CPU estimates. ({e})")
                    self._rapl_reader = None

        # Start CodeCarbon
        try:
            project_name = f"{self.experiment_name}_{stage_id}"
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if self.offline_mode:
                self._codecarbon_tracker = OfflineEmissionsTracker(
                    project_name=project_name,
                    output_dir=str(self.codecarbon_dir),
                    country_iso_code=self.country_iso_code,
                    log_level="error",
                )
            else:
                self._codecarbon_tracker = EmissionsTracker(
                    project_name=project_name,
                    output_dir=str(self.codecarbon_dir),
                    log_level="error",
                )
            self._codecarbon_tracker.start()
        except Exception as e:
            print(f"Warning: CodeCarbon start failed: {e}")
            self._codecarbon_tracker = None

        # Sync GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Start NVML polling
        self._nvml_poller = NVMLPoller(poll_interval_ms=self.nvml_poll_interval_ms)
        self._nvml_poller.start()

        stage_metrics.start_time = time.time()

        print(f"[EnergyTracker] Started stage: {stage_id}")

    def end_stage(self, tokens_processed: Optional[int] = None) -> StageMetrics:
        """End current stage and collect metrics."""
        if self.current_stage is None:
            raise RuntimeError("No stage is currently running")

        # Sync GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        stage_id = self.current_stage
        stage_metrics = self.stages[stage_id]
        if tokens_processed is not None:
            stage_metrics.tokens_processed = tokens_processed
        stage_metrics.end_time = time.time()
        stage_metrics.duration_seconds = stage_metrics.end_time - stage_metrics.start_time

        # Stop NVML and compute GPU energy
        if self._nvml_poller is not None:
            readings = self._nvml_poller.stop()
            self._nvml_poller = None

            if readings:
                readings.sort(key=lambda r: r["timestamp"])
                stage_metrics.gpu_power_samples = [r["total_power_w"] for r in readings]

                # Integrate power over the exact stage window [start_time, end_time]
                start_t = stage_metrics.start_time
                end_t = stage_metrics.end_time

                def power_at(t: float) -> float:
                    """
                    Sample-and-hold power at time t using the last reading with timestamp <= t.
                    If t is before the first reading, use the first reading's power.
                    """
                    p = readings[0]["total_power_w"]
                    for r in readings:
                        if r["timestamp"] <= t:
                            p = r["total_power_w"]
                        else:
                            break
                    return p

                # Build a window aligned to stage boundaries to avoid edge undercount
                window = [{"timestamp": start_t, "total_power_w": power_at(start_t)}]
                for r in readings:
                    ts = r["timestamp"]
                    if start_t < ts < end_t:
                        window.append({"timestamp": ts, "total_power_w": r["total_power_w"]})
                window.append({"timestamp": end_t, "total_power_w": power_at(end_t)})

                # Trapezoidal integration over the padded window
                e_j = 0.0
                for a, b in zip(window, window[1:]):
                    p0, t0 = a["total_power_w"], a["timestamp"]
                    p1, t1 = b["total_power_w"], b["timestamp"]
                    dt = max(0.0, t1 - t0)
                    e_j += 0.5 * (p0 + p1) * dt

                stage_metrics.gpu_energy_joules = e_j

                # Avg/peak power (now duration + energy exist)
                if stage_metrics.duration_seconds > 0:
                    stage_metrics.gpu_avg_power_watts = stage_metrics.gpu_energy_joules / stage_metrics.duration_seconds
                stage_metrics.gpu_peak_power_watts = max(stage_metrics.gpu_power_samples)

        # Stop CodeCarbon
        if self._codecarbon_tracker is not None:
            codecarbon_dir = self.energy_root / "codecarbon" / stage_id
            try:
                emissions = self._codecarbon_tracker.stop()
                if emissions is not None:
                    stage_metrics.codecarbon_emissions_kg = float(emissions)
            except Exception as e:
                print(f"Warning: CodeCarbon stop failed: {e}")
            finally:
                self._codecarbon_tracker = None

            # Read energy from emissions.csv
            stage_metrics.total_codecarbon_energy_kwh = self._read_total_codecarbon_energy_kwh(
                self.codecarbon_dir,
                project_name=f"{self.experiment_name}_{stage_id}",
            )

        # RAPL CPU energy
        if self.track_cpu and self._rapl_reader is not None:
            try:
                deltas = self._rapl_reader.stop()
                pkg_j = deltas.get("package", 0.0)
                stage_metrics.cpu_energy_joules = pkg_j
            except Exception as e:
                print(f"[EnergyTracker] RAPL stop failed: {e}")

        # Compute derived metrics
        stage_metrics.compute_derived_metrics(total_energy_policy=self.total_energy_policy)

        # Save stage metrics to JSON
        safe_stage_id = _safe_filename(stage_id)
        stage_dir = self._stage_dirs.get(stage_id, self.stages_dir / safe_stage_id)
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_path = stage_dir / f"{safe_stage_id}.json"
        stage_path_root = self.stages_dir / f"{safe_stage_id}.json"
        payload = stage_metrics.to_dict(include_power_samples=True)
        payload["nvml_poll_interval_ms"] = self.nvml_poll_interval_ms
        _write_json(stage_path, payload)
        if stage_path_root != stage_path:
            _write_json(stage_path_root, payload)

        print(f"[EnergyTracker] Ended stage: {stage_id}")
        print(f"  Duration: {stage_metrics.duration_seconds:.2f}s")
        print(f"  GPU Energy: {stage_metrics.gpu_energy_joules:.2f} J")
        print(f"  Tokens: {stage_metrics.tokens_processed}, Joules/token: {stage_metrics.joules_per_token:.4f}")

        self.current_stage = None
        return stage_metrics

    def add_tokens(self, count: int):
        """Add tokens to current stage's count (for incremental updates)."""
        if self.current_stage and self.current_stage in self.stages:
            self.stages[self.current_stage].tokens_processed += count

    def get_stage_metrics(self, stage_name: str) -> Optional[StageMetrics]:
        """Get metrics for a specific stage."""
        return self.stages.get(stage_name)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all stages."""
        total_gpu_joules = sum(s.gpu_energy_joules for s in self.stages.values())
        total_cpu_joules = sum(s.cpu_energy_joules for s in self.stages.values())
        total_energy_joules = sum(s.total_energy_joules for s in self.stages.values())
        total_tokens = sum(s.tokens_processed for s in self.stages.values())
        total_duration = sum(s.duration_seconds for s in self.stages.values())

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "total_tokens_processed": total_tokens,
            "total_gpu_energy_joules": total_gpu_joules,
            "total_cpu_energy_joules": total_cpu_joules,
            "total_energy_joules": total_energy_joules,
            "total_energy_kwh": total_energy_joules / 3_600_000 if total_energy_joules > 0 else 0.0,
            "overall_joules_per_token": total_energy_joules / total_tokens if total_tokens > 0 else 0.0,
            "overall_tokens_per_second": total_tokens / total_duration if total_duration > 0 else 0.0,
            "stages": {name: metrics.to_dict() for name, metrics in self.stages.items()},
        }

    def save_summary(self, additional_metadata: Optional[Dict] = None) -> Path:
        """Save complete experiment summary to JSON."""
        summary = self.get_summary()
        if additional_metadata:
            summary["metadata"] = additional_metadata

        summary_file = self.energy_root / "experiment_summary.json"
        _write_json(summary_file, summary)
        for stage_id in self.stages.keys():
            safe_stage_id = _safe_filename(stage_id)
            stage_dir = self._stage_dirs.get(stage_id, self.stages_dir / safe_stage_id)
            stage_dir.mkdir(parents=True, exist_ok=True)
            _write_json(stage_dir / "experiment_summary.json", summary)
        print(f"[EnergyTracker] Summary saved to: {summary_file}")
        return summary_file

    def get_wandb_metrics(self, prefix: str = "energy") -> Dict[str, float]:
        metrics = {}
        for stage_name, stage in self.stages.items():
            stage_prefix = f"{prefix}/{stage_name}"
            metrics[f"{stage_prefix}/duration_sec"] = stage.duration_seconds
            metrics[f"{stage_prefix}/gpu_energy_joules"] = stage.gpu_energy_joules
            metrics[f"{stage_prefix}/cpu_energy_joules"] = stage.cpu_energy_joules
            metrics[f"{stage_prefix}/total_energy_joules"] = stage.total_energy_joules
            metrics[f"{stage_prefix}/gpu_avg_power_watts"] = stage.gpu_avg_power_watts
            metrics[f"{stage_prefix}/tokens_processed"] = stage.tokens_processed
            metrics[f"{stage_prefix}/joules_per_token"] = stage.joules_per_token
            metrics[f"{stage_prefix}/tokens_per_second"] = stage.tokens_per_second

        summary = self.get_summary()
        metrics[f"{prefix}/total_gpu_energy_joules"] = summary["total_gpu_energy_joules"]
        metrics[f"{prefix}/total_cpu_energy_joules"] = summary["total_cpu_energy_joules"]
        metrics[f"{prefix}/total_energy_joules"] = summary["total_energy_joules"]
        metrics[f"{prefix}/total_energy_kwh"] = summary["total_energy_kwh"]
        metrics[f"{prefix}/total_tokens"] = summary["total_tokens_processed"]
        metrics[f"{prefix}/overall_joules_per_token"] = summary["overall_joules_per_token"]

        return metrics

    def _read_total_codecarbon_energy_kwh(self, codecarbon_dir: Path, project_name: str) -> float:
        emissions_csv = codecarbon_dir / "emissions.csv"
        if not emissions_csv.exists():
            candidates = [p for p in codecarbon_dir.glob("*.csv") if p.name.startswith("emissions")]
            if candidates:
                emissions_csv = sorted(candidates)[-1]
            else:
                return 0.0

        for _ in range(5):
            try:
                last_match = None
                with open(emissions_csv, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row and row.get("project_name") == project_name:
                            last_match = row
                if last_match:
                    val = last_match.get("energy_consumed", "")
                    return float(val) if val not in (None, "") else 0.0
            except Exception:
                pass
            time.sleep(0.1)

        return 0.0

    def _cleanup(self):
        """Cleanup resources on exit."""
        if self.current_stage is not None:
            try:
                self.end_stage()
            except Exception:
                pass

        if self._nvml_poller is not None:
            try:
                self._nvml_poller.stop()
            except Exception:
                pass

        if self._codecarbon_tracker is not None:
            try:
                self._codecarbon_tracker.stop()
            except Exception:
                pass

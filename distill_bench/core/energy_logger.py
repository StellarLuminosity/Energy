import os
import json
import time
import threading
import atexit
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import pynvml


@dataclass
class StageMetrics:
    """Metrics collected for a single stage of the pipeline."""
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
    codecarbon_energy_kwh: float = 0.0
    codecarbon_emissions_kg: float = 0.0
    
    # Derived metrics
    joules_per_token: float = 0.0
    kwh_total: float = 0.0
    tokens_per_second: float = 0.0
    
    def compute_derived_metrics(self):
        """Compute derived metrics after stage completion."""
        self.duration_seconds = self.end_time - self.start_time
        
        if self.duration_seconds > 0 and self.tokens_processed > 0:
            self.tokens_per_second = self.tokens_processed / self.duration_seconds
        
        # Use GPU energy as primary source, fall back to CodeCarbon
        total_joules = self.gpu_energy_joules
        if total_joules == 0 and self.codecarbon_energy_kwh > 0:
            total_joules = self.codecarbon_energy_kwh * 3_600_000  # kWh to joules
        
        if self.tokens_processed > 0 and total_joules > 0:
            self.joules_per_token = total_joules / self.tokens_processed
        
        # Combine energy sources for total kWh
        self.kwh_total = max(
            self.codecarbon_energy_kwh,
            self.gpu_energy_joules / 3_600_000  # joules to kWh
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding raw power samples for summary."""
        d = asdict(self)
        d.pop('gpu_power_samples', None)  # Exclude raw samples from summary
        return d


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
        
    def start(self):
        """Initialize NVML and start polling thread."""
        try:
            pynvml.nvmlInit()
            self._initialized = True
            
            # Get device handles
            device_count = pynvml.nvmlDeviceGetCount()
            if self.device_indices is None:
                self.device_indices = list(range(device_count))
            
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) 
                for i in self.device_indices
            ]
            
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
        
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._initialized = False
        
        readings = self.power_readings.copy()
        self.power_readings.clear()
        return readings
    
    def get_current_readings(self) -> List[Dict[str, float]]:
        """Get readings collected so far without stopping."""
        return self.power_readings.copy()


class EnergyTracker:
    """
    Unified energy tracker for distillation experiments.
    
    Wraps CodeCarbon, experiment-impact-tracker, and NVML for comprehensive
    energy measurement with stage-wise accounting.
    
    Usage:
        tracker = EnergyTracker(output_dir, config)
        
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
        output_dir: str,
        experiment_name: str = "experiment",
        nvml_poll_interval_ms: int = 500,
        track_cpu: bool = True,
        country_iso_code: str = "USA",
        offline_mode: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.nvml_poll_interval_ms = nvml_poll_interval_ms
        self.track_cpu = track_cpu
        self.country_iso_code = country_iso_code
        self.offline_mode = offline_mode
        
        # Create output directories
        self.energy_dir = self.output_dir / "energy_logs"
        self.energy_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage tracking
        self.stages: Dict[str, StageMetrics] = {}
        self.current_stage: Optional[str] = None
        self._stage_start_readings: List[Dict] = []
        
        # Tool instances (created per-stage)
        self._codecarbon_tracker: Optional[EmissionsTracker] = None
        self._nvml_poller: Optional[NVMLPoller] = None
        
        # Experiment metadata
        self.start_time = datetime.now().isoformat()
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    def start_stage(self, stage_name: str):
        """Start tracking a new stage."""
        if self.current_stage is not None:
            print(f"Warning: Stage '{self.current_stage}' still running. Ending it first.")
            self.end_stage()
        
        self.current_stage = stage_name
        stage_metrics = StageMetrics(stage_name=stage_name)
        stage_metrics.start_time = time.time()
        self.stages[stage_name] = stage_metrics
        
        # Start CodeCarbon
        try:
            codecarbon_dir = self.energy_dir / "codecarbon" / stage_name
            codecarbon_dir.mkdir(parents=True, exist_ok=True)
            
            if self.offline_mode:
                self._codecarbon_tracker = OfflineEmissionsTracker(
                    project_name=f"{self.experiment_name}_{stage_name}",
                    output_dir=str(codecarbon_dir),
                    country_iso_code=self.country_iso_code,
                    log_level="warning",
                )
            else:
                self._codecarbon_tracker = EmissionsTracker(
                    project_name=f"{self.experiment_name}_{stage_name}",
                    output_dir=str(codecarbon_dir),
                    log_level="warning",
                )
            self._codecarbon_tracker.start()
        except Exception as e:
            print(f"Warning: CodeCarbon start failed: {e}")
            self._codecarbon_tracker = None
        
        # Start NVML polling
        self._nvml_poller = NVMLPoller(poll_interval_ms=self.nvml_poll_interval_ms)
        self._nvml_poller.start()
        self._stage_start_readings = []
        
        # Sync GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"[EnergyTracker] Started stage: {stage_name}")
    
    def end_stage(self, tokens_processed: int = 0) -> StageMetrics:
        """End current stage and collect metrics."""
        if self.current_stage is None:
            raise RuntimeError("No stage is currently running")
        
        # Sync GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        stage_name = self.current_stage
        stage_metrics = self.stages[stage_name]
        stage_metrics.end_time = time.time()
        stage_metrics.tokens_processed = tokens_processed
        
        # Stop NVML and compute GPU energy
        if self._nvml_poller is not None:
            readings = self._nvml_poller.stop()
            if readings:
                powers = [r["total_power_w"] for r in readings]
                stage_metrics.gpu_power_samples = powers
                stage_metrics.gpu_avg_power_watts = sum(powers) / len(powers)
                stage_metrics.gpu_peak_power_watts = max(powers)
                
                # Compute energy: integrate power over time
                # Each reading is poll_interval apart
                interval_sec = self.nvml_poll_interval_ms / 1000.0
                stage_metrics.gpu_energy_joules = sum(powers) * interval_sec
            self._nvml_poller = None
        
        # Stop CodeCarbon
        if self._codecarbon_tracker is not None:
            try:
                emissions = self._codecarbon_tracker.stop()
                if emissions is not None:
                    stage_metrics.codecarbon_emissions_kg = emissions
                # Read energy from tracker's internal state
                if hasattr(self._codecarbon_tracker, '_total_energy'):
                    stage_metrics.codecarbon_energy_kwh = self._codecarbon_tracker._total_energy.kWh
            except Exception as e:
                print(f"Warning: CodeCarbon stop failed: {e}")
            self._codecarbon_tracker = None
        
        # Compute derived metrics
        stage_metrics.compute_derived_metrics()
        
        # Save stage metrics to JSON
        stage_file = self.energy_dir / f"stage_{stage_name}.json"
        with open(stage_file, 'w') as f:
            json.dump(stage_metrics.to_dict(), f, indent=2)
        
        # Also save raw power samples if available
        if stage_metrics.gpu_power_samples:
            samples_file = self.energy_dir / f"stage_{stage_name}_power_samples.json"
            with open(samples_file, 'w') as f:
                json.dump({
                    "stage": stage_name,
                    "poll_interval_ms": self.nvml_poll_interval_ms,
                    "samples": stage_metrics.gpu_power_samples
                }, f)
        
        print(f"[EnergyTracker] Ended stage: {stage_name}")
        print(f"  Duration: {stage_metrics.duration_seconds:.2f}s")
        print(f"  GPU Energy: {stage_metrics.gpu_energy_joules:.2f} J")
        print(f"  Tokens: {tokens_processed}, Joules/token: {stage_metrics.joules_per_token:.4f}")
        
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
        total_energy_joules = sum(s.gpu_energy_joules for s in self.stages.values())
        total_tokens = sum(s.tokens_processed for s in self.stages.values())
        total_duration = sum(s.duration_seconds for s in self.stages.values())
        
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "total_tokens_processed": total_tokens,
            "total_energy_joules": total_energy_joules,
            "total_energy_kwh": total_energy_joules / 3_600_000,
            "overall_joules_per_token": total_energy_joules / total_tokens if total_tokens > 0 else 0,
            "overall_tokens_per_second": total_tokens / total_duration if total_duration > 0 else 0,
            "stages": {name: metrics.to_dict() for name, metrics in self.stages.items()},
        }
    
    def save_summary(self, additional_metadata: Optional[Dict] = None) -> Path:
        """Save complete experiment summary to JSON."""
        summary = self.get_summary()
        if additional_metadata:
            summary["metadata"] = additional_metadata
        
        summary_file = self.energy_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[EnergyTracker] Summary saved to: {summary_file}")
        return summary_file
    
    def get_wandb_metrics(self, prefix: str = "energy") -> Dict[str, float]:
        """Get metrics formatted for W&B logging."""
        metrics = {}
        
        for stage_name, stage in self.stages.items():
            stage_prefix = f"{prefix}/{stage_name}"
            metrics[f"{stage_prefix}/duration_sec"] = stage.duration_seconds
            metrics[f"{stage_prefix}/gpu_energy_joules"] = stage.gpu_energy_joules
            metrics[f"{stage_prefix}/gpu_avg_power_watts"] = stage.gpu_avg_power_watts
            metrics[f"{stage_prefix}/tokens_processed"] = stage.tokens_processed
            metrics[f"{stage_prefix}/joules_per_token"] = stage.joules_per_token
            metrics[f"{stage_prefix}/tokens_per_second"] = stage.tokens_per_second
        
        # Totals
        summary = self.get_summary()
        metrics[f"{prefix}/total_energy_joules"] = summary["total_energy_joules"]
        metrics[f"{prefix}/total_energy_kwh"] = summary["total_energy_kwh"]
        metrics[f"{prefix}/total_tokens"] = summary["total_tokens_processed"]
        metrics[f"{prefix}/overall_joules_per_token"] = summary["overall_joules_per_token"]
        
        return metrics
    
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

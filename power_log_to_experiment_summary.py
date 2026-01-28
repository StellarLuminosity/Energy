#!/usr/bin/env python3
"""
Reconstruct a minimal experiment_summary.json from power_log.csv.

Usage:
    python power_log_to_experiment_summary.py /path/to/power_log.csv \
        --stage-name kd_32b_to_7b_tulu \
        --experiment-name kd_32b_to_7b_tulu \
        --poll-interval-ms 500

If timestamps are present in the CSV, the script will try to infer the
poll interval; otherwise you *must* pass --poll-interval-ms.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def infer_power_column(df: pd.DataFrame) -> str:
    """Guess which column holds GPU power in Watts."""
    candidates = ["power_watts", "gpu_power_watts", "power", "gpu_power"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a power column in {list(df.columns)}; tried {candidates}")


def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    """Guess which column (if any) holds timestamps in seconds."""
    candidates = ["timestamp", "time", "t", "seconds"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_energy_from_power(
    power: np.ndarray,
    time: Optional[np.ndarray],
    poll_interval_ms: Optional[float],
) -> tuple[float, float, float, float]:
    """
    Compute duration, average power, peak power, and energy in Joules
    from power samples and optional timestamps.

    Returns:
        duration_seconds, gpu_avg_power_watts, gpu_peak_power_watts, gpu_energy_joules
    """
    if power.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    gpu_peak = float(np.max(power))

    if time is not None and time.size == power.size:
        # Assume time is in seconds; use trapezoidal rule for energy
        time = time.astype(float)
        duration = float(time[-1] - time[0]) if time.size > 1 else 0.0
        if time.size > 1:
            energy_j = float(np.trapz(power, time))
        else:
            # Single sample: approximate energy as power * (poll_interval or 0)
            dt = poll_interval_ms / 1000.0 if poll_interval_ms is not None else 0.0
            energy_j = float(power[0] * dt)
    else:
        # No usable timestamps; fall back to poll_interval_ms
        if poll_interval_ms is None:
            raise ValueError(
                "No time column and no --poll-interval-ms specified; cannot compute energy."
            )
        dt = poll_interval_ms / 1000.0
        duration = float(power.size * dt)
        energy_j = float(power.sum() * dt)

    gpu_avg = float(power.mean()) if duration > 0 else float(power.mean())
    return duration, gpu_avg, gpu_peak, energy_j


def build_stage_metrics_from_power_log(
    power_log_path: Path,
    stage_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    poll_interval_ms: Optional[float] = None,
) -> dict:
    """Build a minimal experiment_summary-like dict from a single power_log.csv."""
    df = pd.read_csv(power_log_path)

    power_col = infer_power_column(df)
    time_col = infer_time_column(df)

    power = df[power_col].to_numpy(dtype=float)
    time = df[time_col].to_numpy(dtype=float) if time_col is not None else None

    # If we have timestamps and no poll_interval, infer it from median delta
    if poll_interval_ms is None and time is not None and time.size > 1:
        dt_est = float(np.median(np.diff(time)))
        poll_interval_ms = dt_est * 1000.0

    duration_s, gpu_avg_w, gpu_peak_w, gpu_energy_j = compute_energy_from_power(
        power, time, poll_interval_ms
    )

    total_energy_j = gpu_energy_j
    total_energy_kwh = total_energy_j / 3.6e6

    stage_name = stage_name or power_log_path.parent.name
    experiment_name = experiment_name or stage_name

    # Python None -> JSON null
    stage_metrics = {
        "stage_id": stage_name,
        "stage_name": stage_name,
        # Timing (we can't reliably recover absolute wall-clock times)
        "start_time": None,
        "end_time": None,
        "duration_seconds": duration_s,
        # Tokens / throughput (unknown from power_log alone)
        "tokens_processed": None,
        "tokens_per_second": None,
        "joules_per_token": None,
        # GPU power/energy
        "gpu_power_samples": power.tolist(),
        "nvml_poll_interval_ms": poll_interval_ms,
        "gpu_avg_power_watts": gpu_avg_w,
        "gpu_peak_power_watts": gpu_peak_w,
        "gpu_energy_joules": gpu_energy_j,
        # CPU + total energy; we only know GPU, so treat total as GPU-only
        "cpu_energy_joules": None,
        "total_energy_joules": total_energy_j,
        "total_energy_kwh": total_energy_kwh,
        "kwh_total": total_energy_kwh,
        # CodeCarbon-derived fields – unknown
        "total_codecarbon_energy_kwh": None,
        "codecarbon_emissions_kg": None,
        "codecarbon_cpu_energy_kwh": None,
        "codecarbon_gpu_energy_kwh": None,
        "codecarbon_ram_energy_kwh": None,
        # Provenance / snapshot flags
        "source": "power_log_reconstructed",
        "is_snapshot": False,
        "snapshot_step": None,
        "snapshot_type": None,
        "snapshot_time": None,
    }

    return {
        "experiment_id": experiment_name,
        "experiment_name": experiment_name,
        "stages": {
            stage_name: stage_metrics
        },
    }


def write_experiment_summary_from_power_log(
    power_log_path: Path,
    stage_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    poll_interval_ms: Optional[float] = None,
    summary_filename: str = "experiment_summary.json",
) -> Path:
    """Create or update experiment_summary.json next to power_log.csv."""
    summary = build_stage_metrics_from_power_log(
        power_log_path,
        stage_name=stage_name,
        experiment_name=experiment_name,
        poll_interval_ms=poll_interval_ms,
    )

    out_path = power_log_path.parent / summary_filename

    # If an experiment_summary.json already exists, merge stages
    if out_path.exists():
        with out_path.open("r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        stages = existing.get("stages", {})
        stages.update(summary["stages"])
        existing["stages"] = stages
        existing.setdefault("experiment_id", summary["experiment_id"])
        existing.setdefault("experiment_name", summary["experiment_name"])
        summary = existing

    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] Wrote {out_path} with stage '{stage_name or power_log_path.parent.name}'")
    return out_path


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct a minimal experiment_summary.json from power_log.csv"
    )
    parser.add_argument("power_log", help="Path to power_log.csv")
    parser.add_argument("--stage-name", help="Stage name / id (defaults to parent directory name)")
    parser.add_argument("--experiment-name", help="Experiment name / id (defaults to stage name)")
    parser.add_argument(
        "--poll-interval-ms",
        type=float,
        default=None,
        help="NVML polling interval in milliseconds (optional; inferred from timestamps if possible)",
    )
    parser.add_argument(
        "--summary-filename",
        default="experiment_summary.json",
        help="Name of the summary file to write (default: experiment_summary.json)",
    )

    args = parser.parse_args(argv[1:])

    power_log_path = Path(args.power_log).expanduser().resolve()
    if not power_log_path.exists():
        print(f"ERROR: {power_log_path} does not exist", file=sys.stderr)
        return 1

    write_experiment_summary_from_power_log(
        power_log_path,
        stage_name=args.stage_name,
        experiment_name=args.experiment_name,
        poll_interval_ms=args.poll_interval_ms,
        summary_filename=args.summary_filename,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

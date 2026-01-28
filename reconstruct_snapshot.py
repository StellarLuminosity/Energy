import json
import csv
from pathlib import Path

# ==========================
# CONFIG: SET THIS PATH
# ==========================
RUN_DIR = Path("/home/klambert/projects/aip-craffel/klambert/Energy/logs/stages/kd_olmo2_32b_to_1b_nosft")

OUTPUT_NAME = "experiment_summary.json"

def load_power_log(power_log_path: Path):
    """
    Reads power_log.csv and returns:
    - timestamps (seconds since epoch)
    - total_power_w (watts)
    """
    timestamps = []
    powers = []

    with open(power_log_path, "r") as f:
        reader = csv.DictReader(f)
        if "timestamp" not in reader.fieldnames or "total_power_w" not in reader.fieldnames:
            raise ValueError(
                f"power_log.csv must have 'timestamp' and 'total_power_w' columns; got {reader.fieldnames}"
            )
        for row in reader:
            ts = float(row["timestamp"])
            p = float(row["total_power_w"])
            timestamps.append(ts)
            powers.append(p)

    if len(timestamps) < 2:
        raise ValueError("Not enough power samples to integrate energy (need >= 2 rows).")

    return timestamps, powers


def integrate_energy_joules(timestamps, powers):
    """
    Trapezoidal integration of power over time.
    Power in W, time in s → Joules.
    """
    energy_j = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt < 0:
            # If timestamps are out of order, skip this interval
            continue
        avg_p = 0.5 * (powers[i] + powers[i - 1])
        energy_j += avg_p * dt
    return energy_j


def load_latest_snapshot(run_dir: Path):
    """
    Finds the latest *_step_*.json snapshot and returns (path, data).
    """
    snapshots = sorted(run_dir.glob("*_step_*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No step snapshot files found in {run_dir}")

    latest = snapshots[-1]
    with open(latest, "r") as f:
        data = json.load(f)

    return latest, data


def main():
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    power_log = run_dir / "power_log.csv"
    if not power_log.exists():
        raise FileNotFoundError(f"power_log.csv not found in {run_dir}")

    # --- Power integration ---
    timestamps, powers = load_power_log(power_log)
    gpu_energy_joules = integrate_energy_joules(timestamps, powers)

    start_ts = timestamps[0]
    end_ts = timestamps[-1]
    duration_seconds = end_ts - start_ts

    # Convert to ISO timestamps (to match energy_logger summary style)
    start_time_iso = datetime.fromtimestamp(start_ts).isoformat()
    end_time_iso = datetime.fromtimestamp(end_ts).isoformat()

    # --- Tokens from latest snapshot ---
    latest_snapshot_path, latest_snapshot = load_latest_snapshot(run_dir)
    tokens_processed = latest_snapshot.get("tokens_processed")
    stage_id = latest_snapshot.get("stage_id", run_dir.name)
    stage_name = latest_snapshot.get("stage_name", run_dir.name)

    if tokens_processed is None:
        raise KeyError(
            f"'tokens_processed' not found in latest snapshot {latest_snapshot_path.name}"
        )

    # --- Derived metrics ---
    total_gpu_energy_joules = gpu_energy_joules
    total_cpu_energy_joules = 0.0  # we don't reconstruct CPU energy here
    total_energy_joules = total_gpu_energy_joules + total_cpu_energy_joules
    total_energy_kwh = total_energy_joules / 3_600_000.0 if total_energy_joules > 0 else 0.0

    tokens_per_second = (
        tokens_processed / duration_seconds if duration_seconds > 0 else 0.0
    )
    joules_per_token = (
        total_energy_joules / tokens_processed if tokens_processed > 0 else 0.0
    )

    # --- Stage-level metrics payload (similar to StageMetrics.to_dict) ---
    stage_metrics = {
        "stage_id": stage_id,
        "stage_name": stage_name,
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_seconds": duration_seconds,
        "tokens_processed": tokens_processed,
        "gpu_energy_joules": total_gpu_energy_joules,
        "gpu_avg_power_watts": sum(powers) / len(powers),
        "gpu_peak_power_watts": max(powers),
        "gpu_power_samples": powers,  # optional but nice to keep
        # CodeCarbon-like fields: we don't reconstruct here, so set to 0.
        "total_codecarbon_energy_kwh": 0.0,
        "codecarbon_emissions_kg": 0.0,
        "codecarbon_cpu_energy_kwh": 0.0,
        "codecarbon_gpu_energy_kwh": 0.0,
        "codecarbon_ram_energy_kwh": 0.0,
        "cpu_energy_joules": total_cpu_energy_joules,
        "total_energy_joules": total_energy_joules,
        "total_energy_kwh": total_energy_kwh,
        "joules_per_token": joules_per_token,
        "kwh_total": total_energy_kwh,
        "tokens_per_second": tokens_per_second,
        "nvml_poll_interval_ms": latest_snapshot.get("nvml_poll_interval_ms", 500),
        "snapshot": False,
        "snapshot_type": "reconstructed",
        "snapshot_step": latest_snapshot.get("snapshot_step", None),
        "snapshot_time": datetime.now().isoformat(),
    }

    # --- Build experiment_summary.json (compatible with energy_logger.get_summary) ---
    summary = {
        "experiment_id": run_dir.name,
        "experiment_name": run_dir.name,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "total_duration_seconds": duration_seconds,
        "total_tokens_processed": tokens_processed,
        "total_gpu_energy_joules": total_gpu_energy_joules,
        "total_cpu_energy_joules": total_cpu_energy_joules,
        "total_energy_joules": total_energy_joules,
        "total_energy_kwh": total_energy_kwh,
        "overall_joules_per_token": joules_per_token,
        "overall_tokens_per_second": tokens_per_second,
        "stages": {
            stage_name: stage_metrics
        },
    }

    out_path = run_dir / OUTPUT_NAME
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote reconstructed {out_path}")
    print(f"     Tokens: {tokens_processed:,}")
    print(f"     Energy: {total_energy_kwh:.3f} kWh")
    print(f"     J/token: {joules_per_token:.4f}")

    # --- Delete snapshot files ( *_step_*.json ) ---
    snapshots = sorted(run_dir.glob("*_step_*.json"))
    if snapshots:
        print("\n[INFO] Deleting snapshot files:")
        for snap_path in snapshots:
            print(f"  - {snap_path.name}")
            try:
                snap_path.unlink()
            except Exception as e:
                print(f"    ! Could not delete {snap_path}: {e}")
    else:
        print("\n[INFO] No snapshot files found to delete.")

    print("\n[DONE] Reconstruction + cleanup complete.")


if __name__ == "__main__":
    main()

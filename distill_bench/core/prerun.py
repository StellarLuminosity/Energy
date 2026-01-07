"""
Pre-run validation and calibration for energy benchmarking experiments.

This module provides checks to ensure:
1. Idle baseline power measurement for net energy calculation
2. Burn-in testing to verify energy logging works correctly
3. Sampling interval validation to ensure accurate measurements
4. Hardware assertions to verify expected configuration

These checks help catch issues before running expensive, long experiments.
"""

import os
import json
import time
import warnings
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import pynvml

from .energy_logger import EnergyTracker, NVMLPoller
from .environment import collect_environment, collect_gpu_info
from .utils import _write_json

# Heuristic thresholds for idle baseline stability
IDLE_MAX_STD_WATTS = 1.0  # std dev <= 1 W
IDLE_MAX_SPIKE_OVER_MEAN_WATTS = 10.0  # max - mean <= 10 W


@dataclass
class IdleBaseline:
    """Idle power baseline measurements."""

    duration_seconds: float
    gpu_avg_power_watts: float
    gpu_min_power_watts: float
    gpu_max_power_watts: float
    gpu_std_power_watts: float
    stable: bool = True
    stability_reason: str = "ok"
    cpu_baseline_watts: Optional[float] = None  # If available from other tools
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BurnInResult:
    """Results from burn-in test."""

    num_steps: int
    duration_seconds: float
    energy_logged: bool
    gpu_energy_joules: float
    avg_gpu_utilization_pct: float
    tokens_per_second: float
    energy_logs_valid: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingIntervalResult:
    """Results from sampling interval validation."""

    interval_1s_energy_joules: float
    interval_15s_energy_joules: float
    difference_percent: float
    converged: bool
    recommended_interval_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareValidation:
    """Hardware validation results."""

    expected_gpu_type: Optional[str]
    actual_gpu_type: str
    gpu_match: bool
    expected_vram_gb: Optional[float]
    actual_vram_gb: float
    vram_sufficient: bool
    gpu_count: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreRunReport:
    """Complete pre-run validation report."""

    timestamp: str
    idle_baseline: IdleBaseline
    burn_in: BurnInResult
    sampling_interval: SamplingIntervalResult
    hardware_validation: HardwareValidation
    all_checks_passed: bool
    critical_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "idle_baseline": self.idle_baseline.to_dict(),
            "burn_in": self.burn_in.to_dict(),
            "sampling_interval": self.sampling_interval.to_dict(),
            "hardware_validation": self.hardware_validation.to_dict(),
            "all_checks_passed": self.all_checks_passed,
            "critical_warnings": self.critical_warnings,
        }


def _bool_icon(ok: bool) -> str:
    """Return a short status string for booleans."""
    return "OK" if ok else "ATTN"


def _print_prerun_summary(report: PreRunReport, report_file: Path):
    """Pretty-print the critical metrics needed to decide readiness."""
    hardware = report.hardware_validation
    idle = report.idle_baseline
    burn = report.burn_in
    sampling = report.sampling_interval

    status_label = "READY" if report.all_checks_passed else "ATTENTION REQUIRED"

    print("\n" + "=" * 70)
    print("PRE-RUN VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Status : {status_label}")
    print(f"Report : {report_file}")

    # Hardware
    print("\nHardware")
    print(f"  GPUs          : {hardware.gpu_count} x {hardware.actual_gpu_type}")
    if hardware.expected_gpu_type:
        print(f"  Expected GPU  : {hardware.expected_gpu_type} | Match: {hardware.gpu_match}")
    print(f"  Min VRAM (GB) : {hardware.actual_vram_gb:.1f} | Sufficient: {hardware.vram_sufficient}")
    if hardware.warnings:
        print("  Warnings      :")
        for w in hardware.warnings:
            print(f"    - {w}")

    # Idle baseline
    idle_skipped = idle.duration_seconds == 0
    print("\nIdle Baseline")
    if idle_skipped:
        print("  Status        : skipped")
    else:
        print(f"  Duration (s)  : {idle.duration_seconds:.0f}")
        print(
            f"  Avg/Min/Max W : {idle.gpu_avg_power_watts:.2f} / {idle.gpu_min_power_watts:.2f} / {idle.gpu_max_power_watts:.2f}"
        )
        print(f"  Std Dev (W)   : {idle.gpu_std_power_watts:.2f}")
        print(f"  Stable        : {idle.stable} ({idle.stability_reason})")

    # Burn-in
    burn_skipped = burn.num_steps == 0
    print("\nBurn-in")
    if burn_skipped:
        print("  Status        : skipped")
    else:
        print(f"  Steps         : {burn.num_steps}, Duration: {burn.duration_seconds:.1f}s")
        print(f"  GPU Energy (J): {burn.gpu_energy_joules:.2f}")
        print(f"  Throughput    : {burn.tokens_per_second:.0f} tok/s | GPU util est: {burn.avg_gpu_utilization_pct:.1f}%")
        print(f"  Energy logs   : {_bool_icon(burn.energy_logs_valid)}")
        if burn.warnings:
            print("  Warnings      :")
            for w in burn.warnings:
                print(f"    - {w}")

    # Sampling interval
    sampling_skipped = sampling.interval_1s_energy_joules == 0 and sampling.interval_15s_energy_joules == 0
    print("\nSampling Interval")
    if sampling_skipped:
        print("  Status        : skipped")
    else:
        print(f"  Energy 1s/2s (J): {sampling.interval_1s_energy_joules:.2f} / {sampling.interval_15s_energy_joules:.2f}")
        print(f"  Diff Percent  : {sampling.difference_percent:.2f}%")
        print(f"  Converged     : {sampling.converged} | Recommended poll: {sampling.recommended_interval_ms} ms")

    # Critical warnings
    if report.critical_warnings:
        print("\nCRITICAL WARNINGS")
        for w in report.critical_warnings:
            print(f"  - {w}")
    else:
        print("\nCritical Warnings: none")

    print("=" * 70)


def measure_idle_baseline(
    duration_minutes: float = 5.0,
    poll_interval_ms: int = 500,
    output_dir: Optional[str] = None,
) -> IdleBaseline:
    """
    Measure idle GPU power for baseline calculation.

    This helps calculate net energy by subtracting the idle power consumption
    from active training power consumption.
    """
    print("\n" + "=" * 70)
    print("IDLE BASELINE MEASUREMENT")
    print("=" * 70)
    print(f"Duration: {duration_minutes:.1f} minutes")
    print("Please ensure no GPU workloads are running...")
    print()

    # Start NVML poller
    poller = NVMLPoller(poll_interval_ms=poll_interval_ms)
    poller.start()

    start_time = time.time()
    duration_seconds = duration_minutes * 60

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(1)
    finally:
        # Always stop the poller even if something raises
        readings = poller.stop()

    if not readings:
        raise RuntimeError("Failed to collect any power readings during idle baseline")

    # Extract total power values
    powers = [r["total_power_w"] for r in readings]

    avg_power = statistics.mean(powers)
    min_power = min(powers)
    max_power = max(powers)
    std_power = statistics.stdev(powers) if len(powers) > 1 else 0.0
    spike_over_mean = max_power - avg_power

    stable = True
    reasons = []

    if std_power > IDLE_MAX_STD_WATTS:
        stable = False
        reasons.append(f"std {std_power:.2f}W > {IDLE_MAX_STD_WATTS:.2f}W")
    if spike_over_mean > IDLE_MAX_SPIKE_OVER_MEAN_WATTS:
        stable = False
        reasons.append(f"max-mean {spike_over_mean:.2f}W > {IDLE_MAX_SPIKE_OVER_MEAN_WATTS:.2f}W")

    # Extra check: other processes on GPUs during "idle" window
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        busy_gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if procs:
                busy_gpus.append(i)
        pynvml.nvmlShutdown()

        if busy_gpus:
            stable = False
            reasons.append(f"GPU(s) {busy_gpus} had running compute processes during idle baseline")
    except Exception:
        pass

    stability_reason = "ok" if stable else "; ".join(reasons) if reasons else "unknown"

    baseline = IdleBaseline(
        duration_seconds=duration_seconds,
        gpu_avg_power_watts=avg_power,
        gpu_min_power_watts=min_power,
        gpu_max_power_watts=max_power,
        gpu_std_power_watts=std_power,
        stable=stable,
        stability_reason=stability_reason,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print("\n--- Idle Baseline Results ---")
    print(f"  Average Power: {avg_power:.2f} W")
    print(f"  Min Power: {min_power:.2f} W")
    print(f"  Max Power: {max_power:.2f} W")
    print(f"  Stable: {stable} ({stability_reason})")
    print(f"  Std Dev: {std_power:.2f} W")
    print(f"  Samples: {len(powers)}")

    # Save raw readings if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        baseline_file = output_path / "idle_baseline.json"
        payload = baseline.to_dict()
        payload.update(
            {
                "poll_interval_ms": poll_interval_ms,
                "duration_seconds": duration_seconds,
                "readings": readings,
                "power_samples": powers,
            }
        )
        _write_json(baseline_file, payload)
        print(f"\n  Saved to: {baseline_file}")
    print("=" * 70)

    return baseline


def run_burn_in_test(
    output_dir: str,
    num_steps: int = 500,
    batch_size: int = 4,
    seq_length: int = 512,
    model_dim: int = 1024,
    num_layers: int = 12,
    config: Optional[Any] = None,
) -> BurnInResult:
    """
    Run a quick burn-in test to verify energy logging works correctly.

    This creates a small dummy model and runs training for a few steps,
    checking that:
    - Energy tracking initializes correctly
    - Logs are written properly
    - GPU utilization is reasonable
    """
    print("\n" + "=" * 70)
    print("BURN-IN TEST")
    print("=" * 70)
    print(f"Steps: {num_steps}, Batch Size: {batch_size}, Seq Length: {seq_length}")
    print()

    warnings_list: List[str] = []

    # Directory config
    root_dir = Path(output_dir)
    prerun_dir = root_dir / "prerun"
    prerun_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy model
    if not torch.cuda.is_available():
        warnings_list.append("CUDA not available - burn-in test running on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    print(f"  Creating dummy model on {device}...")

    VOCAB_SIZE = 8192

    class DummyModel(nn.Module):
        def __init__(self, dim, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, dim)
            self.layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True) for _ in range(num_layers)]
            )
            self.lm_head = nn.Linear(dim, VOCAB_SIZE)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)

    model = DummyModel(model_dim, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Untracked warm-up to stabilize kernels and clocks
    if device.type == "cuda":
        print("  Running untracked warm-up steps...")
        model.train()
        warmup_steps = min(10, max(1, num_steps // 10))
        for _ in range(warmup_steps):
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)
            labels = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Create energy tracker
    burn_in_dir = Path(output_dir)
    burn_in_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(
        output_dir=str(burn_in_dir),
        experiment_name=getattr(config, "experiment_name", "burn_in_test") if config else "burn_in_test",
        config=config,
    )

    # Start tracking
    print("  Starting energy tracking...")
    tracker.start_stage("burn_in")

    start_time = time.time()
    total_tokens = 0

    # Training loop (tracked)
    print(f"  Running {num_steps} training steps...")
    for step in range(num_steps):
        # Generate random input
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)
        labels = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)

        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_tokens += batch_size * seq_length

        # Print progress
        if (step + 1) % 100 == 0 or (step + 1) == num_steps:
            print(f"    Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    # End tracking
    print("  Ending energy tracking...")
    stage_metrics = tracker.end_stage(tokens_processed=total_tokens)
    duration = time.time() - start_time

    # Validate energy logs were written
    energy_logs_valid = True
    stage_file = root_dir / "energy" / "stages" / "burn_in.json"

    if not stage_file.exists():
        warnings_list.append(f"Energy log file not found: {stage_file}")
        energy_logs_valid = False
    else:
        try:
            with open(stage_file, "r") as f:
                log_data = json.load(f)
            if log_data.get("gpu_energy_joules", 0) == 0:
                warnings_list.append("GPU energy is zero in logs - check NVML and GPU visibility")
        except Exception as e:
            warnings_list.append(f"Failed to read energy log: {e}")
            energy_logs_valid = False

    # Check GPU utilization (heuristic: should process reasonable tokens/sec)
    tokens_per_sec = stage_metrics.tokens_per_second
    expected_min_tps = 1000  # Very conservative minimum

    if tokens_per_sec < expected_min_tps and device.type == "cuda":
        warnings_list.append(f"Low throughput: {tokens_per_sec:.0f} tokens/sec (expected > {expected_min_tps})")

    # Estimate GPU utilization from power
    avg_gpu_util = 0.0
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_limit_w = power_limit_mw / 1000.0
            pynvml.nvmlShutdown()

            if power_limit_w > 0:
                avg_gpu_util = (stage_metrics.gpu_avg_power_watts / power_limit_w) * 100
        except Exception:
            warnings_list.append("Failed to estimate GPU utilization from NVML")

    if device.type != "cuda":
        warnings_list.append("Burn-in ran on CPU; GPU energy logging was not validated.")

    result = BurnInResult(
        num_steps=num_steps,
        duration_seconds=duration,
        energy_logged=energy_logs_valid,
        gpu_energy_joules=stage_metrics.gpu_energy_joules,
        avg_gpu_utilization_pct=avg_gpu_util,
        tokens_per_second=tokens_per_sec,
        energy_logs_valid=energy_logs_valid,
        warnings=warnings_list,
    )

    print("\n--- Burn-in Test Results ---")
    print(f"  Duration: {duration:.2f}s")
    print(f"  GPU Energy: {result.gpu_energy_joules:.2f} J")
    print(f"  Tokens/sec: {result.tokens_per_second:.0f}")
    print(f"  Avg GPU Util: {result.avg_gpu_utilization_pct:.1f}%")
    print(f"  Energy Logs Valid: {result.energy_logs_valid}")

    if result.warnings:
        print("\n  ⚠ Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")

    # Save burn-in summary as JSON for reproducible inspection
    summary_file = prerun_dir / "burn_in_result.json"
    _write_json(summary_file, result.to_dict())
    print(f"\n  Burn-in summary saved to: {summary_file}")

    print("=" * 70)

    return result


def validate_sampling_interval(
    output_dir: str,
    test_duration_seconds: float = 60.0,
) -> SamplingIntervalResult:
    """
    Validate that different sampling intervals produce consistent energy estimates.

    Runs a simple GPU workload with two different polling intervals (1s and 2s)
    and checks that the energy estimates converge.
    """
    print("\n" + "=" * 70)
    print("SAMPLING INTERVAL VALIDATION")
    print("=" * 70)
    print(f"Test Duration: {test_duration_seconds:.0f}s")
    print()

    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available - skipping sampling interval validation")
        return SamplingIntervalResult(
            interval_1s_energy_joules=0.0,
            interval_15s_energy_joules=0.0,
            difference_percent=0.0,
            converged=True,
            recommended_interval_ms=1000,
        )

    device = torch.device("cuda:0")
    matrix_size = 4096
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Untracked warm-up workload
    print("  Running short warm-up workload (untracked)...")
    warmup_end = time.time() + min(10.0, test_duration_seconds / 3.0)
    while time.time() < warmup_end:
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    def run_workload(duration_sec: float, poll_interval_ms: int, label: str) -> float:
        """Run matrix multiplications for specified duration and integrate power via timestamps."""
        poller = NVMLPoller(poll_interval_ms=poll_interval_ms)
        poller.start()

        try:
            start_t = time.time()
            while time.time() - start_t < duration_sec:
                a = torch.randn(matrix_size, matrix_size, device=device)
                b = torch.randn(matrix_size, matrix_size, device=device)
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
        finally:
            readings = poller.stop()

        if not readings:
            return 0.0

        # Sort by timestamp and trapezoid-integrate P(t)
        readings.sort(key=lambda r: r["timestamp"])
        energy_joules = 0.0
        for prev, cur in zip(readings, readings[1:]):
            dt = max(0.0, cur["timestamp"] - prev["timestamp"])
            energy_joules += 0.5 * (prev["total_power_w"] + cur["total_power_w"]) * dt

        # Save raw readings for this interval
        readings_file = output_path / f"{label}_readings.json"
        with open(readings_file, "w") as f:
            json.dump(
                {
                    "label": label,
                    "poll_interval_ms": poll_interval_ms,
                    "duration_seconds": duration_sec,
                    "readings": readings,
                },
                f,
                indent=2,
            )
        print(f"    Saved raw readings to: {readings_file}")

        return energy_joules

    # Test with 1s interval
    print("  Testing with 1000ms polling interval...")
    energy_1s = run_workload(test_duration_seconds, poll_interval_ms=1000, label="interval_1000ms")
    print(f"    Energy: {energy_1s:.2f} J")

    # Small cooldown
    time.sleep(2)

    print("  Testing with 2000ms polling interval...")
    energy_2s = run_workload(test_duration_seconds, poll_interval_ms=2000, label="interval_2000ms")
    print(f"    Energy: {energy_2s:.2f} J")

    # Calculate difference
    if energy_1s > 0:
        diff_percent = abs(energy_1s - energy_2s) / energy_1s * 100
    else:
        diff_percent = 0.0

    converged = diff_percent < 10.0
    recommended_ms = 1000 if converged else 500

    result = SamplingIntervalResult(
        interval_1s_energy_joules=energy_1s,
        interval_15s_energy_joules=energy_2s,
        difference_percent=diff_percent,
        converged=converged,
        recommended_interval_ms=recommended_ms,
    )

    print("\n--- Sampling Interval Results ---")
    print(f"  1s interval: {energy_1s:.2f} J")
    print(f"  2s interval: {energy_2s:.2f} J")
    print(f"  Difference: {diff_percent:.2f}%")
    print(f"  Converged: {converged}")
    print(f"  Recommended: {recommended_ms}ms")

    if not converged:
        print("\n  ⚠ Warning: Large difference between intervals!")
        print(f"    Consider using {recommended_ms}ms polling interval")

    # Save summary JSON
    summary_file = output_path / "sampling_interval_result.json"
    with open(summary_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n  Sampling interval summary saved to: {summary_file}")

    print("=" * 70)

    return result


def validate_hardware(
    expected_gpu_type: Optional[str] = None,
    expected_vram_gb: Optional[float] = None,
    min_vram_gb: float = 16.0,
) -> HardwareValidation:
    """
    Validate hardware configuration meets requirements.

    Checks:
    - GPU type matches expected (if specified)
    - VRAM is sufficient for workload (based on min VRAM across GPUs)
    - GPU count and heterogeneity
    """
    print("\n" + "=" * 70)
    print("HARDWARE VALIDATION")
    print("=" * 70)

    warnings_list: List[str] = []

    # Collect GPU info
    gpus = collect_gpu_info()

    if not gpus:
        warnings_list.append("No GPUs detected!")
        return HardwareValidation(
            expected_gpu_type=expected_gpu_type,
            actual_gpu_type="None",
            gpu_match=False,
            expected_vram_gb=expected_vram_gb,
            actual_vram_gb=0.0,
            vram_sufficient=False,
            gpu_count=0,
            warnings=warnings_list,
        )

    gpu_count = len(gpus)
    gpu_types = {g["name"] for g in gpus}
    # Representative type: first GPU
    actual_gpu_type = gpus[0]["name"]
    # Use minimum VRAM across GPUs as the safe bound
    vram_values_gb = [g["memory_total_mb"] / 1024.0 for g in gpus]
    actual_vram_gb = min(vram_values_gb)

    # Check GPU type match (all GPUs should match expected)
    gpu_match = True
    if expected_gpu_type:
        for t in gpu_types:
            if expected_gpu_type.lower() not in t.lower():
                gpu_match = False
        if not gpu_match:
            warnings_list.append(
                f"GPU mismatch: expected '{expected_gpu_type}' on all devices, " f"but detected types: {sorted(gpu_types)}"
            )

    if len(gpu_types) > 1:
        warnings_list.append(f"Heterogeneous GPU types detected: {sorted(gpu_types)}")

    # Check VRAM
    vram_sufficient = actual_vram_gb >= min_vram_gb
    if not vram_sufficient:
        warnings_list.append(f"Insufficient VRAM: minimum across GPUs is {actual_vram_gb:.1f} GB < {min_vram_gb:.1f} GB required")

    if expected_vram_gb and abs(actual_vram_gb - expected_vram_gb) > 2.0:
        warnings_list.append(
            f"VRAM differs from expected: min across GPUs is {actual_vram_gb:.1f} GB vs {expected_vram_gb:.1f} GB"
        )

    result = HardwareValidation(
        expected_gpu_type=expected_gpu_type,
        actual_gpu_type=actual_gpu_type,
        gpu_match=gpu_match,
        expected_vram_gb=expected_vram_gb,
        actual_vram_gb=actual_vram_gb,
        vram_sufficient=vram_sufficient,
        gpu_count=gpu_count,
        warnings=warnings_list,
    )

    print(f"\n  GPU Count: {gpu_count}")
    print(f"  GPU Types: {', '.join(sorted(gpu_types))}")
    print(f"  Min VRAM across GPUs: {actual_vram_gb:.1f} GB")

    if expected_gpu_type:
        print(f"  Expected GPU: {expected_gpu_type}")
        print(f"  Match on all devices: {gpu_match}")

    print(f"  VRAM Sufficient: {vram_sufficient} (min {min_vram_gb:.1f} GB)")

    if result.warnings:
        print("\n  ⚠ Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")

    print("=" * 70)

    return result


def run_prerun_validation(
    output_dir: str,
    idle_duration_minutes: float = 5.0,
    burn_in_steps: int = 500,
    expected_gpu_type: Optional[str] = None,
    expected_vram_gb: Optional[float] = None,
    skip_idle: bool = False,
    skip_burn_in: bool = False,
    skip_sampling: bool = False,
    config: Optional[Any] = None,
) -> PreRunReport:
    """
    Run complete pre-run validation suite.

    This is the main entry point for running all validation checks before
    a full experiment. It performs:
    1. Idle baseline measurement
    2. Burn-in test
    3. Sampling interval validation
    4. Hardware assertions

    Args:
        output_dir: Directory for validation outputs
        idle_duration_minutes: Duration of idle baseline measurement
        burn_in_steps: Number of steps for burn-in test
        expected_gpu_type: Expected GPU type
        expected_vram_gb: Expected VRAM in GB
        skip_idle: Skip idle baseline (useful for debugging)
        skip_burn_in: Skip burn-in test
        skip_sampling: Skip sampling interval validation

    Returns:
        PreRunReport with all validation results
    """
    print("\n" + "=" * 70)
    print("PRE-RUN VALIDATION SUITE")
    print("=" * 70)
    print(f"Output Directory: {output_dir}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    critical_warnings = []

    # 1. Hardware Validation (always run first)
    print("Step 1/4: Hardware Validation")
    hardware = validate_hardware(
        expected_gpu_type=expected_gpu_type,
        expected_vram_gb=expected_vram_gb,
    )

    if not hardware.vram_sufficient:
        critical_warnings.append("Insufficient VRAM for workload")

    # 2. Idle Baseline
    if skip_idle:
        print("\nStep 2/4: Idle Baseline (SKIPPED)")
        idle_baseline = IdleBaseline(
            duration_seconds=0.0,
            gpu_avg_power_watts=0.0,
            gpu_min_power_watts=0.0,
            gpu_max_power_watts=0.0,
            gpu_std_power_watts=0.0,
            stable=True,
            stability_reason="skipped",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        print("\nStep 2/4: Idle Baseline")
        idle_baseline = measure_idle_baseline(
            duration_minutes=idle_duration_minutes,
            output_dir=str(output_path / "idle_baseline"),
        )

    if not idle_baseline.stable:
        warnings.warn(f"Idle baseline unstable: {idle_baseline.stability_reason}")

    # 3. Burn-in Test
    if skip_burn_in:
        print("\nStep 3/4: Burn-in Test (SKIPPED)")
        burn_in = BurnInResult(
            num_steps=0,
            duration_seconds=0.0,
            energy_logged=True,
            gpu_energy_joules=0.0,
            avg_gpu_utilization_pct=0.0,
            tokens_per_second=0.0,
            energy_logs_valid=True,
            warnings=[],
        )
    else:
        print("\nStep 3/4: Burn-in Test")
        burn_in = run_burn_in_test(
            output_dir=Path(output_path),
            num_steps=burn_in_steps,
            config=config,
        )

        if not burn_in.energy_logs_valid:
            critical_warnings.append("Energy logging validation failed")

    # 4. Sampling Interval Validation
    if skip_sampling:
        print("\nStep 4/4: Sampling Interval Validation (SKIPPED)")
        sampling = SamplingIntervalResult(
            interval_1s_energy_joules=0.0,
            interval_15s_energy_joules=0.0,
            difference_percent=0.0,
            converged=True,
            recommended_interval_ms=1000,
        )
    else:
        print("\nStep 4/4: Sampling Interval Validation")
        sampling = validate_sampling_interval(
            output_dir=str(output_path / "sampling_validation"),
        )

        if not sampling.converged:
            critical_warnings.append(f"Sampling intervals diverged by {sampling.difference_percent:.1f}%")

    # Create report
    all_checks_passed = len(critical_warnings) == 0

    report = PreRunReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        idle_baseline=idle_baseline,
        burn_in=burn_in,
        sampling_interval=sampling,
        hardware_validation=hardware,
        all_checks_passed=all_checks_passed,
        critical_warnings=critical_warnings,
    )

    # Save report
    report_file = output_path / "prerun_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    _print_prerun_summary(report, report_file)

    return report


# Convenience function for quick validation
def quick_validation(output_dir: str = "./prerun_validation", config: Optional[Any] = None) -> bool:
    """
    Run a quick validation with reduced durations (for testing).

    Args:
        output_dir: Output directory

    Returns:
        True if all checks passed
    """
    report = run_prerun_validation(
        output_dir=output_dir,
        idle_duration_minutes=1.0,  # 1 minute instead of 5
        burn_in_steps=100,  # 100 steps instead of 500
        skip_sampling=False,
        config=config,
    )

    return report.all_checks_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-run validation for energy benchmarking")
    parser.add_argument("--output-dir", type=str, default="./prerun_validation", help="Output directory for validation results")
    parser.add_argument("--idle-duration", type=float, default=5.0, help="Duration of idle baseline measurement in minutes")
    parser.add_argument("--burn-in-steps", type=int, default=500, help="Number of steps for burn-in test")
    parser.add_argument("--expected-gpu", type=str, default=None, help="Expected GPU type (e.g., 'A100', 'H100')")
    parser.add_argument("--expected-vram", type=float, default=None, help="Expected VRAM in GB")
    parser.add_argument("--quick", action="store_true", help="Run quick validation with reduced durations")
    parser.add_argument("--skip-idle", action="store_true", help="Skip idle baseline measurement")
    parser.add_argument("--skip-burn-in", action="store_true", help="Skip burn-in test")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling interval validation")

    args = parser.parse_args()

    if args.quick:
        success = quick_validation(args.output_dir)
        exit(0 if success else 1)
    else:
        report = run_prerun_validation(
            output_dir=args.output_dir,
            idle_duration_minutes=args.idle_duration,
            burn_in_steps=args.burn_in_steps,
            expected_gpu_type=args.expected_gpu,
            expected_vram_gb=args.expected_vram,
            skip_idle=args.skip_idle,
            skip_burn_in=args.skip_burn_in,
            skip_sampling=args.skip_sampling,
        )

        exit(0 if report.all_checks_passed else 1)

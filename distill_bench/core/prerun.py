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
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import pynvml

from .energy_logger import EnergyTracker, NVMLPoller
from .environment import collect_environment, collect_gpu_info


@dataclass
class IdleBaseline:
    """Idle power baseline measurements."""
    duration_seconds: float
    gpu_avg_power_watts: float
    gpu_min_power_watts: float
    gpu_max_power_watts: float
    gpu_std_power_watts: float
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


def measure_idle_baseline(
    duration_minutes: float = 5.0,
    poll_interval_ms: int = 500,
    output_dir: Optional[str] = None,
) -> IdleBaseline:
    """
    Measure idle GPU power for baseline calculation.
    
    This helps calculate net energy by subtracting the idle power consumption
    from active training power consumption.
    
    Args:
        duration_minutes: How long to measure (default 5 minutes)
        poll_interval_ms: Power polling interval in milliseconds
        output_dir: Optional directory to save raw power readings
        
    Returns:
        IdleBaseline with statistics
    """
    print("\n" + "="*70)
    print("IDLE BASELINE MEASUREMENT")
    print("="*70)
    print(f"Duration: {duration_minutes:.1f} minutes")
    print("Please ensure no GPU workloads are running...")
    print()
    
    # Start NVML poller
    poller = NVMLPoller(poll_interval_ms=poll_interval_ms)
    poller.start()
    
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    # Print progress periodically
    last_print = start_time
    while time.time() - start_time < duration_seconds:
        current_time = time.time()
        if current_time - last_print >= 30:  # Print every 30 seconds
            elapsed = current_time - start_time
            remaining = duration_seconds - elapsed
            print(f"  Progress: {elapsed:.0f}s / {duration_seconds:.0f}s (remaining: {remaining:.0f}s)")
            last_print = current_time
        time.sleep(1)
    
    # Stop poller and get readings
    readings = poller.stop()
    
    if not readings:
        raise RuntimeError("Failed to collect any power readings during idle baseline")
    
    # Extract total power values
    powers = [r["total_power_w"] for r in readings]
    
    # Calculate statistics
    import statistics
    avg_power = statistics.mean(powers)
    min_power = min(powers)
    max_power = max(powers)
    std_power = statistics.stdev(powers) if len(powers) > 1 else 0.0
    
    baseline = IdleBaseline(
        duration_seconds=duration_seconds,
        gpu_avg_power_watts=avg_power,
        gpu_min_power_watts=min_power,
        gpu_max_power_watts=max_power,
        gpu_std_power_watts=std_power,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    print("\n--- Idle Baseline Results ---")
    print(f"  Average Power: {avg_power:.2f} W")
    print(f"  Min Power: {min_power:.2f} W")
    print(f"  Max Power: {max_power:.2f} W")
    print(f"  Std Dev: {std_power:.2f} W")
    print(f"  Samples: {len(powers)}")
    
    # Save raw readings if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        baseline_file = output_path / "idle_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline.to_dict(), f, indent=2)
        
        readings_file = output_path / "idle_baseline_readings.json"
        with open(readings_file, 'w') as f:
            json.dump({
                "poll_interval_ms": poll_interval_ms,
                "duration_seconds": duration_seconds,
                "readings": readings,
                "power_samples": powers,
            }, f)
        
        print(f"\n  Saved to: {baseline_file}")
    
    print("="*70)
    
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
    
    Args:
        output_dir: Directory for energy logs
        num_steps: Number of training steps (default 500)
        batch_size: Batch size for dummy training
        seq_length: Sequence length for dummy data
        model_dim: Model hidden dimension
        num_layers: Number of transformer layers
        
    Returns:
        BurnInResult with validation outcomes
    """
    print("\n" + "="*70)
    print("BURN-IN TEST")
    print("="*70)
    print(f"Steps: {num_steps}, Batch Size: {batch_size}, Seq Length: {seq_length}")
    print()
    
    warnings_list = []
    
    # Create dummy model
    if not torch.cuda.is_available():
        warnings_list.append("CUDA not available - burn-in test running on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    
    print(f"  Creating dummy model on {device}...")
    
    # Simple transformer-like model
    class DummyModel(nn.Module):
        def __init__(self, dim, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(50000, dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
                for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(dim, 50000)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)
    
    model = DummyModel(model_dim, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create energy tracker
    burn_in_dir = Path(output_dir) / "burn_in"
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
    
    # Training loop
    print(f"  Running {num_steps} training steps...")
    for step in range(num_steps):
        # Generate random input
        input_ids = torch.randint(0, 50000, (batch_size, seq_length), device=device)
        labels = torch.randint(0, 50000, (batch_size, seq_length), device=device)
        
        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, 50000), labels.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_tokens += batch_size * seq_length
        
        # Print progress
        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
    
    # End tracking
    print("  Ending energy tracking...")
    stage_metrics = tracker.end_stage(tokens_processed=total_tokens)
    duration = time.time() - start_time
    
    # Validate energy logs were written
    energy_logs_valid = True
    stage_file = burn_in_dir / "energy_logs" / "stage_burn_in.json"
    
    if not stage_file.exists():
        warnings_list.append(f"Energy log file not found: {stage_file}")
        energy_logs_valid = False
    else:
        try:
            with open(stage_file, 'r') as f:
                log_data = json.load(f)
            if log_data.get("gpu_energy_joules", 0) == 0:
                warnings_list.append("GPU energy is zero in logs - check NVML")
        except Exception as e:
            warnings_list.append(f"Failed to read energy log: {e}")
            energy_logs_valid = False
    
    # Check GPU utilization (heuristic: should process reasonable tokens/sec)
    tokens_per_sec = stage_metrics.tokens_per_second
    expected_min_tps = 1000  # Very conservative minimum
    
    if tokens_per_sec < expected_min_tps:
        warnings_list.append(
            f"Low throughput: {tokens_per_sec:.0f} tokens/sec (expected > {expected_min_tps})"
        )
    
    # Estimate GPU utilization from power
    avg_gpu_util = 0.0
    if torch.cuda.is_available():
        # Very rough heuristic: assume max power ~ 100% util
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_limit_w = power_limit_mw / 1000.0
            pynvml.nvmlShutdown()
            
            avg_gpu_util = (stage_metrics.gpu_avg_power_watts / power_limit_w) * 100
        except Exception:
            warnings_list.append("Failed to estimate GPU utilization")
    
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
    
    print("="*70)
    
    return result


def validate_sampling_interval(
    output_dir: str,
    test_duration_seconds: float = 60.0,
) -> SamplingIntervalResult:
    """
    Validate that different sampling intervals produce consistent energy estimates.
    
    Runs a simple GPU workload with two different polling intervals (1s and 15s)
    and checks that the energy estimates converge.
    
    Args:
        output_dir: Directory for logs
        test_duration_seconds: Duration of test workload (default 60s)
        
    Returns:
        SamplingIntervalResult with comparison
    """
    print("\n" + "="*70)
    print("SAMPLING INTERVAL VALIDATION")
    print("="*70)
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
    
    # Create a simple GPU workload
    device = torch.device("cuda:0")
    matrix_size = 4096
    
    def run_workload(duration_sec, poll_interval_ms):
        """Run matrix multiplications for specified duration."""
        poller = NVMLPoller(poll_interval_ms=poll_interval_ms)
        poller.start()
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            # Matrix multiplication to keep GPU busy
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)
            c = torch.mm(a, b)
            del a, b, c
        
        torch.cuda.synchronize()
        readings = poller.stop()
        
        # Calculate energy
        if not readings:
            return 0.0
        
        powers = [r["total_power_w"] for r in readings]
        interval_sec = poll_interval_ms / 1000.0
        energy_joules = sum(powers) * interval_sec
        
        return energy_joules
    
    # Test with 1s interval
    print("  Testing with 1000ms polling interval...")
    energy_1s = run_workload(test_duration_seconds, poll_interval_ms=1000)
    print(f"    Energy: {energy_1s:.2f} J")
    
    # Small cooldown
    time.sleep(2)
    
    # Test with 15s interval
    print("  Testing with 15000ms polling interval...")
    energy_15s = run_workload(test_duration_seconds, poll_interval_ms=15000)
    print(f"    Energy: {energy_15s:.2f} J")
    
    # Calculate difference
    if energy_1s > 0:
        diff_percent = abs(energy_1s - energy_15s) / energy_1s * 100
    else:
        diff_percent = 0.0
    
    # Check convergence (allow up to 10% difference)
    converged = diff_percent < 10.0
    
    # Recommend interval based on convergence
    if converged:
        recommended_ms = 1000  # 1s is fine
    else:
        recommended_ms = 500  # Use shorter interval for better accuracy
    
    result = SamplingIntervalResult(
        interval_1s_energy_joules=energy_1s,
        interval_15s_energy_joules=energy_15s,
        difference_percent=diff_percent,
        converged=converged,
        recommended_interval_ms=recommended_ms,
    )
    
    print("\n--- Sampling Interval Results ---")
    print(f"  1s interval: {energy_1s:.2f} J")
    print(f"  15s interval: {energy_15s:.2f} J")
    print(f"  Difference: {diff_percent:.2f}%")
    print(f"  Converged: {converged}")
    print(f"  Recommended: {recommended_ms}ms")
    
    if not converged:
        print("\n  ⚠ Warning: Large difference between intervals!")
        print(f"    Consider using {recommended_ms}ms polling interval")
    
    print("="*70)
    
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
    - VRAM is sufficient for workload
    - GPU count
    
    Args:
        expected_gpu_type: Expected GPU name (e.g., "A100", "H100")
        expected_vram_gb: Expected VRAM in GB
        min_vram_gb: Minimum required VRAM in GB
        
    Returns:
        HardwareValidation with checks
    """
    print("\n" + "="*70)
    print("HARDWARE VALIDATION")
    print("="*70)
    
    warnings_list = []
    
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
    
    # Use first GPU for checks
    gpu = gpus[0]
    actual_gpu_type = gpu["name"]
    actual_vram_gb = gpu["memory_total_mb"] / 1024.0
    gpu_count = len(gpus)
    
    # Check GPU type match
    gpu_match = True
    if expected_gpu_type:
        if expected_gpu_type.lower() not in actual_gpu_type.lower():
            gpu_match = False
            warnings_list.append(
                f"GPU mismatch: expected '{expected_gpu_type}', got '{actual_gpu_type}'"
            )
    
    # Check VRAM
    vram_sufficient = actual_vram_gb >= min_vram_gb
    if not vram_sufficient:
        warnings_list.append(
            f"Insufficient VRAM: {actual_vram_gb:.1f} GB < {min_vram_gb:.1f} GB required"
        )
    
    if expected_vram_gb and abs(actual_vram_gb - expected_vram_gb) > 2.0:
        warnings_list.append(
            f"VRAM differs from expected: {actual_vram_gb:.1f} GB vs {expected_vram_gb:.1f} GB"
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
    print(f"  GPU Type: {actual_gpu_type}")
    print(f"  VRAM: {actual_vram_gb:.1f} GB")
    
    if expected_gpu_type:
        print(f"  Expected GPU: {expected_gpu_type}")
        print(f"  Match: {gpu_match}")
    
    print(f"  VRAM Sufficient: {vram_sufficient} (min {min_vram_gb:.1f} GB)")
    
    if result.warnings:
        print("\n  ⚠ Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")
    
    print("="*70)
    
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
    print("\n" + "="*70)
    print("PRE-RUN VALIDATION SUITE")
    print("="*70)
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
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        print("\nStep 2/4: Idle Baseline")
        idle_baseline = measure_idle_baseline(
            duration_minutes=idle_duration_minutes,
            output_dir=str(output_path / "idle_baseline"),
        )
    
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
            output_dir=str(output_path / "burn_in"),
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
            critical_warnings.append(
                f"Sampling intervals diverged by {sampling.difference_percent:.1f}%"
            )
    
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
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("PRE-RUN VALIDATION SUMMARY")
    print("="*70)
    print(f"\nAll Checks Passed: {all_checks_passed}")
    
    if critical_warnings:
        print("\n⚠ CRITICAL WARNINGS:")
        for warning in critical_warnings:
            print(f"  - {warning}")
        print("\nPlease address these issues before running full experiments.")
    else:
        print("\n✓ All validation checks passed successfully!")
        print("  System is ready for energy benchmarking experiments.")
    
    print(f"\nFull report saved to: {report_file}")
    print("="*70)
    
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
    parser.add_argument("--output-dir", type=str, default="./prerun_validation",
                      help="Output directory for validation results")
    parser.add_argument("--idle-duration", type=float, default=5.0,
                      help="Duration of idle baseline measurement in minutes")
    parser.add_argument("--burn-in-steps", type=int, default=500,
                      help="Number of steps for burn-in test")
    parser.add_argument("--expected-gpu", type=str, default=None,
                      help="Expected GPU type (e.g., 'A100', 'H100')")
    parser.add_argument("--expected-vram", type=float, default=None,
                      help="Expected VRAM in GB")
    parser.add_argument("--quick", action="store_true",
                      help="Run quick validation with reduced durations")
    parser.add_argument("--skip-idle", action="store_true",
                      help="Skip idle baseline measurement")
    parser.add_argument("--skip-burn-in", action="store_true",
                      help="Skip burn-in test")
    parser.add_argument("--skip-sampling", action="store_true",
                      help="Skip sampling interval validation")
    
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

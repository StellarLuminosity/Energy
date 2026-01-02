# Pre-Run Validation Guide

## Overview

The pre-run validation module (`distill_bench/core/prerun.py`) provides essential checks before running expensive energy benchmarking experiments. It helps catch configuration issues, hardware problems, and measurement errors early.

## What It Does

### 1. **Idle Baseline Measurement**
Measures GPU power consumption when idle (no workload running). This baseline can be subtracted from active power to calculate net energy consumption.

- **Duration**: 5-10 minutes (configurable)
- **Output**: Average, min, max, and standard deviation of idle power
- **Use case**: Accurate net energy calculation

### 2. **Burn-In Test**
Runs a quick training simulation (500-1000 steps) with a dummy model to verify that energy tracking works correctly.

- **Checks**:
  - Energy logs are written properly
  - GPU utilization is reasonable
  - Token throughput is within expected range
- **Use case**: Catch energy logging bugs before long experiments

### 3. **Sampling Interval Validation**
Compares energy measurements with different polling intervals (1s vs 15s) to ensure consistency.

- **Checks**: Energy estimates should converge within 10%
- **Recommendation**: Suggests optimal polling interval
- **Use case**: Verify measurement accuracy

### 4. **Hardware Assertions**
Validates hardware configuration against expectations.

- **Checks**:
  - GPU type matches expected
  - Sufficient VRAM available
  - GPU count
- **Use case**: Prevent running experiments on wrong hardware

## Usage

### Quick Start (Reduced Durations)

For testing or debugging, run quick validation with reduced durations:

```python
from distill_bench.core.prerun import quick_validation

success = quick_validation(output_dir="./prerun_validation")
if not success:
    print("Validation failed - check warnings before running experiments")
```

Or via command line:

```bash
python distill_bench/core/prerun.py --quick --output-dir ./validation
```

### Full Validation (Before Real Experiments)

Run complete validation with full durations:

```python
from distill_bench.core.prerun import run_prerun_validation

report = run_prerun_validation(
    output_dir="./prerun_validation",
    idle_duration_minutes=5.0,      # 5-10 minutes recommended
    burn_in_steps=500,               # 500-1000 steps
    expected_gpu_type="A100",        # Optional: validate GPU type
    expected_vram_gb=80.0,           # Optional: expected VRAM
)

if report.all_checks_passed:
    print("✓ Ready for experiments!")
else:
    print("⚠ Issues detected:")
    for warning in report.critical_warnings:
        print(f"  - {warning}")
```

Or via command line:

```bash
python distill_bench/core/prerun.py \
    --output-dir ./validation \
    --idle-duration 5.0 \
    --burn-in-steps 500 \
    --expected-gpu A100 \
    --expected-vram 80.0
```

### Skipping Specific Checks

You can skip specific checks if needed:

```bash
python distill_bench/core/prerun.py \
    --output-dir ./validation \
    --skip-idle \              # Skip idle baseline (saves time)
    --skip-burn-in \           # Skip burn-in test
    --skip-sampling            # Skip sampling validation
```

### Individual Checks

You can also run individual checks separately:

```python
from distill_bench.core.prerun import (
    measure_idle_baseline,
    run_burn_in_test,
    validate_sampling_interval,
    validate_hardware,
)

# Just measure idle baseline
baseline = measure_idle_baseline(
    duration_minutes=5.0,
    poll_interval_ms=500,
    output_dir="./idle_baseline"
)
print(f"Idle power: {baseline.gpu_avg_power_watts:.2f} W")

# Just validate hardware
hardware = validate_hardware(
    expected_gpu_type="A100",
    min_vram_gb=16.0
)
print(f"GPU: {hardware.actual_gpu_type}, VRAM: {hardware.actual_vram_gb:.1f} GB")
```

## Output Files

After running validation, you'll find:

```
prerun_validation/
├── prerun_validation_report.json    # Complete validation report
├── idle_baseline/
│   ├── idle_baseline.json           # Baseline statistics
│   └── idle_baseline_readings.json  # Raw power readings
├── burn_in/
│   └── energy_logs/                 # Energy tracking logs
│       ├── stage_burn_in.json
│       └── ...
└── sampling_validation/
    └── ...                          # Sampling test outputs
```

## Integration with Experiments

Before running a full experiment, integrate validation:

```python
from distill_bench.core.prerun import run_prerun_validation
from distill_bench.pipelines.kd_main import run_kd

# 1. Run pre-run validation
print("Running pre-run validation...")
report = run_prerun_validation(
    output_dir="./experiments/exp001/prerun",
    idle_duration_minutes=5.0,
    expected_gpu_type="A100",
)

if not report.all_checks_passed:
    print("Validation failed! Aborting experiment.")
    for warning in report.critical_warnings:
        print(f"  - {warning}")
    exit(1)

print("✓ Validation passed, starting experiment...")

# 2. Run actual experiment
run_kd(config_path="./configs/kd_7b_to_1b.yaml")
```

## Interpreting Results

### Idle Baseline
- **Typical values**: 20-50W for modern GPUs at idle
- **High variance**: May indicate background processes using GPU
- **Action**: If std dev > 5W, check for background workloads

### Burn-In Test
- **Energy logged = False**: Critical issue with energy tracking setup
- **Low throughput**: GPU not being utilized properly
- **Action**: Check CUDA installation, GPU availability

### Sampling Interval
- **Converged = False**: Large difference between intervals
- **Action**: Use recommended shorter polling interval (500ms)

### Hardware Validation
- **GPU mismatch**: Running on different hardware than expected
- **Insufficient VRAM**: Model may not fit in memory
- **Action**: Adjust model size or use correct hardware

## Common Issues

### Issue: "NVML initialization failed"
**Solution**: Check that NVIDIA drivers are installed and `nvidia-smi` works

### Issue: "No GPUs detected"
**Solution**: 
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU visibility: `echo $CUDA_VISIBLE_DEVICES`

### Issue: Burn-in test shows 0 energy
**Solution**:
- NVML may not have permission to read power
- Try running with sudo or check system permissions
- Verify `nvidia-smi -q -d POWER` shows power readings

### Issue: High idle power variance
**Solution**:
- Close other GPU applications
- Wait for GPU to cool down
- Run validation when system is quiet

## Best Practices

1. **Always run validation** before expensive multi-day experiments
2. **Save validation reports** with experiment artifacts for reproducibility
3. **Use full durations** (5+ min idle, 500+ steps burn-in) for production
4. **Check hardware** matches expected configuration in shared clusters
5. **Monitor trends** - compare validation results across runs to detect hardware degradation

## Advanced: Custom Validation

You can create custom validation workflows:

```python
from distill_bench.core.prerun import *

# Custom workflow
def validate_for_my_experiment(output_dir: str):
    # 1. Quick hardware check
    hw = validate_hardware(expected_gpu_type="A100", min_vram_gb=40.0)
    if not hw.vram_sufficient:
        raise RuntimeError("Need at least 40GB VRAM")
    
    # 2. Short idle baseline (1 minute for testing)
    baseline = measure_idle_baseline(duration_minutes=1.0, output_dir=output_dir)
    if baseline.gpu_avg_power_watts > 100:
        print("Warning: High idle power - check for background processes")
    
    # 3. Custom burn-in with specific model size
    burn_in = run_burn_in_test(
        output_dir=output_dir,
        num_steps=200,
        batch_size=8,
        model_dim=2048,  # Closer to actual model
        num_layers=24,
    )
    
    return burn_in.energy_logs_valid

validate_for_my_experiment("./my_validation")
```

## Technical Details

### Energy Calculation
- GPU energy is calculated by integrating power over time: `E = Σ(P_i × Δt)`
- Power is sampled at regular intervals via NVML
- Default interval: 500ms (adjustable)

### Idle Baseline Usage
To calculate net energy:
```python
net_energy = total_energy - (idle_power_watts × duration_seconds)
```

### Burn-In Model
- Simple transformer-like architecture
- Configurable size (default: 1024 dim, 12 layers)
- Random data (no actual learning)
- Purpose: stress test energy tracking infrastructure

## See Also

- `energy_logger.py` - Core energy tracking implementation
- `environment.py` - Hardware metadata collection
- Phase 7 in `dist.plan.md` - Integration into pipelines


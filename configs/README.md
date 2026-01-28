# Configuration System

YAML-based configuration with inheritance for reproducible distillation experiments.

## Structure

```
configs/
├── base.yaml                          # Shared base config (FIXED settings)
└── experiments/
    ├── kd_7b_to_1b.yaml              # Knowledge Distillation
    ├── sft_7b_to_1b.yaml             # Data Distillation (SFT)
    └── eval_olmo2_1b.yaml            # Eval-only harness example
```

## Usage

### Load a config:
```python
from distill_bench.core.config_loader import load_config

# Load experiment config (automatically inherits from base.yaml)
config = load_config("configs/experiments/kd_7b_to_1b.yaml")

# Access values
print(config.batch_size)           # Flat access (backward compatible)
print(config.get('training.batch_size'))  # Nested access
print(config.to_dict())             # Full config as dict
```

### Command line:
```bash
# Run experiment with specific config
python run_experiment.py --config configs/experiments/kd_7b_to_1b.yaml
```

## Config Hierarchy

Experiment configs **inherit** from `base.yaml` and **override** specific values:

1. `base.yaml` loaded first (all standard settings)
2. Experiment config merged on top (overrides specific values)
3. Environment variables can override at runtime (optional)

## Environment Variable Overrides

Useful for SLURM jobs:

```bash
export DISTILL_OUTPUT_DIR=/scratch/user/run_123
export DISTILL_WANDB_ENTITY=my-team
python run_experiment.py --config configs/experiments/kd_7b_to_1b.yaml
```

## Base Config

`base.yaml` contains **FIXED** settings for valid experiment comparison:

- **Seeds**: Fixed at 42 for reproducibility
- **Batch size**: Fixed for fair comparison
- **Optimizer**: AdamW with fixed LR
- **Precision**: bf16
- **Dataset**: Tulu-3 SFT mixture
- **Energy tracking**: Enabled with 500ms NVML polling

❗ **These should NOT be changed between experiments** for valid comparison.

## Experiment Configs

Each experiment config specifies:

- `pipeline`: "kd" | "sft" | "dpo"
- `model`: teacher and student models
- Pipeline-specific hyperparameters
- `output.output_dir`: Where to save results
- `benchmark.*`: Defaults for the evaluation harness (model to score, output dir, tasks)

### KD Config (`kd_7b_to_1b.yaml`)
- `distillation.alpha`: CE/KL loss weighting
- `distillation.temperature`: KL temperature

### SFT Config (`sft_7b_to_1b.yaml`)
- `synthetic_data.generation`: Teacher generation settings
- `synthetic_data.filtering`: Quality filters

### DPO Config (`dpo_7b_to_1b.yaml`)
- `dpo.beta`: DPO temperature parameter
- `model.judge`: Judge model for preference labeling

## Creating New Experiments

```yaml
# configs/experiments/my_experiment.yaml
pipeline: "kd"

experiment:
  name: "my_kd_experiment"

model:
  teacher: "allenai/OLMo-2-1124-7B-SFT"
  student: "allenai/OLMo-2-0425-1B-SFT"

distillation:
  alpha: 0.3
  temperature: 3.0

output:
  output_dir: "/scratch/user/my_experiment"

# Optional: defaults for evaluation harness
benchmark:
  output_dir: "/scratch/user/benchmarks"
  model: "/scratch/user/my_experiment/final_model/hf_format"  # or an HF id
  model_type: "allenai/OLMo-2-0425-1B-SFT"
  subfolder_name: "my_experiment_benchmark"
  tasks:
    - "gsm8k"
    - "mmlu"
    - "ifeval"
    - "alpaca_eval"
    - "mt_bench_101"
```

All other settings inherit from `base.yaml`.

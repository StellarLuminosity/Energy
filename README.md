# ⚡ ML Energy Harness

**Stage-wise GPU + CPU + Carbon tracking for machine learning workloads**

[![Paper](https://img.shields.io/badge/ICML%202026-Paper-blue)](https://arxiv.org/abs/2605.13981)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)
[![NVML](https://img.shields.io/badge/GPU%20Telemetry-NVML-76b900)](https://developer.nvidia.com/management-library-nvml)

---

Most energy benchmarks for ML are incomplete. They count student training and call it done, but don't pay attention to teacher inference for distillation, the data generation side, and evaluation. **This harness measures everything.**

`distill_bench/core/energy_logger.py` is a drop-in energy tracking interface that combines:

- **NVML** — real-time GPU power telemetry sampled every 0.5 s, integrated to kWh
- **CodeCarbon / RAPL** — CPU energy estimation in process-tracking mode
- **CO₂e** — carbon estimates derived from regional grid intensity and PUE
- **Stage-wise protocol** — explicit start/end timestamps per pipeline stage so you know *where* the energy actually goes

Built for the [ICML 2026 paper](https://arxiv.org/abs/2605.13981) on end-to-end distillation energy accounting. Also works on any ML training or inference workload

---

## What gets measured

Each run is decomposed into disjoint stages with explicit timestamps:

| Stage | What it covers |
|---|---|
| `prerun` | Environment stabilization and smoke test; validates logging before any real work |
| `teacher` | Teacher-side compute — synthetic data generation (`gen`) or logit caching (`logit`) |
| `student` | Student training (SFT, KD, or synthetic SFT) |
| `eval` | Benchmark evaluation suite (GSM8K, MMLU, IFEval, AlpacaEval 2, MT-Bench-101) |

For each stage the logger records wall-clock time, tokens processed, GPU energy (kWh), CPU energy, and derived CO₂e. Outputs are aggregated into stage-wise summaries and a full pipeline total.

---
## Repository structure

```
Energy/
├── run_experiment.py           # Single entrypoint for all pipelines and data scripts
├── configs/
│   ├── base.yaml               # Fixed defaults: seed, optimizer, energy logging
│   └── experiments/            # Per-run overrides (pipeline, teacher/student, paths)
├── distill_bench/
│   ├── core/
│   │   └── energy_logger.py    # ← Energy tracking interface (NVML + CPU + CO₂e)
│   ├── pipelines/
│   │   ├── kd_main.py          # KD training loop
│   │   └── sft_main.py         # SFT / synthetic SFT training loop
│   └── data/                   # Data, preprocessing, and benchmark scripts
├── run_kd_32b_to_{1b,7b,13b}.sh
├── run_sft_32b_to_{1b,7b,13b}.sh
├── run_pipeline.sh
└── LAUNCH_GUIDE.md             # Quick reference for cluster launch commands
```

---

## Setup

```bash
git clone https://github.com/StellarLuminosity/Energy.git
cd Energy

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Optional: evaluation extras (lm-eval-harness, alpaca_eval, mt-bench-101)
pip install -e .[eval]
```

**Requirements:**
- Python 3.10+
- NVIDIA GPU with recent drivers (NVML required for energy logging)
- PyTorch with CUDA matching your driver
- Dataset and model paths - defaults in `configs/base.yaml` use `/scratch/...` placeholders; override for your filesystem

---

## Running the energy harness

### Smoke test (validate logging first)

```bash
python run_experiment.py \
  --config configs/experiments/sft_32b_to_1b.yaml \
  --data-script prerun \
  --run-dir /tmp/prerun_test
```

This stabilizes the GPU environment and confirms NVML is reading correctly before any real workload.

### Preprocessing / data scripts

```bash
# Instruction following (Tulu)
python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml \
  --data-script tulu_preprocess_dataset

# Math (OpenR1)
python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml \
  --data-script openr1_math_preprocess_dataset

# Code (Codeforces)
python run_experiment.py --config configs/experiments/sft_32b_to_13b.yaml \
  --data-script codeforces_preprocess_dataset

# Generate teacher logits (KD)
python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml \
  --data-script logit_caching

# Generate synthetic data (synthetic SFT)
python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml \
  --data-script synthetic_generation
```

### Training

**Knowledge Distillation (KD):**

```bash
# 1B student
python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml \
  --run-dir /your/output/kd_1b

# 7B student
python run_experiment.py --config configs/experiments/kd_32b_to_7b.yaml \
  --run-dir /your/output/kd_7b

# 13B student
python run_experiment.py --config configs/experiments/kd_32b_to_13b.yaml \
  --run-dir /your/output/kd_13b
```

**Supervised Fine-Tuning (baseline or synthetic):**

```bash
# Plain SFT — 1B / 7B / 13B
python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml --run-dir /your/output/sft_1b
python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml --run-dir /your/output/sft_7b
python run_experiment.py --config configs/experiments/sft_32b_to_13b.yaml --run-dir /your/output/sft_13b

# Synthetic SFT (generate then train)
python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml \
  --data-script synthetic_generation --run-dir /your/output/synth_7b
```

### Benchmark evaluation (energy-tracked)

Evaluate any model or checkpoint with full energy accounting across GSM8K, MMLU, IFEval, AlpacaEval 2, and MT-Bench-101:

```bash
# Dry run — list available tasks without executing
python distill_bench/data/olmo_benchmark.py \
  --config configs/experiments/eval_olmo2_1b.yaml \
  --tasks list --run-dir /tmp/bench --dry-run

# Run a smoke-test subset
python distill_bench/data/olmo_benchmark.py \
  --config configs/experiments/eval_olmo2_1b.yaml \
  --tasks gsm8k,mmlu,alpaca_eval \
  --max-samples 2 \
  --run-dir /tmp/bench_run

# Full evaluation — 7B
python run_experiment.py \
  --config configs/experiments/kd_32b_to_7b.yaml \
  --data-script olmo_benchmark \
  --run-dir /your/output/eval_7b
```

`benchmark.model` in the config accepts a HuggingFace model ID, a local HF directory, or a checkpoint file (auto-converted). Results are under `--run-dir` as per-task JSON files `benchmark_summary.json`.

---

## Output artifacts

Every run produces a self-contained log directory:

```
run-dir/
├── config_merged.yaml          # Exact config used (reproducibility)
├── environment_snapshot.json   # Hardware + software versions
├── experiment_summary.json     # Top-level energy and quality totals
├── logs/
│   ├── stages/                 # Per-stage energy summaries (kWh, J/tok, wall time)
│   └── codecarbon/             # CodeCarbon CSV files for CPU + CO₂e
└── checkpoints/
    ├── final_model/            # HF-formatted student weights
    └── final_policy/           # Raw weights (KD runs also store intermediate)
```

---

## Citation

If you use this harness or the accounting protocol, please cite:

```bibtex
@inproceedings{lambert2026distillation,
  title     = {Towards Resource-Efficient {LLM}s: End-to-End Energy Accounting of Distillation Pipelines},
  author    = {Lambert, Katherine and Luccioni, Sasha},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2605.13981}
}
```

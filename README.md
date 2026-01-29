Distillation Energy Benchmark
=============================

What this repo is
-----------------
Reference distillation **benchmark and evaluation protocol** aimed at the community (and our paper submission): single-GPU harness that measures quality/throughput/energy trade-offs and is intended to be cited as the standard for reporting distillation cost. It supports:
- **Knowledge Distillation (KD)**: train students from cached teacher logits (CE + KL).
- **Synthetic SFT**: train on teacher-generated data with optional filtering/decoding ablations.
- **Benchmark harness**: run GSM8K, MMLU, IFEval, AlpacaEval 2, MT-Bench-101, and OLMES tasks with per-task energy tracking.

Key structure (where things live)
---------------------------------
- `run_experiment.py`: one entrypoint; chooses KD/SFT or a data/benchmark script via `--data-script`.
- `configs/base.yaml`: fixed defaults (seed, token budget, optimizer, energy logging, dataset paths).
- `configs/experiments/*.yaml`: per-run overrides (pipeline type, teacher/student, output dirs, benchmark defaults).
- `distill_bench/pipelines/`: training loops (`kd_main.py`, `sft_main.py`).
- `distill_bench/data/`: data + benchmark scripts (`logit_caching`, `synthetic_generation`, `olmo_benchmark`, etc.).
- `distill_bench/core/`: utilities (energy logger, environment capture, config loader, trainer abstractions).
- `logs/` (or `--run-dir`): run artifacts (configs, checkpoints, metrics, energy traces).

Requirements
------------
- Python 3.10+ and a GPU with recent NVIDIA drivers (NVML needed for energy logging).
- `torch` with CUDA matching your driver.
- Datasets and model checkpoints must be reachable; default paths in `configs/base.yaml` use `/scratch/...` placeholders—override them for your environment.

Setup
-----
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Core deps
pip install -e .
# Optional eval extras (lm-eval-harness, alpaca_eval, mt-bench-101, jsonlines)
pip install -e .[eval]
```

Quickstart (local GPU)
----------------------
- KD train: `python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml --run-dir /tmp/kd_run`
- SFT train: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml --run-dir /tmp/sft_run`
- Data-only preprocessing/generation: add `--data-script logit_caching` (or `tulu_preprocess_dataset`, `codeforces_preprocess_dataset`, `openr1_math_preprocess_dataset`, `synthetic_generation`, `preference_dataset`, `prerun`).

Benchmark-only harness
----------------------
Evaluate a model or checkpoint without training:
```bash
# List tasks without running
python distill_bench/data/olmo_benchmark.py --config configs/experiments/eval_olmo2_1b.yaml --tasks list --run-dir /tmp/bench --dry-run

# Run a small subset
python distill_bench/data/olmo_benchmark.py \
  --config configs/experiments/eval_olmo2_1b.yaml \
  --tasks gsm8k,mmlu,alpaca_eval \
  --max-samples 2 \
  --run-dir /tmp/bench_run
```
`benchmark.model` in the config can be a HF model id, a local HF directory, or a checkpoint file (it will be auto-converted). Outputs land under the chosen `--run-dir` with per-task JSON plus `benchmark_summary.json`.

Reproducibility & logging
-------------------------
- Seeds are fixed in `configs/base.yaml` (default 42); override per-experiment if needed.
- Energy tracking is on by default: GPU power via NVML, optional CPU via RAPL, CodeCarbon estimates; interval set by `energy.nvml_poll_interval_ms`.
- W&B logging is enabled by default (project `distillation-energy-benchmark`); set `WANDB_ENTITY`/config to route logs or disable via `wandb.enabled=false`.

How energy is accounted (stage-wise protocol)
---------------------------------------------
This harness is designed as an evaluation **standard**: total distillation energy = teacher-side work + student training + evaluation, logged as explicit stages with start/end timestamps:
- **Prerun** — smoke test to stabilize the environment and validate logging.
- **Teacher** — synthetic-data generation (**gen**) or logit caching (**logit**) depending on SFT or KD.
- **Student** — the student training phase (shared across KD/SFT).
- **Eval** — core and auxiliary evaluation suites for quality metrics.

For each stage we log wall-clock time, token counts, and energy, then aggregate into stage-wise and pipeline totals. These numbers are meant for accurate, reproducible reporting and for cost/quality Pareto frontiers in the paper and future work.

Artifacts
---------
- Merged config used for the run.
- Environment snapshot (hardware + software).
- Energy logs: stage summaries under `logs/stages/`, CodeCarbon CSVs under `logs/codecarbon/`, and run-level `experiment_summary.json` or `benchmark_summary.json`.
- Checkpoints: periodic + final student outputs (`final_model/` or `final_policy/`); KD stores both raw and HF-formatted weights when available.

Limitations / tips
------------------
- Default dataset and output paths assume the AIP cluster; change them for your filesystem.
- Large benchmarks can be slow; use `--max-samples` and `--tasks` to smoke-test first.
- If energy tracking fails (e.g., no NVML), set `energy.enabled=false` to avoid interruptions.

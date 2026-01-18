Energy
Distillation Energy Benchmark
================================

This repo runs standardized distillation experiments and measures the energy/quality/throughput trade-offs. It compares three pipelines: knowledge distillation (KD, teacher logits), synthetic data SFT (teacher generations), and preference distillation (DPO, judge preferences) with consistent hardware, token budgets, and logging. Teacher-side costs (generation, judging, evaluation) are included so you can see the full energy bill.

Pipelines at a glance
---------------------
- Knowledge Distillation (KD): cache teacher logits over a fixed corpus, then train the student with CE + KL loss (temperature τ, weight α).
- Data / Sequence Distillation (Synthetic SFT): teacher generates responses or CoT traces; students train with standard SFT on the synthetic set, with optional filtering and decoding ablations.
- Preference Distillation (DPO): teacher/judge produces preference pairs (chosen vs rejected); students train with a DPO objective (beta, judge variants).
- Optional self-distillation: treat a previous student checkpoint as the teacher and compare against straight additional training.

Project layout
--------------
- run_experiment.py: dispatches to KD/SFT/DPO pipelines or data-only scripts via `--data-script`.
- run_pipeline.sh: SLURM batch wrapper (single-GPU defaults) that activates the venv and launches `run_experiment.py`.
- configs/base.yaml: fixed settings for fair comparison (seed, token budget, optimizer, energy logging defaults, datasets).
- configs/experiments/*.yaml: per-run overrides for KD/SFT/DPO (teacher/student models, beta/alpha/temperature, output dirs).
- distill_bench/pipelines/: main training loops for kd_main.py, sft_main.py, dpo_main.py.
- distill_bench/data/: preprocessing and generation scripts (logit caching, synthetic generation, preference dataset, Tulu/Codeforces/OpenR1 preprocess).
- distill_bench/core/: shared utilities (energy_logger.py, environment capture, trainer abstractions, config loader).
- logs/: default run_dir where stage outputs, codecarbon CSVs, and summaries are written.

Quickstart (launch & more detail in LAUNCH_GUIDE.md)
----------------------------------------------------
- SLURM batch: `sbatch run_pipeline.sh configs/experiments/kd_32b_to_1b.yaml` (optionally add `--data-script logit_caching` or other data steps first).
- Interactive: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml [--data-script synthetic_generation] [--run-dir /scratch/you/run123]`.
- Data scripts: pass `--data-script` for preprocessing or generation (`logit_caching`, `tulu_preprocess_dataset`, `codeforces_preprocess_dataset`, `openr1_math_preprocess_dataset`, `synthetic_generation`, `preference_dataset`, `prerun`).
- Run order guidance, partitions, and output paths are in LAUNCH_GUIDE.md.

Datasets and models
-------------------
- Datasets (see configs/base.yaml): Tulu-3 SFT mixture (default), Codeforces COTS, OpenR1-Math; preprocessed paths are referenced in configs.
- Models: teacher ~13B (and optional larger judge/teacher); students at ~1B, ~7B, ~13B; tokenizer defaults come from the student model unless overridden.
- Hardware regimes: single-GPU SLURM presets for H100 or L40S, with optional power caps if you want comparability across regimes.

What is tracked (energy + metrics)
----------------------------------
- Energy tracking: NVML polling for GPU power, optional RAPL for CPU, plus CodeCarbon for CO₂/energy estimates; sampling interval set in configs/base.yaml.
- Stage metrics: per-stage kWh/J, average/peak power, tokens/sec, tokens processed; total energy policy selectable (measured/codecarbon/gpu_only).
- Experiment metadata: seeds, hyperparameters, hardware snapshot (GPU/CPU details, software versions), and W&B logging to project `distillation-energy-benchmark` if enabled.

What is saved
-------------
- Configs: the merged config used for the run plus any overrides; one config file per run in the run_dir.
- Environment snapshot: hardware/software metadata saved alongside runs (see distill_bench/core/environment.py).
- Energy logs: stage JSON summaries under `logs/stages/`, CodeCarbon CSVs under `logs/codecarbon/`, and `experiment_summary.json` per run.
- Models and checkpoints: periodic checkpoints and final student outputs under the run’s output_dir (`final_model/` or `final_policy/`); KD stores `model.pt` plus HF format.
- Outputs directory: defaults to `logs/` unless overridden via `--run-dir` or `output.output_dir` in the experiment config.

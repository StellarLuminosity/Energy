# Distillation Experiment Launch Guide

Short commands and output pointers for the KD, SFT, and DPO pipelines.

## Prerequisites
- Data/caches live at the paths already in configs (no downloads):  
  - KD logits: `/scratch/klambert/dataset/logprob_cache`  
  - Prerun baseline writes to `<output_dir>/prerun_validation` when enabled.  
  - SFT synthetic dataset: `/scratch/klambert/dataset/synthetic_7b_to_1b` (auto-generated if missing and `use_existing: false`).  
  - DPO preference dataset cached at `<output_dir>/preference_dataset` if not already present.
- Energy logs write to `output.run_dir` from `configs/base.yaml` (default: `logs/` in repo). Make sure it is writable.
- SLURM environment: `run_pipeline.sh` loads `gcc`, `arrow/18.1.0`, and activates `/home/klambert/projects/aip-craffel/shared/slm_ensemble/prj/bin/activate`. Use the same env if running interactively.

## Launch Commands
- SLURM (recommended):  
  - Pipelines: `sbatch run_pipeline.sh configs/experiments/kd_7b_to_1b.yaml` (or sft/dpo configs)  
  - Data scripts via dispatcher: `sbatch run_pipeline.sh configs/experiments/kd_7b_to_1b.yaml --data-script logit_caching`
- Interactive (single GPU):  
  - Pipelines: `python run_experiment.py --config <config.yaml>`  
  - Data scripts: `python run_experiment.py --config <config.yaml> --data-script <name_or_path>`
- `--data-script` accepts a basename (e.g., `logit_caching`, `preference_dataset`, `synthetic_generation`) or a path to a data script; it bypasses the training pipeline and runs that script’s `__main__`.

## Pipeline-Specific Notes
- **KD (`configs/experiments/kd_7b_to_1b.yaml`)**  
  - Requires cached teacher logits at `distillation.logprob_cache_path`.  
  - Energy stage: `student_train`.  
  - Outputs: checkpoints in `<output_dir>/checkpoints`, final model in `<output_dir>/final_model`.
- **SFT / Data Distillation (`configs/experiments/sft_7b_to_1b.yaml`)**  
  - If `synthetic_dataset_path` exists, it is reused; otherwise, teacher generation runs (stage `teacher_generation`) before training (`student_train`).  
  - Outputs: synthetic dataset saved to `synthetic_dataset_path`, final model in `<output_dir>/final_model`.
- **DPO (`configs/experiments/dpo_7b_to_1b.yaml`)**  
  - Builds or loads preference pairs; generation is tracked as `teacher_generation` if needed. Training stage is `dpo_training`.  
  - Outputs: preference dataset cached at `<output_dir>/preference_dataset`, final policy in `<output_dir>/final_policy`.

## Where to Look for Results
- SLURM logs: `/scratch/klambert/run_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}`.
- W&B: project `distillation-energy-benchmark`, run names set in each config (overridable). Contains losses, throughput, and energy metrics from `EnergyTracker`.
- Energy logs (per run_dir):  
  - Summary: `<output.run_dir>/experiment_summary.json` (total/stage energy, tokens, throughput, joules_per_token).  
  - Stage details: `<output.run_dir>/stages/<stage>.json` plus CodeCarbon CSVs under `<output.run_dir>/codecarbon/`.  
  - Use `jq '.stages.student_train' <summary>` (or other stage ids) to inspect.
- Model artifacts: checkpoints + `final_*` folders under each experiment’s `output.output_dir`.

## Metrics and Interpretation
- Key fields: `total_energy_kwh`, per-stage `gpu_energy_joules`/`cpu_energy_joules`, `tokens_processed`, `tokens_per_second`, `joules_per_token`.  
- Compare pipelines by stage energy:  
  - KD: single `student_train` stage.  
  - SFT: `teacher_generation` vs `student_train` shows teacher vs student cost split.  
  - DPO: `teacher_generation` (label generation) vs `dpo_training`.
- Quality/learning signals live in W&B and console logs: train/eval loss curves, checkpoint losses, and any eval hooks you add.  
- Throughput vs energy: higher `tokens_per_second` with lower `joules_per_token` indicates more efficient training; track both alongside validation loss to pick Pareto-efficient runs.

## Dataset-Specific Notes
- `logit_caching` (KD prep): reads cached/preprocessed Tulu data from `distillation.logprob_cache_path` and writes consolidated `teacher_logprobs` under that path; energy stage name `logit_caching`.
- `synthetic_generation` (SFT prep): generates or reuses synthetic data at `synthetic_data.synthetic_dataset_path`; when run via dispatcher, energy stage name `synthetic_generation` (pipeline uses `teacher_generation`).
- `preference_dataset` (DPO prep): builds or loads preference pairs at `dpo_preference_dataset_path` or `<output_dir>/preference_dataset`; energy stage name `preference_dataset` for standalone runs (pipeline uses `teacher_generation`).

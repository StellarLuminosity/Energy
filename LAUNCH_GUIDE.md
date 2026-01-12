# Distillation Experiment Launch Guide

## Quick launcher
- SLURM: `sbatch run_pipeline.sh <config.yaml> [--data-script <name_or_path>]`
- Interactive: `python run_experiment.py --config <config.yaml> [--data-script <name_or_path>]`
- `--data-script` runs the data script only (e.g., `synthetic_generation`, `preference_dataset`, `logit_caching`, `tulu_preprocess_dataset`, `prerun`).
- Prereqs: logits already cached at `/scratch/klambert/dataset/logprob_cache`; synthetic dataset path and preference dataset path are read from configs. Energy logs land in `logs/` by default.

## Run order
1) Data generation: `sbatch run_pipeline.sh configs/experiments/sft_32b_to_1b.yaml --data-script synthetic_generation`, `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_1b.yaml --data-script preference_dataset`, `sbatch run_pipeline.sh configs/experiments/kd_32b_to_1b.yaml --data-script logit_caching`
2) KD training (parallel): `sbatch run_pipeline.sh configs/experiments/kd_32b_to_1b.yaml`, `sbatch run_pipeline.sh configs/experiments/kd_32b_to_7b.yaml`, `sbatch run_pipeline.sh configs/experiments/kd_32b_to_13b.yaml`
3) SFT training (after synthetic data is ready; parallel): `sbatch run_pipeline.sh configs/experiments/sft_32b_to_1b.yaml`, `sbatch run_pipeline.sh configs/experiments/sft_32b_to_7b.yaml`, `sbatch run_pipeline.sh configs/experiments/sft_32b_to_32b.yaml`
4) DPO training (after preference dataset is ready; parallel): `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_1b.yaml`, `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_7b.yaml`, `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_32b.yaml`

## Outputs to check
- SLURM logs: `/scratch/klambert/run_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}`
- Energy + metrics per run: `<output.run_dir>/experiment_summary.json`, stage details in `<output.run_dir>/stages/`, CodeCarbon CSVs in `<output.run_dir>/codecarbon/`
- Models and artifacts: checkpoints and `final_*` folders under each run’s `output.output_dir`; W&B project `distillation-energy-benchmark` for curves/throughput/energy tracking

# Distillation Launch Guide

## Data generation (in order)
- Tulu: `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml --data-script tulu_preprocess_dataset`
- Math: `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml --data-script openr1_math_preprocess_dataset`
- Codeforces: `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml --data-script codeforces_preprocess_dataset`

Choices for `-data-script`: `synthetic_generation`, `preference_dataset`, `logit_caching`, `tulu_preprocess_dataset`, `codeforces_preprocess_dataset`

## KD training
- `sbatch run_kd_32b_to_1b.sh configs/experiments/kd_32b_to_1b.yaml`
- `sbatch run_kd_32b_to_7b.sh configs/experiments/kd_32b_to_7b.yaml`
- `sbatch run_kd_32b_to_13b.sh configs/experiments/kd_32b_to_13b.yaml`

## SFT training
- Synthetic data (1B): `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml --data-script synthetic_generation`
- Synthetic data (7B): `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml --data-script synthetic_generation`
- Synthetic data (13B): `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml --data-script synthetic_generation`
- `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml`
- `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml`
- `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml`

## DPO training
- `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_1b.yaml`
- `sbatch run_pipeline.sh configs/experiments/dpo_32b_to_7b.yaml`

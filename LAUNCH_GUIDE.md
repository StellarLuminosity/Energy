# Distillation Launch Guide

Activate first if running interactively (skip `sbatch`): `source /home/klambert/projects/aip-craffel/klambert/Energy/.venv/bin/activate`

## Data scripts / preprocessing (run first)
- Tulu: `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml --data-script tulu_preprocess_dataset`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml --data-script tulu_preprocess_dataset`
- Math: `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml --data-script openr1_math_preprocess_dataset`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml --data-script openr1_math_preprocess_dataset`
- Codeforces: `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml --data-script codeforces_preprocess_dataset`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_13b.yaml --data-script codeforces_preprocess_dataset`
- Other data-only scripts: use the config you want plus one of
  `logit_caching`, `synthetic_generation`, `preference_dataset`, `tulu_preprocess_dataset`, `codeforces_preprocess_dataset`, `openr1_math_preprocess_dataset`, `prerun`.

## KD training
- 1B student: `sbatch run_kd_32b_to_1b.sh configs/experiments/kd_32b_to_1b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml [--run-dir /scratch/you/run123]`
- 7B student: `sbatch run_kd_32b_to_7b.sh configs/experiments/kd_32b_to_7b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_7b.yaml [--run-dir /scratch/you/run123]`
- 13B student: `sbatch run_kd_32b_to_13b.sh configs/experiments/kd_32b_to_13b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_13b.yaml [--run-dir /scratch/you/run123]`

## SFT training
- Synthetic data (1B): `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml --data-script synthetic_generation`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml --data-script synthetic_generation [--run-dir /scratch/you/run123]`
- Synthetic data (7B): `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml --data-script synthetic_generation`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml --data-script synthetic_generation [--run-dir /scratch/you/run123]`
- Synthetic data (13B): `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml --data-script synthetic_generation`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_13b.yaml --data-script synthetic_generation [--run-dir /scratch/you/run123]`
- Plain SFT (13B): `sbatch run_sft_32b_to_13b.sh configs/experiments/sft_32b_to_13b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_13b.yaml [--run-dir /scratch/you/run123]`
- Plain SFT (1B): `sbatch run_sft_32b_to_1b.sh configs/experiments/sft_32b_to_1b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml [--run-dir /scratch/you/run123]`
- Plain SFT (7B): `sbatch run_sft_32b_to_7b.sh configs/experiments/sft_32b_to_7b.yaml`
  - Direct: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml [--run-dir /scratch/you/run123]`

## Benchmark evaluation (Olmo)
- 1B: `sbatch run_kd_32b_to_1b.sh configs/experiments/kd_32b_to_1b.yaml --data-script olmo_benchmark`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml --data-script olmo_benchmark [--run-dir /scratch/you/run123]`
- 7B: `sbatch run_kd_32b_to_7b.sh configs/experiments/kd_32b_to_7b.yaml --data-script olmo_benchmark`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_7b.yaml --data-script olmo_benchmark [--run-dir /scratch/you/run123]`
- 13B: `sbatch run_kd_32b_to_13b.sh configs/experiments/kd_32b_to_13b.yaml --data-script olmo_benchmark`
  - Direct: `python run_experiment.py --config configs/experiments/kd_32b_to_13b.yaml --data-script olmo_benchmark [--run-dir /scratch/you/run123]`

## Direct launch with `run_experiment.py` (skip `sbatch`)
If you already have an interactive shell on a GPU node (e.g., via `srun --pty ... bash`) and want to avoid SLURM batch scripts:

1) Activate the venv: `source /home/klambert/projects/aip-craffel/klambert/Energy/.venv/bin/activate`
2) Run the desired command:
   - Data scripts only: `python run_experiment.py --config configs/experiments/sft_32b_to_1b.yaml --data-script synthetic_generation`
   - KD training: `python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml [--run-dir /scratch/you/run123]`
   - SFT training: `python run_experiment.py --config configs/experiments/sft_32b_to_7b.yaml [--run-dir /scratch/you/run123]`
   - Benchmark eval (Olmo): `python run_experiment.py --config configs/experiments/kd_32b_to_1b.yaml --data-script olmo_benchmark`

Notes:
- Pick any config under `configs/experiments/*.yaml`; the pipeline (kd/sft) is chosen from the config itself.
- Use `--data-script` to run preprocessing/generation only (`logit_caching`, `tulu_preprocess_dataset`, `codeforces_preprocess_dataset`, `openr1_math_preprocess_dataset`, `synthetic_generation`, `preference_dataset`, `prerun`, `olmo_benchmark`).
- Add `--run-dir ...` to override the output folder for a one-off run; extra args after the command are forwarded to the underlying script when applicable.

#!/bin/bash
#SBATCH --job-name=tulu_dataset_preprocess
#SBATCH --exclusive
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out                
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err 
#SBATCH --partition=gpubase_h100_b2                                             
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=120GB
#SBATCH --account=aip-craffel                                             
#SBATCH --time=6:00:00

# Unified experiment launcher for KD/SFT/DPO pipelines (single-GPU)
# srun --exclusive -c 4 --gres=gpu:l40s:1 --partition=gpubase_l40s_b3 --mem=120GB --pty --time=7:00:00 --account=aip-craffel bash
# srun -c 4 --gres=gpu:h100:1 --partition=gpubase_h100_b1 --mem=120GB --pty --time=3:00:00 --account=aip-craffel bash

# Get config path and extra args
CONFIG_PATH=${1:-"configs/experiments/kd_7b_to_1b.yaml"}
EXTRA_ARGS="${@:2}"

echo "==============================================="
echo "Distillation Energy Benchmark"
echo "==============================================="
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_PATH"
echo "GPU resources: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Extra args: $EXTRA_ARGS"
echo "==============================================="

# Export hardware metadata for energy tracking
export SLURM_JOB_ID=$SLURM_JOB_ID
export SLURM_JOB_NAME=$SLURM_JOB_NAME
export SLURM_NODELIST=$SLURM_NODELIST

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules
module load gcc arrow/18.1.0
source /home/klambert/projects/aip-craffel/klambert/Energy/.venv/bin/activate

# Run experiment or data script
python run_experiment.py --config "$CONFIG_PATH" $EXTRA_ARGS

EXIT_CODE=$?

echo "==============================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="

exit $EXIT_CODE

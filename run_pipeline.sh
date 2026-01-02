#!/bin/bash
#SBATCH --job-name=distill_exp
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out                
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err 
#SBATCH --partition=gpubase_l40s_b3                                                
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=60GB
#SBATCH --account=aip-craffel                                             
#SBATCH --time=11:58:00

# Unified experiment launcher for KD/SFT/DPO pipelines (single-GPU)
#
# Usage:
#   sbatch run_pipeline.sh configs/experiments/kd_7b_to_1b.yaml
#   sbatch run_pipeline.sh configs/experiments/sft_7b_to_1b.yaml --run-prerun

# Get config path from command line argument
CONFIG_PATH=${1:-"configs/experiments/kd_7b_to_1b.yaml"}
EXTRA_ARGS="${@:2}"  # All arguments after the first

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
source /home/klambert/projects/aip-craffel/shared/slm_ensemble/prj/bin/activate

# Run experiment (single-process)
python run_experiment.py --config "$CONFIG_PATH" $EXTRA_ARGS

EXIT_CODE=$?

echo "==============================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="

exit $EXIT_CODE

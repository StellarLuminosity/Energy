#!/bin/bash
#SBATCH --job-name=sft_13b_distill
#SBATCH --exclusive
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out                
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err                                            
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH --account=def-lylan                                
#SBATCH --time=9:00:00

# Unified experiment launcher for KD/SFT/DPO pipelines (single-GPU)
# 1 H100:        srun -c 16 --gres=gpu:h100:1 --partition=gpubase_h100_b5 --mem=120GB --pty --time=3:00:00 --account=aip-craffel bash
# 1 L40:         srun -c 1 --gres=gpu:l40s:1 --partition=gpubase_l40s_b2 --mem=120GB --pty --time=3:00:00 --account=aip-craffel bash
# CPU-only:      srun -c 16 --partition=gpubase_h100_b1 --mem=120GB --pty --time=3:00:00 --account=aip-craffel bash
# Example override of run_dir:
#   bash run_pipeline.sh configs/experiments/dpo_32b_to_13b.yaml --run-dir /tmp/my_run

set -e
set -x

# Get config path and extra args
CONFIG_PATH=${1:-"configs/experiments/sft_32b_to_13b.yaml"}
EXTRA_ARGS="${@:2}"

echo "==============================================="
echo "Distillation Energy Benchmark"
echo "==============================================="
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_PATH"
echo "GPU resources: ${CUDA_VISIBLE_DEVICES}"
echo "Extra args: $EXTRA_ARGS"
echo "==============================================="

# Export hardware metadata for energy tracking
export SLURM_JOB_ID=$SLURM_JOB_ID
export SLURM_JOB_NAME=$SLURM_JOB_NAME
export SLURM_NODELIST=$SLURM_NODELIST

# Device Settings
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules
module load gcc python/3.11 arrow/21
source /home/klambert/.venv/bin/activate

# Show which Python we're actually running and whether torch is visible
echo "Python in batch job:"
which python
python -c "import torch; print('torch version in batch job =', torch.__version__)"

# Run experiment or data script
python run_experiment.py --config "$CONFIG_PATH" $EXTRA_ARGS

EXIT_CODE=$?

echo "==============================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="

exit $EXIT_CODE

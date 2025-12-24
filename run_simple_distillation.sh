#!/bin/bash

#SBATCH --job-name=alpha1
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out                
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err 
#SBATCH --partition=gpubase_l40s_b3                                                
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=120GB
#SBATCH --account=aip-craffel                                             
#SBATCH --time=11:58:00

# srun -c 4 --gres=gpu:l40s:2 --partition=gpubase_l40s_b2 --mem=120GB --pty --time=6:00:00 --account=aip-craffel bash

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES" 

# Auto-detect number of GPUs
# Try SLURM variable first, fallback to counting CUDA_VISIBLE_DEVICES
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    export WORLD_SIZE=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count GPUs from CUDA_VISIBLE_DEVICES (comma-separated)
    export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # Fallback: use nvidia-smi to count GPUs
    export WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
fi

echo "Detected $WORLD_SIZE GPU(s) for training"

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_LAUNCH_BLOCKING removed for better performance

# NCCL optimization settings
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes
export NCCL_DEBUG=INFO    # Enable debugging
export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues  

# Load modules
module load gcc arrow/18.1.0
source /home/klambert/projects/aip-craffel/shared/slm_ensemble/prj/bin/activate

# Run training
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    src_simple/main_simple.py \
    --mixed-precision

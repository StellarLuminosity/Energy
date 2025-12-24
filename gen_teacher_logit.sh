#!/bin/bash

#SBATCH --job-name=gen_teacher_logit
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out                
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err 
#SBATCH --partition=gpubase_l40s_b3                                                             
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=120GB
#SBATCH --account=aip-craffel                                             
#SBATCH --time=23:58:00

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES" 

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
python -u /home/klambert/projects/aip-craffel/klambert/SLMensembles/src_simple/logit_caching.py

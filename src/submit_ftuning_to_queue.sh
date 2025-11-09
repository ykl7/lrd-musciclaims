#!/bin/bash

#=========================================================================================
# SLURM JOB-DEFINITION
#=========================================================================================
#SBATCH --job-name=qwen-finetune-3gpu-robust
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24                   # Adjusted for 3 GPUs
#SBATCH --gres=gpu:4                         # ✅ We still ASK for 4 to get the node
#SBATCH --mem=128G
#SBATCH --time=07:59:00
#SBATCH --output=final-run-%j.out
#SBATCH --error=final-run-%j.err

#=========================================================================================
# ENVIRONMENT SETUP & DIAGNOSTICS
#=========================================================================================
echo "================================================================================"
echo "Date:               $(date)"
echo "Host:               $(hostname)"
echo "Job ID:             $SLURM_JOB_ID"
echo "Allocated GPUs by Slurm: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=2,3
echo "Overriding environment. Now using GPUs: $CUDA_VISIBLE_DEVICES"

# Activate your Python virtual environment.
source /gpfs/projects/MaffeiGroup/lrd_uv_p311_venv/bin/activate
echo "Python environment activated."


# APPLICATION LAUNCH
cd /gpfs/projects/MaffeiGroup/lrd-musciclaims/src/
echo "Current directory: $(pwd)"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
echo "Using MASTER_PORT=$MASTER_PORT"

echo "Starting torchrun on 2 healthy GPUs..."
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_judgment_only_v2.py

echo "================================================================================"
echo "Job finished with exit code $? at: $(date)"
echo "================================================================================"

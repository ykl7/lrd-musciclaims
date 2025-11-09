#!/bin/bash

#SBATCH --job-name=env-check
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=check-%j.out
#SBATCH --error=check-%j.err

echo "========= Starting Environment Check on Host: $(hostname) =========="
nvidia-smi

# Activate your environment
source /gpfs/projects/MaffeiGroup/lrd_uv_p311_venv/bin/activate

# Run the diagnostic script with torchrun
torchrun --nproc_per_node=4 check_env.py
echo "========= Environment Check Finished =========="

#!/bin/bash

#SBATCH --job-name=vllm-server
#SBATCH --partition=a100-long              # Use the a100 queue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # Request 1 GPU (vLLM can use 1 or more)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:00:00               # 8 hour max runtime
#SBATCH --output=vllm-server-%j.log
#SBATCH --error=vllm-server-%j.err

echo "================================================================================"
echo "🚀 Starting vLLM Server"
echo "================================================================================"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# Show GPU status
nvidia-smi

# Activate your Python environment
source /gpfs/projects/MaffeiGroup/lrd_uv_p311_venv/bin/activate
echo "✅ Python environment activated"

# Define the port (choose a unique port between 8000-9000 to avoid conflicts)
export VLLM_PORT=8003

# Get the hostname (this is what you'll connect to)
export VLLM_HOST=$(hostname)

echo ""
echo "================================================================================"
echo "📡 vLLM Server will be accessible at:"
echo "   Internal: http://${VLLM_HOST}:${VLLM_PORT}"
echo "   You will need to SSH tunnel to this from your local machine"
echo "================================================================================"
echo ""

# Launch vLLM server
# Adjust model name and parameters as needed
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust-remote-code

echo "================================================================================"
echo "vLLM server stopped at: $(date)"
echo "================================================================================"

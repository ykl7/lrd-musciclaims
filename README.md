# lrd-musciclaims


# 🚧 Project Status: Fine-Tuning Qwen3-VL (Judgment Task)

**Current State:** ⚠️ **Active / Experimental**

The fine-tuning pipeline for the "Judgment Only" task is currently under development. We are transitioning from full fine-tuning to QLoRA (4-bit quantization) on Qwen3-VL-2B/8B, but are currently resolving stability issues with **Distributed Data Parallel (DDP)** training on the A100 cluster.

### 📂 Key Files
* **`src/ft_label_judgment_only/train_judgment_only_torchrun_v1.py`**
  * **Role:** Main / Legacy Script.
  * **Status:** Baseline implementation (Full Fine-Tuning).
  
* **`src/ft_label_judgment_only/train_judgment_only_torchrun_v3.py`**
  * **Role:** **Current Working File** (Experimental).
  * **Status:** Implements QLoRA (4-bit quantization) + Custom Data Collator.
  * **Current Issues:** Unstable in multi-GPU setup.

### 🐛 Known Issues & Blockers
1. **Distributed Training (DDP) Crash:**
   * **Error:** `RuntimeError: CUDA driver error: invalid argument` at `torch.prod(grid_thw)`.
2. **Data Pipeline Crashes:**
   * **Error:** `IndexError: list index out of range` or `numel overflow`.


#### How to Run
To trigger the experimental QLoRA training on 4 GPUs:

```bash
cd src/ft_label_judgment_only/
torchrun --nproc_per_node=4 train_judgment_only_torchrun_v3.py


## Steps to install flash-attn2

# 1. Create Environment
uv venv venv_flash_opt --python 3.11
source venv_flash_opt/bin/activate

# 2. Install PyTorch (Bundled with CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Flash Attention 2 (Will find the matching pre-built wheel)
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation

# 4. Install the rest
uv pip install transformers accelerate deepspeed peft bitsandbytes wandb pandas scipy pillow scikit-learn sentencepiece protobuf



cd /gpfs/projects/MaffeiGroup/ && cd lrd-musciclaims/src/ &&  source ../../lrd_uv_p311_venv/bin/activate && conda deactivate
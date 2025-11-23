# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "train_data_file": "./../data_splits/train.csv",
    "dev_data_file": "./../data_splits/dev.csv",
    "output_dir": "./qwen3_vl_judgment_lora",
    "max_length": 1024,
    "batch_size": 1, # Reduced to 1 for stability with dynamic resolution
    "gradient_accumulation_steps": 64, # Compensate for batch size 1 (1*64*4 = 256 effective)
    "dataloader_num_workers": 4,
    "gradient_checkpointing": True,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "lora_r": 16,
    "lora_alpha": 16, # r=alpha is standard for QLoRA
    "lora_dropout": 0.05,
    "seed": 42,
    "use_wandb": True,
    "wandb_project": "qwen3-vl-ft-label-only",
    "wandb_run_name": "qwen3-8b-fixed-patch",
    "max_samples": None,
    "eval_every_steps": 500,
    "eval_samples": None,
    "per_device_eval_batch_size": 1,
    'early_stop_patience': 10,
    "early_stop_delta": 0.001,
    "evaluation_strategy": "no", 
    "save_strategy": "epoch",
    "load_best_model_at_end": False,
    "metric_for_best_model": None, 
    "optim": "adamw_torch_fused",
    "dataloader_persistent_workers": False,
    "dataloader_prefetch_factor": 2,
    "dataloader_pin_memory": True,
    'logging_steps': 1,
    'ddp_backend': 'nccl',
}

import os
import torch
import pandas as pd
import warnings
import logging
import json
import types  # Needed for monkey patching
from PIL import Image
from typing import Dict, List, Any
import wandb
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoProcessor,
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration
)
from peft import LoraConfig, get_peft_model

# Fix for Qwen3 import if available
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========================================================================
# 🔧 MONKEY PATCH FOR QWEN3 ROTARY EMBEDDING CRASH
# ========================================================================
def monkey_patch_qwen_rotary(model):
    """
    Replaces the rot_pos_emb function in Qwen3-VL to run grid calculations 
    on the CPU. This fixes the 'CUDA driver error: invalid argument' crash.
    """
    # Locate the visual model
    visual_model = None
    if hasattr(model, "visual"):
        visual_model = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_model = model.model.visual
    
    if visual_model is None:
        print("Warning: Could not find visual model to patch.")
        return

    def safe_rot_pos_emb(self, grid_thw):
        # Move grid to CPU for the calculation that crashes CUDA
        grid_thw_cpu = grid_thw.to("cpu")
        
        # Safe calculation on CPU
        total_tokens = int(torch.prod(grid_thw_cpu, dim=1).sum().item())
        
        # Create output tensor on the correct GPU device
        pos_ids = torch.zeros(total_tokens, 3, dtype=torch.long, device=grid_thw.device)
        
        # Standard Qwen logic for filling pos_ids
        mrope_section = self.mrope_section * 2
        start = 0
        
        # Iterate on CPU values to avoid sync issues
        for i in range(len(grid_thw)):
            t, h, w = grid_thw_cpu[i].tolist()
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w).flatten()
            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1).flatten()
            tpos_ids = torch.arange(t, device=grid_thw.device).unsqueeze(1).expand(-1, h * w).flatten()
            
            end = start + t * h * w
            pos_ids[start:end, 0] = tpos_ids
            pos_ids[start:end, 1] = hpos_ids
            pos_ids[start:end, 2] = wpos_ids
            start = end
            
        cos_list, sin_list = [], []
        for i in range(3):
            freq_pairs = self.rotary_emb.inv_freq[i].expand(pos_ids.shape[0], -1)
            freq_pairs = freq_pairs * pos_ids[:, i].unsqueeze(1)
            rot = torch.cat([freq_pairs, freq_pairs], dim=-1)
            cos_list.append(rot.cos())
            sin_list.append(rot.sin())

        cos = torch.cat(cos_list, dim=-1)
        sin = torch.cat(sin_list, dim=-1)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    # Apply the patch to the instance method
    visual_model.rot_pos_emb = types.MethodType(safe_rot_pos_emb, visual_model)
    print("✅ Applied CPU-offload patch to Qwen3 Visual Rotary Embeddings")


# ========================================================================
# CUDA DEVICE SETUP
# ========================================================================
def setup_cuda_devices():
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Running in distributed mode")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("Running in single-GPU mode")

setup_cuda_devices()

# ============================================================================
# Helper Functions
# ============================================================================
def is_main_process():
    rank = int(os.environ.get('RANK', -1))
    return rank == 0 if rank != -1 else True

def get_rank():
    return int(os.environ.get('RANK', 0))

def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))

logger = logging.getLogger("train")
if is_main_process():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ============================================================================
# Data Loading
# ============================================================================
def load_pre_split_data(train_file, dev_file, seed=42):
    if is_main_process():
        print(f"Loading data from: {train_file} and {dev_file}")

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)

    if is_main_process():
        print(f"Train samples: {len(train_df)}")
        print(f"Dev samples: {len(dev_df)}")

    return train_df, dev_df

# ============================================================================
# Model Setup
# ============================================================================
def setup_model_and_processor(model_name, lora_r, lora_alpha, lora_dropout):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_main_process(): print(f"Loading Processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if is_main_process(): print(f"Loading Model: {model_name}")

    # Try loading specific Qwen3 class, fall back to AutoModel
    if Qwen3VLForConditionalGeneration:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={'': local_rank}
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={'': local_rank}
        )

    model.config.use_cache = False
    
    # --- APPLY PATCH HERE ---
    monkey_patch_qwen_rotary(model)
    # ------------------------

    if CONFIG['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    if is_main_process():
        model.print_trainable_parameters()

    return model, processor

# ============================================================================
# Dataset Class (Updated for Robustness)
# ============================================================================
class MultimodalJudgmentDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join("./../data_collection", row['local_image_path'].lstrip('./'))
        judgment = str(row.get('class', 'NEUTRAL')).strip().upper()
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Create valid dummy black image
            image = Image.new('RGB', (196, 196), (0, 0, 0))
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._create_prompt(str(row.get('panels', '')), str(row.get('new_claim', '')), str(row.get('caption', '')))}
                ]
            },
            {"role": "assistant", "content": [{"type": "text", "text": judgment}]}
        ]
        
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # Pass raw PIL image to processor (it handles resizing for Qwen3)
        inputs = self.processor(
            text=[text],
            images=[image], 
            padding=False,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        answer_tokens = self.processor.tokenizer(judgment, add_special_tokens=False)["input_ids"]
        labels = torch.full_like(input_ids, -100)
        input_list = input_ids.tolist()
        len_ans = len(answer_tokens)
        for i in range(len(input_list) - len_ans + 1):
            if input_list[i:i+len_ans] == answer_tokens:
                labels[i:i+len_ans] = input_ids[i:i+len_ans]
                break

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        
        # Return 3D tensors (1, X, Y) - Collator will concatenate
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values.squeeze(0) if pixel_values is not None else torch.zeros(1, 1176, dtype=torch.bfloat16),
            "image_grid_thw": image_grid_thw if image_grid_thw is not None else torch.tensor([[1, 1, 1]], dtype=torch.long),
            "labels": labels,
        }

    def _create_prompt(self, panels, claim, caption):
        return f"""CLAIM: {claim}\nCAPTION: {caption}\nPANELS: {panels}\nDecision (SUPPORT/CONTRADICT/NEUTRAL):"""

# ============================================================================
# Data Collator (Fixed Concatenation)
# ============================================================================
class MultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for feature in features:
            pad_len = batch_max_len - feature["input_ids"].size(0)
            input_ids.append(torch.cat([feature["input_ids"], torch.full((pad_len,), self.processor.tokenizer.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([feature["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        pixel_values_list = [f["pixel_values"] for f in features]
        image_grid_thw_list = [f["image_grid_thw"] for f in features]

        if pixel_values_list:
            # Concatenate visual tokens (Flattening)
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0).contiguous()
        
        if image_grid_thw_list:
            # Concatenate grids (N_images, 3)
            # Ensure it is Long and Contiguous for the model
            batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0).to(dtype=torch.long).contiguous()

        return batch

# ============================================================================
# Main
# ============================================================================
def train_model(model, processor, train_dataset, dev_dataset, config):
    if is_main_process(): print("🚀 Starting training...")
    
    if config.get("use_wandb", False) and is_main_process():
        wandb.init(project=config["wandb_project"], config=config)

    args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config['logging_steps'],
        save_strategy=config["save_strategy"],
        save_total_limit=2,
        fp16=False, bf16=True,
        report_to="wandb" if (config.get("use_wandb") and is_main_process()) else "none",
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=False,
        gradient_checkpointing=config['gradient_checkpointing'],
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=MultimodalDataCollator(processor),
    )

    trainer.train()
    
    if is_main_process():
        trainer.save_model(os.path.join(config["output_dir"], "final_model"))
        processor.save_pretrained(os.path.join(config["output_dir"], "final_model"))

    return trainer

def main():
    torch.manual_seed(CONFIG["seed"])
    train_df, dev_df = load_pre_split_data(CONFIG["train_data_file"], CONFIG["dev_data_file"])
    model, processor = setup_model_and_processor(CONFIG["model_name"], CONFIG["lora_r"], CONFIG["lora_alpha"], CONFIG["lora_dropout"])
    train_dataset = MultimodalJudgmentDataset(train_df, processor)
    dev_dataset = MultimodalJudgmentDataset(dev_df, processor)
    train_model(model, processor, train_dataset, dev_dataset, CONFIG)

if __name__ == "__main__":
    main()
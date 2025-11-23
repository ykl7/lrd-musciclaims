# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "train_data_file": "./../data_splits/train.csv",
    "dev_data_file": "./../data_splits/dev.csv",
    "output_dir": "./qwen3_vl_judgment_lora",
    "max_length": 1024,
    "batch_size": 32,
    "gradient_accumulation_steps": 16,
    "dataloader_num_workers": 8,
    "gradient_checkpointing": True,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "seed": 42,
    "use_wandb": True,
    "wandb_project": "qwen3-vl-ft-label-only",
    "wandb_run_name": None,
    "max_samples": 10,
    "eval_every_steps":500,
    "eval_samples": None,
    "per_device_eval_batch_size": 1,
    'early_stop_patience': 10,
    "early_stop_delta": 0.001,
    "evaluation_strategy": "no", # steps
    "save_strategy": "epoch",
    "load_best_model_at_end": False,
    "metric_for_best_model": None, # eval_loss
    "optim": "adamw_torch_fused",
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 2,
    "dataloader_pin_memory": True,
    'logging_steps': 1,
    'ddp_backend': 'nccl',
}

LABEL_MAP = {"SUPPORT": 0, "CONTRADICT": 1, "NEUTRAL": 2}
# ===========================================================================


import os

# ========================================================================
# CUDA DEVICE SETUP - Must be before importing torch/unsloth
# ========================================================================
def setup_cuda_devices():
    """Configure CUDA devices for single or multi-GPU training."""
    # Check if running with torchrun (DDP mode)
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        # Multi-GPU mode with torchrun
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"[Rank {rank}] Running in distributed mode (multi-GPU)")
    else:
        # Single GPU mode
        # Clean up any leftover environment variables from previous runs
        for key in ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            os.environ.pop(key, None)
        
        # Use only GPU 0
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("Running in single-GPU mode")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_CACHE_DIR"] = "/gpfs/projects/MaffeiGroup/triton_cache"
os.environ["WANDB_DISABLE_WEAVE"] = "true"
os.environ['TMPDIR'] = '/gpfs/projects/MaffeiGroup/tmp_build_cache'
os.environ["TORCH_EXTENSIONS_DIR"] = "/gpfs/projects/MaffeiGroup/tmp_build_cache/torch_extensions"
os.environ["HF_HOME"] = "/gpfs/projects/MaffeiGroup/tmp_build_cache/huggingface"
os.environ["TRANSFORMERS_NO_TORCHVISION_WARNING"] = "1"

# Run CUDA setup before any other imports
setup_cuda_devices()

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", message=".*use_cache.*")
warnings.filterwarnings("ignore", message="Gradient accumulation steps mismatch")

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)

from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import numpy as np
import json
import wandb
import torch.distributed as dist
from tqdm import tqdm

from collator import MultimodalDataCollator
from custom_dataset import MultimodalJudgmentDataset
# ============================================================================
# Helper Functions
# ============================================================================

def is_main_process():
    """Check if this is the main process in distributed training."""
    # First check environment variables (set by torchrun)
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    # If torchrun is being used, RANK will be set
    if rank != -1:
        return rank == 0
    
    # Fallback to torch.distributed if initialized
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    
    # If neither, assume single process
    return True

def get_rank():
    """Get the rank of this process."""
    rank = int(os.environ.get('RANK', -1))
    if rank != -1:
        return rank
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get the total number of processes."""
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    if world_size != -1:
        return world_size
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


logger = logging.getLogger("train")
if is_main_process():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
# ============================================================================
# Data Loading
# ============================================================================
def load_pre_split_data(train_file, dev_file, seed=42, max_samples=None):
    """Loads pre-split train and dev CSVs."""
    if is_main_process():
        print(f"Loading pre-split training data from: {train_file}")
        print(f"Loading pre-split development data from: {dev_file}")

    try:
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")

    # The `max_samples` logic is useful for quick tests, so we keep it here.
    if max_samples is not None and max_samples > 0:
        # We'll sample from both to keep the train/dev ratio roughly the same
        train_ratio = len(train_df) / (len(train_df) + len(dev_df))
        num_train_samples = int(max_samples * train_ratio)
        num_dev_samples = max_samples - num_train_samples
        
        train_df = train_df.sample(n=min(num_train_samples, len(train_df)), random_state=seed).reset_index(drop=True)
        dev_df = dev_df.sample(n=min(num_dev_samples, len(dev_df)), random_state=seed).reset_index(drop=True)
        
        if is_main_process():
            print(f"⚠️  QUICK TEST MODE: Using {len(train_df)} train and {len(dev_df)} dev samples.")

    if is_main_process():
        print(f"\nTrain samples: {len(train_df)}")
        print(f"Dev samples: {len(dev_df)}\n")

    return train_df, dev_df


# ============================================================================
# Model Setup
# ============================================================================
def setup_model_and_processor(model_name, lora_r=32, lora_alpha=16, lora_dropout=0.05):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_main_process():
        print(f"Loading Processor: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if is_main_process():
        print(f"Loading Model (Native): {model_name}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # flash_attention_2 sdpa
        device_map={'': local_rank} 
    )

    model.config.use_cache = False 
    
    # 1. Enable Gradient Checkpointing with correct kwargs for DeepSpeed
    if CONFIG['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 2. Enable Input Grads (Required for LoRA + Checkpointing)
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM", 
    )
    
    model = get_peft_model(model, peft_config)
    
    if is_main_process():
        model.print_trainable_parameters()

    return model, processor


# ============================================================================
# Custom Callbacks
# ============================================================================
class GenerationEvalCallback(TrainerCallback):
    """
    • Runs generation-based accuracy ONLY at the end of each epoch.
    • Implements early stopping based on that accuracy.
    """
    def __init__(
        self,
        processor,
        dev_dataset,
        every_steps: int, 
        n_samples: int,
        patience: int,
        delta: float = 0.0,
    ):
        self.processor    = processor
        self.dev_dataset  = dev_dataset
        self.n_samples    = n_samples
        self.patience     = patience
        self.delta        = delta
        
        self.best_metric  = -float("inf")
        self.bad_evals    = 0

    # THIS IS THE ONLY METHOD (on_step_end is DELETED)
    def on_epoch_end(self, args, state, control, **kwargs):
        stop_signal = torch.tensor([0.0], device=kwargs["model"].device)

        if is_main_process():
            model = kwargs["model"]
            model.eval()

            print(f"\n\n🔍 END OF EPOCH {state.epoch}: Running Generation Evaluation...")
            
            correct = total = 0
            with torch.no_grad():
                # Run inference on dev set
                limit = len(self.dev_dataset) if self.n_samples is None else min(self.n_samples, len(self.dev_dataset))
                
                for idx in tqdm(range(limit), desc="Evaluating"):
                    sample = self.dev_dataset[idx]

                    true_tokens = sample["labels"][sample["labels"] != -100]
                    true_text   = self.processor.tokenizer.decode(true_tokens, skip_special_tokens=True).strip()
                    
                    true_decision = None
                    try:
                        true_json = json.loads(true_text)
                        true_decision = true_json.get("decision", "").upper()
                    except:
                        true_decision = next((lab for lab in ["SUPPORT", "CONTRADICT", "NEUTRAL"] if lab in true_text.upper()), None)

                    non_masked_indices = (sample["labels"] != -100).nonzero(as_tuple=True)[0]
                    prompt_len = non_masked_indices[0].item() if len(non_masked_indices) > 0 else len(sample["labels"])

                    inputs = {
                        "input_ids"      : sample["input_ids"][:prompt_len].unsqueeze(0).to(model.device),
                        "attention_mask": sample["attention_mask"][:prompt_len].unsqueeze(0).to(model.device),
                        "pixel_values"  : sample["pixel_values"].unsqueeze(0).to(model.device),
                        "image_grid_thw": sample["image_grid_thw"].unsqueeze(0).to(model.device),
                    }
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )

                    gen_text = self.processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

                    pred_decision = None
                    cleaned_text = gen_text.replace("```json", "").replace("```", "").strip()
                    try:
                        pred_json = json.loads(cleaned_text)
                        if "decision" in pred_json:
                            pred_decision = pred_json["decision"].upper()
                    except:
                        for label in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
                            if label in cleaned_text.upper():
                                pred_decision = label
                                break
                    
                    if pred_decision and true_decision:
                        total += 1
                        if pred_decision == true_decision:
                            correct += 1

            accuracy = correct / total if total else 0.0
            print(f"🟢 Epoch {state.epoch} Accuracy: {accuracy:.4f}")

            if wandb.run is not None:
                wandb.log({"eval_accuracy": accuracy, "epoch": state.epoch})

            if accuracy > self.best_metric + self.delta:
                self.best_metric = accuracy
                self.bad_evals = 0
                print(f"📈 New Best Model! (Patience reset)")
            else:
                self.bad_evals += 1
                print(f"📉 No improvement. Patience: {self.bad_evals}/{self.patience}")

            if self.bad_evals >= self.patience:
                print(f"\n🔴 Early stopping triggered on Rank 0.\n")
                stop_signal[0] = 1.0

            model.train()

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stop_signal, src=0)

        if stop_signal[0] == 1.0:
            control.should_training_stop = True
# ============================================================================
# Training - SIMPLIFIED WITHOUT LOGITS PREPROCESSING
# ============================================================================

def train_model(model, processor, train_dataset, dev_dataset, config):
    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():
        print(f"\n{'='*70}")
        print(f"🚀 Starting training on {world_size} GPU(s)")
        print(f"{'='*70}\n")

    if config.get("use_wandb", False) and int(os.environ.get('RANK', 0)) == 0:
        wandb.init(
            project=config["wandb_project"],
            name=config.get("wandb_run_name"),
            config=config,
        )

    # ===== ADD THIS DIAGNOSTIC HERE =====
    if is_main_process():
        print("\n" + "="*70)
        print("🔍 LABEL TOKENIZATION DIAGNOSTIC")
        print("="*70)
        
        for label in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
            tokens = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
            decoded = [processor.tokenizer.decode([t]) for t in tokens]
            
            print(f"\n'{label}':")
            print(f"  Token IDs: {tokens} ({len(tokens)} tokens)")
            print(f"  Decoded parts: {decoded}")
        
        # Also check what's in an actual training sample
        print("\n\n📋 Actual training sample check:")
        sample = train_dataset[0]
        labels = sample["labels"]
        valid_tokens = labels[labels != -100]
        print(f"  Valid token IDs: {valid_tokens.tolist()}")
        print(f"  Decoded text: '{processor.tokenizer.decode(valid_tokens)}'")
        
        print("="*70 + "\n")
    # ===== END DIAGNOSTIC =====

    # ==== GENERATION-BASED EVALUATION (FIXED) ====
    def evaluate_with_generation(eval_dataset, num_samples):
        """Evaluate by generating text."""
        model.eval()
        correct = 0
        total = 0
        
        print(f"Starting evaluation on {min(num_samples, len(eval_dataset))} samples...")
        
        with torch.no_grad():
            for idx in range(min(num_samples, len(eval_dataset))):
                sample = eval_dataset[idx]
                
                true_labels = sample["labels"]
                true_tokens = true_labels[true_labels != -100]
                true_text = processor.tokenizer.decode(true_tokens, skip_special_tokens=True).strip().upper()
                
                # --- THE FIX IS HERE ---
                # Find the exact start index of the label (where -100 stops)
                non_masked_indices = (true_labels != -100).nonzero(as_tuple=True)[0]
                if len(non_masked_indices) == 0:
                    # Fallback if something is wrong with labels
                    user_length = len(sample["input_ids"])
                else:
                    user_length = non_masked_indices[0].item()
                # -----------------------
                
                inputs = {
                    "input_ids": sample["input_ids"][:user_length].unsqueeze(0).to(model.device),
                    "attention_mask": sample["attention_mask"][:user_length].unsqueeze(0).to(model.device),
                    "pixel_values": sample["pixel_values"].unsqueeze(0).to(model.device),
                    "image_grid_thw": sample["image_grid_thw"].unsqueeze(0).to(model.device),
                }
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64, # Increased from 10 just in case
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                generated_text = processor.tokenizer.decode(outputs[0][user_length:], skip_special_tokens=True).strip().upper()
                
                # Flexible matching logic
                pred_label = None
                cleaned_text = generated_text.replace("```json", "").replace("```", "").strip()
                
                # Check JSON first
                try:
                    import json
                    js = json.loads(cleaned_text)
                    if "decision" in js:
                        pred_label = js["decision"].upper()
                except:
                    pass
                
                # Fallback to string search
                if not pred_label:
                    for label in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
                        if label in cleaned_text:
                            pred_label = label
                            break
                
                true_label = None
                for label in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
                    if label in true_text:
                        true_label = label
                        break
                
                if is_main_process() and idx < 3:
                    print(f"  Sample {idx}: pred='{generated_text}' → {pred_label}, true='{true_text}' → {true_label}")
                
                if pred_label and true_label and pred_label == true_label:
                    correct += 1
                if true_label:
                    total += 1
        
        model.train()
        final_accuracy = correct / total if total > 0 else 0.0
        
        if is_main_process():
            logger.info("✅ Final Accuracy: %.4f", final_accuracy)
            if config.get("use_wandb", False) and wandb.run is not None:
                wandb.log({"final_gen_acc": final_accuracy})
        
        return final_accuracy

    # ==== TRAINING ARGS (Loss monitoring only) ====
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=config['logging_steps'],  # Log loss every 10 steps
        save_strategy=config["save_strategy"],
        save_total_limit=2,
        fp16=False,
        bf16=True,
        report_to="wandb" if (config.get("use_wandb", False) and int(os.environ.get('RANK', 0)) == 0) else "none",
        seed=config["seed"],
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=False,
        gradient_checkpointing=config['gradient_checkpointing'],
        optim=config['optim'],
        ddp_find_unused_parameters=False,
        ddp_backend=config['ddp_backend'], # nccl
        max_grad_norm=1.0,
        # NO eval_strategy - defaults to 'no'
        prediction_loss_only=True,
        load_best_model_at_end=config['load_best_model_at_end'],
        greater_is_better=False,
        save_steps=config['eval_every_steps'],
        eval_strategy=config['evaluation_strategy'],
        metric_for_best_model=config['metric_for_best_model'],
        eval_steps = config["eval_every_steps"],
        dataloader_pin_memory=config['dataloader_pin_memory'],
        deepspeed="./ds_config.json",
        dataloader_persistent_workers=config['dataloader_persistent_workers'],
        dataloader_prefetch_factor=config['dataloader_prefetch_factor'],
    )

    data_collator = MultimodalDataCollator(processor)

    n_samples = config["eval_samples"] if config["eval_samples"] is not None else len(dev_dataset)
    gen_eval_cb = GenerationEvalCallback(
        processor     = processor,
        dev_dataset   = dev_dataset,
        every_steps   = config["eval_every_steps"],
        n_samples     = n_samples,
        patience      = config["early_stop_patience"],
        delta         = config["early_stop_delta"],
    )
    
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        data_collator   = data_collator,
        callbacks       = [gen_eval_cb],
        eval_dataset    = dev_dataset,
    )

    gen_eval_cb.trainer = trainer

    # Train
    if is_main_process():
        print("Starting training...")
    
    trainer.train()

    # Final evaluation with generation
    if is_main_process():
        print("\n" + "="*70)
        print("🔍 FINAL EVALUATION (Generation-based)")
        print("="*70)
        final_accuracy = evaluate_with_generation(dev_dataset, num_samples=len(dev_dataset))
    else:
        final_accuracy = 0.0

    # Save
    if is_main_process():
        final_model_path = os.path.join(config["output_dir"], "final_model")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        if is_main_process():
            print(f"\n✅ Model saved to {final_model_path}")
            print(f"✅ Final Accuracy: {final_accuracy:.4f}\n")

    return trainer, {"eval_label_accuracy": final_accuracy}


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    if is_main_process():
        print("\n" + "="*70)
        print("🔧 Qwen3-VL Judgment Task Fine-tuning")
        print("="*70 + "\n")

    train_df, dev_df = load_pre_split_data(
        CONFIG["train_data_file"],
        CONFIG["dev_data_file"],
        CONFIG["seed"],
        CONFIG.get("max_samples")
    )

    model, processor = setup_model_and_processor(
        CONFIG["model_name"],
        CONFIG["lora_r"],
        CONFIG["lora_alpha"],
        CONFIG["lora_dropout"]
    )

    train_dataset = MultimodalJudgmentDataset(train_df, processor, max_cache_size=10000)
    dev_dataset = MultimodalJudgmentDataset(dev_df, processor, max_cache_size=10000)

    if is_main_process():
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Dev dataset: {len(dev_dataset)} samples\n")

    trainer, final_metrics = train_model(
        model,
        processor,
        train_dataset,
        dev_dataset,
        CONFIG
    )

    if is_main_process():
        print("\n" + "="*70)
        print("🎉 ALL DONE!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

#nohup torchrun --nproc_per_node=4 train_judgment_only_unsloth_v3.py > training.log 2>&1 & tail -f training.log
import os
# Set GPUs before any other imports
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_CACHE_DIR"] = "/gpfs/projects/MaffeiGroup/triton_cache"

import pandas as pd
# Silence verbose compilation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

import torch
# Reduce compiler verbosity
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)

from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    AutoModelForImageTextToText
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import json
from PIL import Image
from typing import Dict, List, Any
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "data_file": "./data_collection/all_data_with_judge_without_fig_ref_v3.csv",
    "output_dir": "./qwen3_vl_judgment_lora",
    "max_length": 512,
    "train_split": 0.9,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "early_stopping_patience": 3,
    "early_stopping_metric": "label_accuracy",
    "show_confusion_matrix": False,
    "seed": 42,
    "use_wandb": True,
    "wandb_project": "qwen3-vl-ft-label-only",
    "wandb_run_name": None
}

# Label mapping
LABEL_MAP = {"SUPPORT": 0, "CONTRADICT": 1, "NEUTRAL": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ============================================================================
# Helper Functions
# ============================================================================

def get_label_token_ids(processor):
    """Get the first token ID for each label."""
    label_token_ids = {}
    for label in LABEL_MAP.keys():
        tokens = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        label_token_ids[label] = tokens[0]
    return label_token_ids


def is_main_process():
    """Check if this is the main process."""
    return int(os.environ.get('RANK', 0)) == 0


def get_rank():
    """Get current process rank."""
    return int(os.environ.get('RANK', 0))


def get_world_size():
    """Get total number of processes."""
    return int(os.environ.get('WORLD_SIZE', 1))


# ============================================================================
# Custom Callbacks (DDP-Safe)
# ============================================================================

class LoggingCallback(TrainerCallback):
    """Enhanced logging - only on main process."""
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is None or not is_main_process():
            return
            
        print("\n" + "="*70)
        print(f"📊 EVALUATION at Epoch {int(state.epoch)}")
        print("="*70)
        print(f"  Loss:            {metrics.get('eval_loss', 0.0):.4f}")
        print(f"  Token Accuracy:  {metrics.get('eval_token_accuracy', 0.0):.4f}  ← Logits-based (all positions)")
        print(f"  Label Accuracy:  {metrics.get('eval_label_accuracy', 0.0):.4f}  ← Classification (judgment position)")
        print("="*70 + "\n")


# ============================================================================
# Dataset Class
# ============================================================================

class MultimodalJudgmentDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        panels = str(row.get('panels', ''))
        image_path = os.path.join("./data_collection", row['local_image_path'].lstrip('./'))
        claim = str(row.get('new_claim', ''))
        caption = str(row.get('caption', ''))
        judgment = str(row.get('class', 'NEUTRAL')).strip().upper()

        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        except FileNotFoundError:
            if is_main_process():
                print(f"Warning: Image not found at {image_path}. Using blank image.")
            image = Image.new('RGB', (224, 224), color='white')
        except Exception as e:
            if is_main_process():
                print(f"Error loading image {image_path}: {e}. Using blank image.")
            image = Image.new('RGB', (224, 224), color='white')

        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._create_prompt(panels, claim, caption)}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": judgment}]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        # Create labels
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find judgment tokens and mask everything before them
        judgment_tokens = self.processor.tokenizer(judgment, add_special_tokens=False)["input_ids"]
        
        judgment_start_idx = None
        for i in range(len(input_ids) - len(judgment_tokens) + 1):
            if input_ids[i:i+len(judgment_tokens)].tolist() == judgment_tokens:
                judgment_start_idx = i
                break

        if judgment_start_idx is not None:
            labels[:judgment_start_idx] = -100
        else:
            if is_main_process():
                print(f"Warning: Could not find judgment tokens in sample {idx}")
            labels[:] = -100

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values.squeeze(0) if pixel_values is not None else torch.zeros(3, 224, 224),
            "image_grid_thw": image_grid_thw.squeeze(0) if image_grid_thw is not None else torch.zeros(1, 3),
            "labels": labels,
        }

    def _create_prompt(self, panels, claim, caption):
        caption_text = caption if caption else "No caption provided"
        return f"""Given the figure above and the following information, determine if the claim is SUPPORTED, CONTRADICTED, or NEUTRAL.

Figure Panels: {panels}
Claim: {claim}
Caption: {caption_text}

Respond with only one word: SUPPORT, CONTRADICTED, or NEUTRAL.

Judgment:"""


# ============================================================================
# Data Collator
# ============================================================================

class MultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) == 0:
            raise ValueError("Empty batch received!")
        
        max_len = max(f["input_ids"].size(0) for f in features)

        batched_input_ids = []
        batched_attention_mask = []
        batched_labels = []

        for feature in features:
            seq_len = feature["input_ids"].size(0)
            pad_len = max_len - seq_len

            padded_input_ids = torch.cat([
                feature["input_ids"],
                torch.full((pad_len,), self.processor.tokenizer.pad_token_id, dtype=torch.long)
            ])

            padded_attention_mask = torch.cat([
                feature["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ])

            padded_labels = torch.cat([
                feature["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

            batched_input_ids.append(padded_input_ids)
            batched_attention_mask.append(padded_attention_mask)
            batched_labels.append(padded_labels)

        batch = {
            "input_ids": torch.stack(batched_input_ids),
            "attention_mask": torch.stack(batched_attention_mask),
            "labels": torch.stack(batched_labels),
        }

        # Handle images
        pixel_values_list = [f.get("pixel_values") for f in features if f.get("pixel_values") is not None]
        image_grid_thw_list = [f.get("image_grid_thw") for f in features if f.get("image_grid_thw") is not None]

        if pixel_values_list:
            batch["pixel_values"] = torch.stack(pixel_values_list)
        if image_grid_thw_list:
            batch["image_grid_thw"] = torch.stack(image_grid_thw_list)

        return batch


# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_data(data_file, train_split=0.9, seed=42):
    """Load CSV and split into train/dev sets - MUST be identical across all ranks."""
    if is_main_process():
        print(f"Loading data from {data_file}...")
    
    df = pd.read_csv(data_file)
    
    required_cols = ['new_claim', 'caption', 'local_image_path', 'class', 'panels']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df['caption'] = df['caption'].fillna('')
    df['class'] = df['class'].str.strip().str.upper()
    
    valid_judgments = set(LABEL_MAP.keys())
    df = df[df['class'].isin(valid_judgments)].copy()
    
    if is_main_process():
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:\n{df['class'].value_counts()}")
    
    # Split with fixed seed - CRITICAL: All ranks must get same split
    if len(df['class'].unique()) > 1:
        train_df, dev_df = train_test_split(
            df, 
            train_size=train_split, 
            random_state=seed, 
            stratify=df['class']
        )
    else:
        train_df, dev_df = train_test_split(
            df, 
            train_size=train_split, 
            random_state=seed
        )
        if is_main_process():
            print("Warning: Only one class present. Cannot stratify split.")
    
    if is_main_process():
        print(f"\nTrain samples: {len(train_df)}")
        print(f"Dev samples: {len(dev_df)}")
        
        # Save splits
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        train_df.to_csv(os.path.join(CONFIG["output_dir"], "train_split.csv"), index=False)
        dev_df.to_csv(os.path.join(CONFIG["output_dir"], "dev_split.csv"), index=False)
        print(f"✅ Saved train/dev splits to {CONFIG['output_dir']}")
    
    return train_df, dev_df


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_processor(model_name, lora_r=32, lora_alpha=16, lora_dropout=0.05):
    """Load model with LoRA - DDP compatible."""
    if is_main_process():
        print(f"Loading model: {model_name}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model - CRITICAL: device_map=None for DDP
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,  # Let Trainer handle device placement
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    if is_main_process():
        model.print_trainable_parameters()

    return model, processor


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(eval_pred, processor, label_token_ids):
    """Compute token accuracy and label accuracy."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Token-level accuracy (all positions)
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    token_level_correct = (predictions == labels) & mask
    token_accuracy = token_level_correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    
    # Label classification accuracy (judgment position only)
    label_correct = 0
    label_total = 0
    
    for sample_logits, sample_labels in zip(logits, labels):
        valid_positions = np.where(sample_labels != -100)[0]
        
        if len(valid_positions) == 0:
            continue
        
        first_valid_pos = valid_positions[0]
        position_logits = sample_logits[first_valid_pos]
        
        # Get logits for label tokens
        label_logits = {
            label: position_logits[token_id] 
            for label, token_id in label_token_ids.items()
        }
        
        predicted_label = max(label_logits, key=label_logits.get)
        
        # Get true label
        true_token_id = sample_labels[first_valid_pos]
        true_label = None
        for label, token_id in label_token_ids.items():
            if token_id == true_token_id:
                true_label = label
                break
        
        if true_label is not None:
            if predicted_label == true_label:
                label_correct += 1
            label_total += 1
    
    label_accuracy = label_correct / label_total if label_total > 0 else 0.0
    
    return {
        "token_accuracy": token_accuracy,
        "label_accuracy": label_accuracy,
    }


# ============================================================================
# Training
# ============================================================================

def train_model(model, processor, train_dataset, dev_dataset, config):
    """Train model with DDP support."""
    
    rank = get_rank()
    world_size = get_world_size()
    
    if is_main_process():
        print(f"\n{'='*70}")
        print(f"🚀 Starting training on {world_size} GPU(s)")
        print(f"{'='*70}\n")
    
    # Initialize WandB on main process only
    if config.get("use_wandb", False) and is_main_process():
        import wandb
        wandb.init(
            project=config["wandb_project"],
            name=config.get("wandb_run_name"),
            config=config,
        )
    
    # Get label token IDs
    label_token_ids = get_label_token_ids(processor)
    
    if is_main_process():
        print("\n" + "="*70)
        print("Label Token Mapping:")
        for label, token_id in label_token_ids.items():
            token_text = processor.tokenizer.decode([token_id])
            print(f"  {label}: token_id={token_id} ('{token_text}')")
        print("="*70 + "\n")
    
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, processor, label_token_ids)
    
    data_collator = MultimodalDataCollator(processor)
    
    # Training arguments - optimized for DDP
    # training_args = TrainingArguments(
    #     output_dir=config["output_dir"],
    #     num_train_epochs=config["num_epochs"],
    #     per_device_train_batch_size=config["batch_size"],
    #     per_device_eval_batch_size=config["batch_size"],
    #     gradient_accumulation_steps=config["gradient_accumulation_steps"],
    #     learning_rate=config["learning_rate"],
    #     lr_scheduler_type="cosine",
    #     warmup_ratio=0.1,
    #     logging_steps=10,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     save_total_limit=2,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="label_accuracy",
    #     greater_is_better=True,
    #     fp16=True,
    #     report_to="wandb" if (config.get("use_wandb", False) and is_main_process()) else "none",
    #     seed=config["seed"],
        
    #     # CRITICAL: Data loading settings for DDP
    #     dataloader_num_workers=4,  # Safe for PIL Image + DDP
    #     dataloader_prefetch_factor=2,
    #     remove_unused_columns=False,
    #     gradient_checkpointing=True,
    #     optim="adamw_torch_fused",
    #     torch_compile=True,
    #     eval_accumulation_steps=4,
    #     include_inputs_for_metrics=False,
    #     prediction_loss_only=False,
        
    #     # DDP settings
    #     ddp_find_unused_parameters=False,  # CRITICAL for PEFT
    #     ddp_backend="gloo",
    # )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
	eval_steps=100,
        save_strategy='steps',
	save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="label_accuracy",
        greater_is_better=True,
        fp16=True,
        report_to="wandb" if (config.get("use_wandb", False) and is_main_process()) else "none",
        seed=config["seed"],
        
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        
        ddp_find_unused_parameters=False,
        ddp_backend="gloo",
        
        eval_accumulation_steps=4,
        include_inputs_for_metrics=False,
    )

    
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
    ]
    
    if is_main_process():
        effective_batch = config["batch_size"] * config["gradient_accumulation_steps"] * world_size
        print("\n" + "="*70)
        print("Training Configuration:")
        print("="*70)
        print(f"  Number of GPUs: {world_size}")
        print(f"  Per-device batch size: {config['batch_size']}")
        print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Epochs: {config['num_epochs']}")
        print("="*70 + "\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train
    trainer.train()
    
    # Save on main process only
    if is_main_process():
        final_model_path = os.path.join(config["output_dir"], "final_model")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        
        with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Model saved to {final_model_path}")
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    
    if is_main_process():
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"  Final Token Accuracy: {final_metrics.get('eval_token_accuracy', 0.0):.4f}")
        print(f"  Final Label Accuracy: {final_metrics.get('eval_label_accuracy', 0.0):.4f}")
        print("="*70 + "\n")
    
    return trainer, final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    # Set seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    if is_main_process():
        print("\n" + "="*70)
        print("🔧 Qwen3-VL Judgment Task Fine-tuning")
        print("="*70 + "\n")
    
    # Load and split data
    train_df, dev_df = load_and_split_data(
        CONFIG["data_file"],
        CONFIG["train_split"],
        CONFIG["seed"]
    )
    
    # Setup model and processor
    model, processor = setup_model_and_processor(
        CONFIG["model_name"],
        CONFIG["lora_r"],
        CONFIG["lora_alpha"],
        CONFIG["lora_dropout"]
    )
    
    # Create datasets
    train_dataset = MultimodalJudgmentDataset(train_df, processor)
    dev_dataset = MultimodalJudgmentDataset(dev_df, processor)
    
    if is_main_process():
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Dev dataset: {len(dev_dataset)} samples\n")
    
    # Train
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

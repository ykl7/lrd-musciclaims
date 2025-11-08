import os
# <<< FIX 1: Explicitly set the GPU to use. Change "0" if you want to use a different one.
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    AutoModelForVision2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import json
from PIL import Image
from typing import Dict, List
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
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "num_epochs": 10,
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

# NEW: Function to get label token IDs
def get_label_token_ids(processor):
    """
    Get the first token ID for each label.
    Since these are multi-token words, we'll use the first token as the primary indicator.
    """
    label_token_ids = {}
    for label in LABEL_MAP.keys():
        # Tokenize without special tokens
        tokens = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        # Use the first token as the representative
        label_token_ids[label] = tokens[0]
    
    return label_token_ids


# ============================================================================
# Custom Callbacks
# ============================================================================

class LoggingCallback(TrainerCallback):
    """Enhanced logging to show both metrics clearly."""
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Print metrics after each evaluation."""
        if metrics is None:
            return
        
        # Only print on main process
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
            
        print("\n" + "="*70)
        print(f"📊 EVALUATION at Epoch {int(state.epoch)}")
        print("="*70)
        print(f"  Loss:            {metrics.get('eval_loss', 0.0):.4f}")
        print(f"  Token Accuracy:  {metrics.get('eval_token_accuracy', 0.0):.4f}  ← Logits-based (all positions)")
        print(f"  Label Accuracy:  {metrics.get('eval_label_accuracy', 0.0):.4f}  ← Classification (judgment position)")
        print("="*70 + "\n")


class FinalGenerationEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, use_wandb=False):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.use_wandb = use_wandb
    
    def on_train_end(self, args, state, control, model, **kwargs):
        # Only run on main process
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        print("\n" + "="*70)
        print("🎯 FINAL GENERATION-BASED EVALUATION")
        print("="*70)
        
        gen_accuracy = evaluate_generation_accuracy(
            model,
            self.eval_dataset,
            self.processor,
            show_confusion=True
        )
        
        # Log to WandB
        if self.use_wandb:
            import wandb
            wandb.log({
                "final_generation_accuracy": gen_accuracy,
                "confusion_matrix": wandb.Image(f"{CONFIG['output_dir']}/confusion_matrix_generation.png")
            })
        
        print(f"✅ Final Generation Accuracy: {gen_accuracy:.4f}\n")


class PeriodicGenerationEvalCallback(TrainerCallback):
    """Run generation eval every N epochs."""
    
    def __init__(self, eval_dataset, processor, every_n_epochs=2):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.every_n_epochs = every_n_epochs
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        # Only run on main process
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        if int(state.epoch) % self.every_n_epochs == 0 and int(state.epoch) > 0:
            print("\n🎯 Running generation-based evaluation...")
            evaluate_generation_accuracy(model, self.eval_dataset, self.processor, show_confusion=False)

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

        try:
            image = Image.open(image_path).convert('RGB')
            # Resize large images
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Using a blank image.")
            image = Image.new('RGB', (224, 224), color='white')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Using a blank image.")
            image = Image.new('RGB', (224, 224), color='white')

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

        # Get the full text with conversation template
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

        # Create labels - FIXED VERSION
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        labels = input_ids.clone()

        # Find where the assistant's response starts in the tokenized sequence
        # Tokenize just the judgment to find it in the sequence
        judgment_tokens = self.processor.tokenizer(judgment, add_special_tokens=False)["input_ids"]

        # Mask everything except the judgment tokens
        # We need to find where judgment_tokens appear in input_ids
        judgment_start_idx = None
        for i in range(len(input_ids) - len(judgment_tokens) + 1):
            if input_ids[i:i+len(judgment_tokens)].tolist() == judgment_tokens:
                judgment_start_idx = i
                break

        if judgment_start_idx is not None:
            # Mask everything before the judgment
            labels[:judgment_start_idx] = -100
        else:
            # If we can't find the judgment, mask everything (shouldn't happen)
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
# Custom Data Collator (CORRECTED)
# ============================================================================

from typing import List, Dict, Any

class MultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find max sequence length in this batch
        max_len = max(f["input_ids"].size(0) for f in features)

        # Manually pad each tensor to max_len
        batched_input_ids = []
        batched_attention_mask = []
        batched_labels = []

        for feature in features:
            seq_len = feature["input_ids"].size(0)
            pad_len = max_len - seq_len

            # Pad input_ids with tokenizer.pad_token_id
            padded_input_ids = torch.cat([
                feature["input_ids"],
                torch.full((pad_len,), self.processor.tokenizer.pad_token_id, dtype=torch.long)
            ])

            # Pad attention_mask with 0
            padded_attention_mask = torch.cat([
                feature["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ])

            # Pad labels with -100 (ignore index)
            padded_labels = torch.cat([
                feature["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

            batched_input_ids.append(padded_input_ids)
            batched_attention_mask.append(padded_attention_mask)
            batched_labels.append(padded_labels)

        # Stack into batch tensors
        batch = {
            "input_ids": torch.stack(batched_input_ids),
            "attention_mask": torch.stack(batched_attention_mask),
            "labels": torch.stack(batched_labels),
        }

        # Collect image features
        image_pixel_values = [f.get("pixel_values") for f in features if f.get("pixel_values") is not None]
        image_grid_thw = [f.get("image_grid_thw") for f in features if f.get("image_grid_thw") is not None]

        if image_pixel_values:
            batch["pixel_values"] = torch.cat(image_pixel_values, dim=0)
        if image_grid_thw:
            # Stack instead of cat to preserve batch dimension
            batch["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)

        return batch

def load_and_split_data(data_file, train_split=0.9, seed=42):
    """Load CSV and split into train/dev sets."""
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
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['class'].value_counts()}")
    # <<< FIX 5: Handle case where a class is missing for stratification
    if len(df['class'].unique()) > 1:
        train_df, dev_df = train_test_split(df, train_size=train_split, random_state=seed, stratify=df['class'])
    else:
        train_df, dev_df = train_test_split(df, train_size=train_split, random_state=seed)
        print("Warning: Only one class present. Cannot stratify split.")
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Dev samples: {len(dev_df)}")
    os.makedirs(os.path.dirname(CONFIG["output_dir"]) or ".", exist_ok=True)
    train_df.to_csv(os.path.join(CONFIG["output_dir"], "train_split.csv"), index=False)
    dev_df.to_csv(os.path.join(CONFIG["output_dir"], "dev_split.csv"), index=False)
    print(f"✅ Saved train/dev splits to {CONFIG['output_dir']}")
    return train_df, dev_df


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_processor(model_name, lora_r=32, lora_alpha=16, lora_dropout=0.05):
    """Load Qwen2-VL model with LoRA."""
    print(f"Loading model: {model_name}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    )
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
    model.print_trainable_parameters()

    return model, processor


# ============================================================================
# Metrics & Evaluation
# ============================================================================

def compute_metrics(eval_pred, processor, label_token_ids):
    """
    Compute TWO metrics:
    1. Token-level logits accuracy (across all non-masked positions)
    2. Label classification accuracy (at the judgment position only)
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # ========================================
    # METRIC 1: Token-level Logits Accuracy
    # ========================================
    # Get predictions for all positions
    predictions = np.argmax(logits, axis=-1)
    
    # Create mask for valid positions (labels != -100)
    mask = labels != -100
    
    # Compute accuracy only on valid positions
    token_level_correct = (predictions == labels) & mask
    token_accuracy = token_level_correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    
    # ========================================
    # METRIC 2: Label Classification Accuracy
    # ========================================
    label_correct = 0
    label_total = 0
    
    # For each sample in the batch
    for sample_logits, sample_labels in zip(logits, labels):
        # Find the first non-masked position (this is where judgment starts)
        valid_positions = np.where(sample_labels != -100)[0]
        
        if len(valid_positions) == 0:
            continue
        
        first_valid_pos = valid_positions[0]
        
        # Get logits at the judgment position
        position_logits = sample_logits[first_valid_pos]
        
        # Extract logits for our three label tokens
        label_logits = {
            label: position_logits[token_id] 
            for label, token_id in label_token_ids.items()
        }
        
        # Get predicted label (highest logit among the three)
        predicted_label = max(label_logits, key=label_logits.get)
        
        # Get true label (find which token ID matches)
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


def decode_model_output(output_text):
    """Extract judgment label from model output (for optional generation-based eval)."""
    output_text = output_text.strip().upper()
    
    for label in ["CONTRADICTED", "SUPPORT", "NEUTRAL"]:
        if label in output_text:
            return label
    
    return "NEUTRAL"


def evaluate_generation_accuracy(model, dataset, processor, show_confusion=False):
    """
    OPTIONAL: Full generation-based evaluation (slower, for final validation).
    This is separate from the fast logits-based metrics.
    """
    model.eval()
    
    predictions = []
    true_labels = []
    
    print(f"Evaluating generation on {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        row = dataset.data.iloc[idx]
        
        panels = str(row.get('panels', ''))
        image_path = os.path.join("./data_collection", row['local_image_path'].lstrip('./'))
        claim = str(row.get('new_claim', ''))
        caption = str(row.get('caption', ''))
        true_label = str(row.get('class', 'NEUTRAL')).strip().upper()
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        except:
            image = Image.new('RGB', (224, 224), color='white')
        
        # Create prompt
        caption_text = caption if caption else "No caption provided"
        prompt = f"""Given the figure above and the following information, determine if the claim is SUPPORTED, CONTRADICTED, or NEUTRAL.

Figure Panels: {panels}
Claim: {claim}
Caption: {caption_text}

Respond with only one word: SUPPORT, CONTRADICTED, or NEUTRAL.

Judgment:"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False
            )
        
        # Decode
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(text):].strip()
        
        predicted_label = decode_model_output(response)
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
    
    # Compute accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*70)
    print("GENERATION-BASED EVALUATION (Full Autoregressive)")
    print("="*70)
    print(classification_report(
        true_labels,
        predictions,
        target_names=list(LABEL_MAP.keys()),
        digits=4
    ))
    
    if show_confusion:
        cm = confusion_matrix(true_labels, predictions, labels=list(LABEL_MAP.keys()))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(LABEL_MAP.keys()),
                    yticklabels=list(LABEL_MAP.keys()))
        plt.title('Confusion Matrix (Generation-Based)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix_generation.png"))
        print(f"✅ Confusion matrix saved to {CONFIG['output_dir']}/confusion_matrix_generation.png")
    
    print("="*70 + "\n")
    
    return accuracy


# ============================================================================
# Training
# ============================================================================

def train_model(model, processor, train_dataset, dev_dataset, config):
    """Train model with dual accuracy tracking."""
    
    # NEW CODE (CORRECT):
    import torch.distributed as dist
    
    # Check if we're in distributed mode and get rank
    if dist.is_initialized():
        is_main_process = dist.get_rank() == 0
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"[Rank {rank}/{world_size}] Process initialized")
    else:
        is_main_process = True
        rank = 0
        world_size = 1
        print("[Single GPU] Process initialized")
    
    # Only initialize WandB on main process
    if config.get("use_wandb", False) and is_main_process:
        import wandb
        wandb.init(
            project=config["wandb_project"],
            name=config.get("wandb_run_name"),
            config=config,
            tags=["qwen3-vl", "lora", "judgment-task", "ddp"]
        )
    
    # Get label token IDs (all processes)
    label_token_ids = get_label_token_ids(processor)
    
    # Only print on main process
    if is_main_process:
        print("\n" + "="*70)
        print("Label Token Mapping:")
        for label, token_id in label_token_ids.items():
            token_text = processor.tokenizer.decode([token_id])
            print(f"  {label}: token_id={token_id} ('{token_text}')")
        print("="*70 + "\n")
    
    # Create compute_metrics function with processor bound
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, processor, label_token_ids)
    
    data_collator = MultimodalDataCollator(processor)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="label_accuracy",
        greater_is_better=True,
        bf16=True,
        report_to="wandb" if (config.get("use_wandb", False) and is_main_process) else "none",
        seed=config["seed"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",

        # DDP specific settings
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",              
        local_rank=-1,                   
    )

    
    # Callbacks
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping_patience"]
        ),
        FinalGenerationEvalCallback(dev_dataset, processor, use_wandb=config.get("use_wandb", False)),
        PeriodicGenerationEvalCallback(dev_dataset, processor, every_n_epochs=2),  # Every 2 epochs
    ]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,  # ← THIS IS THE KEY ADDITION
    )
    
    # Train
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Total epochs: {config['num_epochs']}")
    print(f"  Early stopping patience: {config['early_stopping_patience']}")
    print(f"  Metric for best model: label_accuracy")
    print(f"\n  Metrics tracked:")
    print(f"    • Token Accuracy (logits-based, all positions)")
    print(f"    • Label Accuracy (classification at judgment position)")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(config["output_dir"], "final_model"))
    processor.save_pretrained(os.path.join(config["output_dir"], "final_model"))
    
    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Get final metrics
    final_metrics = trainer.evaluate()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"  Final Token Accuracy: {final_metrics.get('eval_token_accuracy', 0.0):.4f}")
    print(f"  Final Label Accuracy: {final_metrics.get('eval_label_accuracy', 0.0):.4f}")
    print(f"  Model saved to: {config['output_dir']}/final_model")
    print("="*70 + "\n")
    
    return trainer, final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    # Set seed
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # Load data
    train_df, dev_df = load_and_split_data(
        CONFIG["data_file"],
        CONFIG["train_split"],
        CONFIG["seed"]
    )
    
    # Setup model
    model, processor = setup_model_and_processor(
        CONFIG["model_name"],
        CONFIG["lora_r"],
        CONFIG["lora_alpha"],
        CONFIG["lora_dropout"]
    )
    
    # Create datasets
    train_dataset = MultimodalJudgmentDataset(train_df, processor)
    dev_dataset = MultimodalJudgmentDataset(dev_df, processor)
    
    # Train
    trainer, final_metrics = train_model(  # ← Changed return value
        model,
        processor,
        train_dataset,
        dev_dataset,
        CONFIG
    )
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    main()


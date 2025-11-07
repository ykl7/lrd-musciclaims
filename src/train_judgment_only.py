"""
Fine-tune Qwen3-VL-8B-Instruct for multimodal claim judgment.
Uses images + text (panels, claim, caption) to predict SUPPORT/CONTRADICT/NEUTRAL.
"""
import os
# GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import json
from PIL import Image
from qwen_vl_utils import process_vision_info
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "data_file": "your_data.csv",
    "output_dir": "./qwen3_vl_judgment_lora",
    "max_length": 512,
    "train_split": 0.9,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,  # Effective batch = 16
    "learning_rate": 2e-4,
    "num_epochs": 10,
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "early_stopping_patience": 3,
    "early_stopping_metric": "generation_accuracy",  # or "eval_loss"
    "show_confusion_matrix": False,  # Set to True to see confusion matrix
    "seed": 42,
}

# Label mapping
LABEL_MAP = {"SUPPORT": 0, "CONTRADICT": 1, "NEUTRAL": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


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
        
        # Get data
        panels = str(row.get('panels', ''))
        image_path = str(row.get('local_image_path', ''))
        claim = str(row.get('new_claim', ''))
        caption = str(row.get('caption', ''))
        judgment = str(row.get('class', 'NEUTRAL')).strip().upper()
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # Create conversation format for Qwen2-VL
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
        
        # Process with Qwen2-VL processor
        text = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        image_inputs, video_inputs = process_vision_info(conversation)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels (only train on judgment token)
        labels = inputs["input_ids"].clone()
        
        # Find assistant response start
        assistant_start = text.find(judgment)
        if assistant_start > 0:
            prompt_tokens = self.processor.tokenizer(
                text[:assistant_start], 
                add_special_tokens=False
            )["input_ids"]
            labels[0, :len(prompt_tokens)] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values").squeeze(0) if inputs.get("pixel_values") is not None else None,
            "image_grid_thw": inputs.get("image_grid_thw").squeeze(0) if inputs.get("image_grid_thw") is not None else None,
            "labels": labels.squeeze(0),
            "true_label": LABEL_MAP[judgment],
            "image_path": image_path,
            "claim": claim
        }
    
    def _create_prompt(self, panels, claim, caption):
        """Create text prompt."""
        caption_text = caption if caption else "No caption provided"
        return f"""Given the figure above and the following information, determine if the claim is SUPPORTED, CONTRADICTED, or NEUTRAL.

Figure Panels: {panels}
Claim: {claim}
Caption: {caption_text}

Respond with only one word: SUPPORT, CONTRADICTED, or NEUTRAL.

Judgment:"""


# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_data(data_file, train_split=0.9, seed=42):
    """Load CSV and split into train/dev sets."""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Map columns: new_claim, caption, local_image_path, class, panels
    required_cols = ['new_claim', 'caption', 'local_image_path', 'class', 'panels']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Handle missing captions
    df['caption'] = df['caption'].fillna('')
    
    # Clean judgment column
    df['class'] = df['class'].str.strip().str.upper()
    
    # Filter valid judgments
    valid_judgments = set(LABEL_MAP.keys())
    df = df[df['class'].isin(valid_judgments)].copy()
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['class'].value_counts()}")
    
    # Split
    train_df, dev_df = train_test_split(
        df,
        train_size=train_split,
        random_state=seed,
        stratify=df['class']
    )
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Dev samples: {len(dev_df)}")
    
    # Save splits
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
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
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

def decode_model_output(output_text):
    """Extract judgment label from model output."""
    output_text = output_text.strip().upper()
    
    for label in ["CONTRADICT", "SUPPORT", "NEUTRAL"]:
        if label in output_text:
            return label
    
    return "NEUTRAL"


def compute_logits_accuracy(predictions, label_ids, processor):
    """Compute accuracy from logits (first token prediction)."""
    # Get label tokens
    label_tokens = {
        label: processor.tokenizer(label, add_special_tokens=False)["input_ids"][0]
        for label in LABEL_MAP.keys()
    }
    
    correct = 0
    total = 0
    
    logits = predictions.predictions
    labels = predictions.label_ids
    
    for batch_logits, batch_labels in zip(logits, labels):
        # Find first non-masked position
        valid_positions = np.where(batch_labels != -100)[0]
        if len(valid_positions) == 0:
            continue
        
        first_valid = valid_positions[0]
        predicted_token = np.argmax(batch_logits[first_valid])
        true_label_text = None
        
        # Find true label
        for label, token_id in label_tokens.items():
            if batch_labels[first_valid] == token_id:
                true_label_text = label
                break
        
        if true_label_text and predicted_token == label_tokens[true_label_text]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_generation_accuracy(model, dataset, processor, show_confusion=False):
    """Evaluate with actual text generation."""
    model.eval()
    
    predictions = []
    true_labels = []
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        row = dataset.data.iloc[idx]
        
        panels = str(row.get('panels', ''))
        image_path = './data_collection' + '/' + str(row.get('local_image_path', '')).split('./')[-1]
        claim = str(row.get('new_claim', ''))
        caption = str(row.get('caption', ''))
        true_label = str(row.get('class', 'NEUTRAL')).strip().upper()
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
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
        
        image_inputs, video_inputs = process_vision_info(conversation)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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
    print("GENERATION-BASED EVALUATION")
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
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))
        print(f"✅ Confusion matrix saved to {CONFIG['output_dir']}/confusion_matrix.png")
    
    print("="*70 + "\n")
    
    return accuracy


# ============================================================================
# Custom Callbacks
# ============================================================================

class DualAccuracyCallback(TrainerCallback):
    """Compute both logits and generation accuracy."""
    
    def __init__(self, eval_dataset, processor, show_confusion=False):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.show_confusion = show_confusion
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Compute generation accuracy after evaluation."""
        print("\n" + "="*70)
        print(f"EPOCH {int(state.epoch)} EVALUATION")
        print("="*70)
        
        # Generation accuracy
        gen_accuracy = evaluate_generation_accuracy(
            model, 
            self.eval_dataset, 
            self.processor,
            self.show_confusion
        )
        
        # Store in metrics
        metrics["generation_accuracy"] = gen_accuracy
        
        print(f"📊 Generation Accuracy: {gen_accuracy:.4f}")
        print(f"📉 Validation Loss: {metrics.get('eval_loss', 0.0):.4f}")
        print("="*70 + "\n")


# ============================================================================
# Training
# ============================================================================

def train_model(model, processor, train_dataset, dev_dataset, config):
    """Train model with dual accuracy tracking."""
    
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=config["early_stopping_metric"],
        greater_is_better=(config["early_stopping_metric"] == "generation_accuracy"),
        bf16=True,
        report_to="none",
        seed=config["seed"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Callbacks
    callbacks = [
        DualAccuracyCallback(
            dev_dataset, 
            processor, 
            config["show_confusion_matrix"]
        ),
        EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping_patience"]
        )
    ]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print(f"Metric for early stopping: {config['early_stopping_metric']}")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    final_accuracy = evaluate_generation_accuracy(
        model, 
        dev_dataset, 
        processor,
        show_confusion=True  # Always show final confusion matrix
    )
    
    # Save final model
    trainer.save_model(os.path.join(config["output_dir"], "final_model"))
    processor.save_pretrained(os.path.join(config["output_dir"], "final_model"))
    
    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Model saved to {config['output_dir']}/final_model")
    print(f"✅ Final Generation Accuracy: {final_accuracy:.4f}")
    
    return trainer, final_accuracy


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
    trainer, accuracy = train_model(
        model, 
        processor, 
        train_dataset, 
        dev_dataset, 
        CONFIG
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Dev Accuracy: {accuracy:.4f}")
    print(f"Model: {CONFIG['output_dir']}/final_model")
    print("="*70)


if __name__ == "__main__":
    main()

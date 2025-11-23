from torch.utils.data import Dataset
from PIL import Image
import torch
import os

# ============================================================================
# Dataset Class
# ============================================================================

class MultimodalJudgmentDataset(Dataset):
    def __init__(self, dataframe, processor, max_cache_size=0): 
        # Removed cache logic for safety with multi-processing
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        panels = str(row.get('panels', ''))
        image_path = os.path.join("./../data_collection", row['local_image_path'].lstrip('./'))
        claim = str(row.get('new_claim', ''))
        caption = str(row.get('caption', ''))
        judgment = str(row.get('class', 'NEUTRAL')).strip().upper()
    
        try:
            image = Image.open(image_path).convert('RGB')
            max_size = 196
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (196, 196), (0, 0, 0))
        
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
    
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
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
    
        # Extract and process image tensors
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        
        # Process pixel_values
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)  # Remove batch dim
        else:
            # Fallback: create dummy with correct hidden size
            pixel_values = torch.zeros(1, 1536)
        
        # Process image_grid_thw (CRITICAL)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)  # Should become (3,)
            # Sanity check: ensure no zeros
            if (image_grid_thw == 0).any():
                print(f"⚠️ Warning: image_grid_thw contains zeros at idx {idx}: {image_grid_thw}")
                # Set to minimum valid value [1, 1, 1] to prevent crash
                image_grid_thw = torch.tensor([1, 1, 1], dtype=image_grid_thw.dtype)
        else:
            # Fallback: use minimum valid grid [1, 1, 1] instead of [0, 0, 0]
            image_grid_thw = torch.tensor([1, 1, 1], dtype=torch.long)
            print(f"⚠️ Warning: image_grid_thw was None at idx {idx}, using fallback [1,1,1]")
    
        # Diagnostic for first sample
        if idx == 0:
            print("\n" + "="*70)
            print("🔍 DATASET SHAPES (First Sample - AFTER PROCESSING)")
            print("="*70)
            print(f"pixel_values shape: {pixel_values.shape}")
            print(f"image_grid_thw shape: {image_grid_thw.shape}")
            print(f"image_grid_thw values: {image_grid_thw}")
            print(f"input_ids shape: {input_ids.shape}")
            print("="*70 + "\n")
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _create_prompt(self, panels, claim, caption):
        caption_text = caption if caption else "No caption provided"
        return f"""You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning. 
Your job is to analyze a given image(s) and its provided caption(s) to decide whether it SUPPORT or CONTRADICT or NEUTRAL the provided claim.\n\n

CLAIM: {claim}\n\n
IMAGE CAPTION(S): {caption_text}\n\n
PANEL REFERENCES: {panels}\n\n

Guidelines:\n
1. Evaluate the claim's plausibility based on visual elements within the image(s).\n
2. Consider the relevance, meaning, and implications of both the depicted content and the caption(s).\n
3. Analyze the broader context and scope of the image(s) and caption(s) in relation to the claim.\n
4. Use the provided panel references to precisely identify which specific panels (e.g., Panel A, Panel B, Panel C, etc.) are necessary to evaluate the claim.\n

After completing your analysis, output exactly one JSON object with exactly one key: \"decision\".\n
- For \"decision\", output exactly one word — either \"SUPPORT\" or \"CONTRADICT\" or \"NEUTRAL\" (uppercase, no extra text).\n
Do NOT add markdown formatting, code fences, or any additional text. The output must start with an opening curly brace {{ and end with a closing curly brace }}.\n\n

Example output format:\n
{{\"decision\": \"SUPPORT\"}}\n\n

Now, please evaluate the image(s) and caption(s) with respect to the claim provided above, using the provided panel references.
"""
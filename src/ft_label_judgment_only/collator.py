import torch
from typing import Dict, List, Any


class MultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) == 0:
            raise ValueError("Empty batch received")

        # 1. Text Handling (unchanged)
        batch_max_len = max(f["input_ids"].size(0) for f in features)
        batched_input_ids = []
        batched_attention_mask = []
        batched_labels = []

        for feature in features:
            seq_len = feature["input_ids"].size(0)
            pad_len = batch_max_len - seq_len
            
            batched_input_ids.append(torch.cat([
                feature["input_ids"],
                torch.full((pad_len,), self.processor.tokenizer.pad_token_id, dtype=torch.long)
            ]))
            batched_attention_mask.append(torch.cat([
                feature["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
            batched_labels.append(torch.cat([
                feature["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))

        batch = {
            "input_ids": torch.stack(batched_input_ids),
            "attention_mask": torch.stack(batched_attention_mask),
            "labels": torch.stack(batched_labels),
        }

        # 2. Image Handling - THE CRITICAL FIX
        pixel_values_list = []
        image_grid_thw_list = []
        
        for f in features:
            if f.get("pixel_values") is not None and f.get("image_grid_thw") is not None:
                pv = f["pixel_values"]
                gt = f["image_grid_thw"]
                
                # Validate shapes
                if pv.dim() != 2:
                    print(f"⚠️ Unexpected pixel_values dim: {pv.dim()}, expected 2")
                    continue
                if gt.shape != torch.Size([3]):
                    print(f"⚠️ Unexpected grid_thw shape: {gt.shape}, expected [3]")
                    continue
                
                # Verify the grid matches the actual number of patches
                expected_patches = int(gt[0] * gt[1] * gt[2])
                if pv.shape[0] != expected_patches:
                    print(f"⚠️ Mismatch: pixel_values has {pv.shape[0]} patches but grid expects {expected_patches}")
                    print(f"   Grid values: {gt.tolist()}")
                
                pixel_values_list.append(pv.to(dtype=torch.bfloat16))
                image_grid_thw_list.append(gt.to(dtype=torch.long))

        if pixel_values_list and image_grid_thw_list:
            # CRITICAL: Concatenate patches sequentially, stack grids
            # The model will use image_grid_thw to determine boundaries in the concatenated pixel_values
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0).contiguous()
            batch["image_grid_thw"] = torch.stack(image_grid_thw_list, dim=0).contiguous()
            
            # Sanity check
            total_expected = sum(int(gt[0] * gt[1] * gt[2]) for gt in image_grid_thw_list)
            if batch["pixel_values"].shape[0] != total_expected:
                print(f"⚠️ CRITICAL: Total patches {batch['pixel_values'].shape[0]} != expected {total_expected}")
        else:
            print("⚠️ WARNING: No valid image data in batch!")

        return batch
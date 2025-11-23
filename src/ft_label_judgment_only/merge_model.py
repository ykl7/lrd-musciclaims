from unsloth import FastLanguageModel
import torch
import os

# 1. Configuration
# Point this to where your training script saved the final model
lora_model_path = "./qwen3_vl_judgment_lora/final_model" 
output_merged_path = "./qwen3_vl_judgment_merged"

print(f"Loading LoRA from: {lora_model_path}")

# 2. Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_model_path,
    load_in_4bit = False, # IMPORTANT: Load in 16bit for merging
    dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True,
)

# 3. Merge and Save
print(f"Merging and saving to: {output_merged_path}...")
model.save_pretrained_merged(
    output_merged_path,
    tokenizer,
    save_method = "merged_16bit", # Options: "merged_16bit", "merged_4bit"
)

print("✅ Merge complete! You can now point vLLM to:", output_merged_path)


# A note about Unsloth / Speciality of Unsloth:
    # - It stores the model related info in 4bit but processes in full precision (fp16 or 32 - based on what we specified). 
    # hence, faster training.


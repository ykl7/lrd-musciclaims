import torch
import flash_attn

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Flash Attention Installed: {flash_attn.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

try:
    from flash_attn import flash_attn_func
    print("✅ Flash Attention 2 Function loaded successfully!")
except ImportError:
    print("❌ Flash Attention 2 failed to load.")

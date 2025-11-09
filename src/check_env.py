import os
import torch

# Get the rank of this specific process
rank = int(os.environ.get('RANK', -1))

print(f"================  PROCESS RANK: {rank}  ================\n")

# Print the environment variable that torchrun/slurm should be setting
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

# Ask PyTorch how many GPUs it can see
try:
    gpu_count = torch.cuda.device_count()
    print(f"torch.cuda.device_count() = {gpu_count}")

    # Ask PyTorch for the name of the GPU it is currently on
    if gpu_count > 0:
        current_device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device_index)
        print(f"torch.cuda.current_device() reports index: {current_device_index}")
        print(f"Device Name: {device_name}")
    else:
        print("PyTorch sees no available CUDA devices.")

except Exception as e:
    print(f"\n!!!!!! AN ERROR OCCURRED !!!!!!!")
    print(e)

print(f"\n=======================================================\n")

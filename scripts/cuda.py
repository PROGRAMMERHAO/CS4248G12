import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# If a GPU is available, let's print some additional info
if torch.cuda.is_available():
    print("CUDA is available. GPU details:")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
else:
    print("CUDA is not available. Running on CPU.")

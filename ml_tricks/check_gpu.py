import torch

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

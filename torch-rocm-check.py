#!/usr/bin/env python3
import torch


print(f"Checking for CUDA/ROCM with torch")

available = torch.cuda.is_available()

print(f"CUDA (or ROCM in this case) is available: {available}")

x = torch.rand(6, 4)
print(x)

"""
Day 3 — PyTorch Tensors
Run sections one at a time. Read every output. Predict before each print.
"""

import torch

print("="*70)
print("SECTION 1: Creating Tensors")
print("="*70)

# From a Python list
a = torch.tensor([1, 2, 3])
print(f"\nFrom list: {a}, dtype={a.dtype}, shape={a.shape}")

# From a list of lists (2D)
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\n2D tensor:\n{b}")
print(f"  shape={b.shape}, dtype={b.dtype}")
print(f"  Number of dimensions: {b.dim()}")
print(f"  Number of elements: {b.numel()}")

# Special tensors
print(f"\nzeros(2,3):\n{torch.zeros(2, 3)}")
print(f"\nones(2,3):\n{torch.ones(2, 3)}")
print(f"\narange(0, 10):\n{torch.arange(0, 10)}")
print(f"\nlinspace(0, 1, 5):\n{torch.linspace(0, 1, 5)}")

# Random tensors (you'll use these constantly)
torch.manual_seed(42)  # for reproducibility
print(f"\nrand(2,3) — uniform [0,1]:\n{torch.rand(2, 3)}")
print(f"\nrandn(2,3) — normal N(0,1):\n{torch.randn(2, 3)}")

# Specifying dtype
x_float = torch.tensor([1.0, 2.0, 3.0])
x_long = torch.tensor([1, 2, 3], dtype=torch.long)
print(f"\nfloat tensor: {x_float}, dtype={x_float.dtype}")
print(f"long tensor: {x_long}, dtype={x_long.dtype}")

print(f"\nnormal random tensor with specified dtype:\n{torch.randn(2, 3, dtype=torch.float16)}")

print("\n--- KEY POINT ---")
print("dtype matters. Floats for math, longs for indices/labels.")
print("In LLMs: model weights are float, token IDs are long.")

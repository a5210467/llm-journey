"""
Day 3 — Tensor Shapes (the most important skill in ML)
"""

import torch

print("="*70)
print("SECTION 2: Tensor Shapes — THE MOST IMPORTANT SKILL")
print("="*70)

# === Reshape ===
print("\n--- Reshape ---")
x = torch.arange(12)
print(f"Original: {x}, shape={x.shape}")
print(f"reshape(3,4):\n{x.reshape(3, 4)}")
print(f"reshape(2,2,3):\n{x.reshape(2, 2, 3)}")
print(f"reshape(-1, 4): {x.reshape(-1, 4).shape}  # -1 means 'figure it out'")
print(f"view of x: {x.reshape(-1,3).shape}  # same shape as reshape(-1,3)")

# === View vs Reshape (subtle but important) ===
print("\n--- View vs Reshape ---")
print("view() requires contiguous memory; reshape() handles either.")
print("Use view() when you can, reshape() when you can't.")

# === Adding/removing dimensions ===
print("\n--- unsqueeze and squeeze ---")
x = torch.tensor([1, 2, 3])
print(f"x: shape={x.shape}")
print(f"x.unsqueeze(0): shape={x.unsqueeze(0).shape}  # add dim at position 0")
print(f"x.unsqueeze(1): shape={x.unsqueeze(1).shape}  # add dim at position 1")
print(x.unsqueeze(1),"unsqueeze at position 1")  # (1, 3)
print(x.unsqueeze(0),"unsqueeze at position 0")  # (1, 3)

y = torch.tensor([[[1, 2, 3]]])
print(f"\ny: shape={y.shape}")
print(f"y.squeeze(): shape={y.squeeze().shape}  # remove all size-1 dims")

# === Transpose and permute ===
print("\n--- Transpose and permute ---")
m = torch.arange(12).reshape(3, 4)
print(f"m shape: {m.shape}")
print(f"m.T shape: {m.T.shape}")
print(f"m.transpose(0, 1) shape: {m.transpose(0, 1).shape}  # same as .T")

# In LLMs you'll see permute a lot for multi-head attention
t = torch.randn(2, 8, 12, 64)  # (batch, seq, heads, head_dim) - common pattern
print(f"\nLLM-style 4D tensor: {t.shape}  # (batch, seq, heads, head_dim)")
print(f"After permute(0, 2, 1, 3): {t.permute(0, 2, 1, 3).shape}")
print("  (batch, heads, seq, head_dim) — common for attention computation")

# === Broadcasting (THIS IS CRUCIAL) ===
print("\n--- Broadcasting ---")
print("PyTorch automatically expands compatible shapes for ops.\n")

a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = torch.tensor([10, 20, 30])             # shape (3,)
print(f"a shape: {a.shape}, b shape: {b.shape}")
print(f"a + b:\n{a + b}")
print("  b was 'broadcast' to (2,3) — added to each row")

c = torch.tensor([[100], [200]])  # shape (2, 1)
print(f"\na shape: {a.shape}, c shape: {c.shape}")
print(f"a + c:\n{a + c}")
print("  c was 'broadcast' across columns")

print("\nBroadcasting rules:")
print("  1. Align shapes from the right")
print("  2. Each dim must be equal, or one must be 1, or one must be missing")
print("  3. Size 1 dimensions get expanded")

# === Matrix multiplication ===
print("\n--- Matrix Multiplication ---")
a = torch.randn(3, 4)
b = torch.randn(4, 5)
print(f"a @ b: {a.shape} @ {b.shape} = {(a @ b).shape}")

# Batched matmul (you'll see this constantly in attention)
batch_a = torch.randn(3, 2, 3, 4)
batch_b = torch.randn(3, 2, 4, 5)
print(f"batched: {batch_a.shape} @ {batch_b.shape} = {(batch_a @ batch_b).shape}")
print("  PyTorch handles batched matmul automatically.")

print("\n" + "="*70)
print("REMEMBER: When debugging ML code, print .shape constantly!")
print("="*70)

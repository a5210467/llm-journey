"""
Day 4 Exercise B — Cleaner version using PyTorch built-ins.
Same logic as day4_softmax_version.py, but using F.cross_entropy.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PART 1: Generate data (same as before)
# =====================================================================
N_PER_CLASS = 100

pos_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([0.5, 0.5])
neg_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([-0.5, -0.5])

X = torch.cat([pos_points, neg_points], dim=0)
y = torch.cat([
    torch.ones(N_PER_CLASS, dtype=torch.long),
    torch.zeros(N_PER_CLASS, dtype=torch.long),
])

perm = torch.randperm(X.shape[0])
X, y = X[perm], y[perm]

print(f"X shape: {X.shape}, y shape: {y.shape}")


# =====================================================================
# PART 2: Initialize parameters
# =====================================================================
N_FEATURES = 2
N_CLASSES = 2

W = torch.randn(N_FEATURES, N_CLASSES, requires_grad=True)
b = torch.zeros(N_CLASSES, requires_grad=True)


# =====================================================================
# PART 3: The model and loss — using PyTorch built-ins (the clean way)
# =====================================================================

def model(X, W, b):
    """
    Linear layer producing logits, one per class.
    Returns RAW LOGITS — do not apply softmax here.
    F.cross_entropy expects logits, not probabilities.
    """
    return X @ W + b  # shape: (N, N_CLASSES)


def loss_fn(logits, y):
    """
    Cross-entropy loss using PyTorch's built-in.
    
    F.cross_entropy:
    - Takes raw logits (NOT probabilities)
    - Internally computes log_softmax + negative log-likelihood
    - Numerically stable (handles the log-sum-exp trick for you)
    - Returns the mean loss across the batch by default
    """
    return F.cross_entropy(logits, y)


# Sanity check — compare against the from-scratch version conceptually
print("\n--- Sanity check ---")
test_logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
test_y = torch.tensor([0, 1])
test_loss = loss_fn(test_logits, test_y)
print(f"Test logits:\n{test_logits}")
print(f"Test labels: {test_y}")
print(f"Loss (F.cross_entropy): {test_loss.item():.4f}")

# Optional: also see what softmax probabilities look like for inspection
test_probs = F.softmax(test_logits, dim=-1)
print(f"Softmax probs:\n{test_probs}")
print(f"(Each row sums to 1: {test_probs.sum(dim=-1)})")


# =====================================================================
# PART 4: Training loop (same as before)
# =====================================================================
learning_rate = 0.1
n_epochs = 200
losses = []

print(f"\nTraining for {n_epochs} epochs...")
print(f"{'Epoch':<8}{'Loss':<12}{'Accuracy':<12}")
print("-" * 32)

for epoch in range(n_epochs):
    logits = model(X, W, b)
    loss = loss_fn(logits, y)
    
    loss.backward()
    
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
        W.grad.zero_()
        b.grad.zero_()
    
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y).float().mean().item()
        losses.append(loss.item())
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:<8}{loss.item():<12.4f}{accuracy:<12.4f}")


# =====================================================================
# PART 5: Visualize (same as before)
# =====================================================================
xx, yy = torch.meshgrid(
    torch.linspace(-3, 3, 100),
    torch.linspace(-3, 3, 100),
    indexing='xy'
)
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    grid_logits = model(grid, W, b)
    # F.softmax for visualization (we want actual probabilities here)
    grid_probs = F.softmax(grid_logits, dim=-1)
    grid_class1_prob = grid_probs[:, 1].reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contourf(xx, yy, grid_class1_prob, levels=20, cmap='RdBu', alpha=0.7)
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', label='y=1', edgecolors='white', s=40)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', edgecolors='white', s=40)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Decision Boundary')
axes[0].legend()

axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True, alpha=0.3)

plt.savefig('phase1/day4_softmax_clean_result.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved visualization to phase1/day4_softmax_clean_result.png")

print("\n✓ Done")

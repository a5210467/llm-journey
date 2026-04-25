"""
Day 4 — Logistic Regression from scratch.
No nn.Module, no torch.optim. Just tensors and autograd.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PART 1: Generate a toy 2D dataset
# =====================================================================
# Two clusters of points, one centered at (1, 1), other at (-1, -1)
# Each point has a label: 1 for the positive cluster, 0 for negative

N_PER_CLASS = 100

# Positive class (label = 1): centered at (1.5, 1.5)
pos_points = torch.randn(N_PER_CLASS, 2) * 1.5 + torch.tensor([0.5, 0.5])
pos_labels = torch.ones(N_PER_CLASS)
#print(pos_labels)

# Negative class (label = 0): centered at (-1.5, -1.5)
neg_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([-0.5, -0.5])
neg_labels = torch.zeros(N_PER_CLASS)

# Combine and shuffle
X = torch.cat([pos_points, neg_points], dim=0)  # shape (200, 2)
y = torch.cat([pos_labels, neg_labels], dim=0)   # shape (200,)

# Shuffle (in unison)
perm = torch.randperm(X.shape[0])
X = X[perm]
y = y[perm]

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Number of positive examples: {int(y.sum().item())}")
print(f"Number of negative examples: {int((1-y).sum().item())}")

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='y=1', alpha=0.6)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', alpha=0.6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phase1/day4_data.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved data visualization to phase1/day4_data.png")


# =====================================================================
# PART 2: Initialize parameters
# =====================================================================
# w has shape (2,) — one weight per feature
# b has shape () — scalar bias
# requires_grad=True tells autograd to track these

w = torch.randn(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"\nInitial w: {w.data.tolist()}")
print(f"Initial b: {b.data.item()}")


# =====================================================================
# PART 3: Define the model (a function)
# =====================================================================
def model(X, w, b):
    """
    Linear layer + sigmoid.
    X: shape (N, 2)
    w: shape (2,)
    b: shape (1,)
    Returns: probabilities of shape (N,)
    """
    # Linear: z = X @ w + b
    #   X @ w: (N, 2) @ (2,) = (N,)
    #   + b: broadcasting (1,) onto (N,)
    logits = X @ w + b
    
    # Sigmoid: σ(z) = 1 / (1 + exp(-z))
    probs = torch.sigmoid(logits)
    
    return probs


def loss_fn(probs, y, eps=1e-7):
    """
    Binary cross-entropy loss.
    probs: predicted probabilities, shape (N,)
    y: true labels (0 or 1), shape (N,)
    eps: small number to prevent log(0)
    """
    # Clamp to avoid log(0)
    probs = torch.clamp(probs, eps, 1 - eps)
    
    # BCE: -[y * log(p) + (1-y) * log(1-p)]
    bce = -(y * torch.log(probs) + (1 - y) * torch.log(1 - probs))
    
    # Average over the batch
    return bce.mean()


# =====================================================================
# PART 4: Training loop
# =====================================================================
learning_rate = 0.1
n_epochs = 200
losses = []

print(f"\nTraining for {n_epochs} epochs...")
print(f"{'Epoch':<8}{'Loss':<12}{'Accuracy':<12}")
print("-" * 32)

for epoch in range(n_epochs):
    # Forward pass
    probs = model(X, w, b)
    loss = loss_fn(probs, y)
    
    # Backward pass
    loss.backward()
    
    # Manual update — no optimizer
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Reset gradients (critical — otherwise they accumulate)
        w.grad.zero_()
        b.grad.zero_()
    
    # Logging
    with torch.no_grad():
        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        losses.append(loss.item())
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:<8}{loss.item():<12.4f}{accuracy:<12.4f}")

print(f"\nFinal w: {w.data.tolist()}")
print(f"Final b: {b.data.item()}")


# =====================================================================
# PART 5: Visualize the learned decision boundary
# =====================================================================
# Create a grid of points
xx, yy = torch.meshgrid(
    torch.linspace(-3, 3, 100),
    torch.linspace(-3, 3, 100),
    indexing='xy'
)
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # shape (10000, 2)

# Predict on the grid
with torch.no_grad():
    grid_probs = model(grid, w, b)

# Reshape back to grid
grid_probs = grid_probs.reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: decision boundary
axes[0].contourf(xx, yy, grid_probs, levels=20, cmap='RdBu', alpha=0.7)
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', label='y=1', edgecolors='white', s=40)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', edgecolors='white', s=40)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Decision Boundary (color = P(y=1))')
axes[0].legend()

# Right: loss curve
axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Binary Cross-Entropy Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True, alpha=0.3)

plt.savefig('phase1/day4_result.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved result visualization to phase1/day4_result.png")


# =====================================================================
# PART 6: Check gradients (information theory connection)
# =====================================================================
print("\n" + "="*60)
print("INFORMATION THEORY CONNECTION")
print("="*60)

# Compute final cross-entropy
with torch.no_grad():
    final_probs = model(X, w, b)
    final_loss = loss_fn(final_probs, y).item()

# Entropy of the labels (ground truth)
p_pos = y.mean().item()
label_entropy = -(p_pos * np.log(p_pos) + (1-p_pos) * np.log(1-p_pos))

print(f"Cross-entropy of predictions: {final_loss:.4f} nats")
print(f"Entropy of labels: {label_entropy:.4f} nats")
print(f"KL divergence (approx): {final_loss - label_entropy:.4f} nats")
print()
print("Shannon: the lowest achievable cross-entropy IS the label entropy.")
print("Our cross-entropy > label entropy → there's still room to improve.")
print("In this toy problem, we got close because the classes are linearly separable.")

print("\n" + "="*60)
print("✓ Day 4 logistic regression complete!")
print("="*60)

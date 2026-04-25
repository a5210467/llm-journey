"""
Day 4 — Logistic Regression using softmax + F.cross_entropy.
Same problem as day4_logistic_regression.py, but using PyTorch built-ins.

Read the comments marked '# CHANGED:' to see what's different from the
sigmoid version.
"""

import torch
import torch.nn.functional as F  # CHANGED: added — needed for F.cross_entropy
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# PART 1: Generate data
# =====================================================================
N_PER_CLASS = 100

pos_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([1.5, 1.5])
neg_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([-1.5, -1.5])

# CHANGED: labels are now LONG tensors (integers), not floats
# F.cross_entropy expects class INDICES (0, 1, 2, ...) as long tensors
pos_labels = torch.ones(N_PER_CLASS, dtype=torch.long)   # CHANGED: was torch.ones(N_PER_CLASS)
neg_labels = torch.zeros(N_PER_CLASS, dtype=torch.long)  # CHANGED: was torch.zeros(N_PER_CLASS)

X = torch.cat([pos_points, neg_points], dim=0)
y = torch.cat([pos_labels, neg_labels], dim=0)

perm = torch.randperm(X.shape[0])
X = X[perm]
y = y[perm]

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"y dtype: {y.dtype}  # CHANGED: now torch.long, was torch.float")
print(f"Number of class-1 examples: {(y == 1).sum().item()}")  # CHANGED: was y.sum()
print(f"Number of class-0 examples: {(y == 0).sum().item()}")  # CHANGED: was (1-y).sum()

# Visualize the data (no change needed; comparison with y == 1 still works)
plt.figure(figsize=(8, 6))
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='y=1', alpha=0.6)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', alpha=0.6)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Training Data')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('phase1/day4_data_softmax.png', dpi=100, bbox_inches='tight')
plt.close()


# =====================================================================
# PART 2: Initialize parameters
# =====================================================================
N_FEATURES = 2  # CHANGED: added constants for clarity
N_CLASSES = 2   # CHANGED: added — explicit number of output classes

# CHANGED: w is now a MATRIX of shape (n_features, n_classes), not a vector
# Each column gives the weights for one class.
# Sigmoid version: w shape was (2,) — one set of weights, output 1 logit
# Softmax version: w shape is (2, 2) — n_classes sets of weights, output n_classes logits
w = torch.randn(N_FEATURES, N_CLASSES, requires_grad=True)  # CHANGED: was torch.randn(2, ...)

# CHANGED: b is now a VECTOR of shape (n_classes,), not (1,)
# One bias per class.
b = torch.zeros(N_CLASSES, requires_grad=True)  # CHANGED: was torch.zeros(1, ...)

print(f"\nInitial w shape: {w.shape}  # CHANGED: was (2,)")
print(f"Initial b shape: {b.shape}  # CHANGED: was (1,)")


# =====================================================================
# PART 3: Define the model and loss
# =====================================================================
def model(X, w, b):
    """
    Linear layer producing N_CLASSES logits per example.
    
    X: shape (N, 2)
    w: shape (2, 2)
    b: shape (2,)
    Returns: LOGITS of shape (N, 2) — NOT probabilities
    
    CHANGED from sigmoid version:
    - No torch.sigmoid call
    - Returns raw logits (F.cross_entropy will handle softmax internally)
    - Shape change: output is now (N, 2) instead of (N,)
    """
    # X @ w: (N, 2) @ (2, 2) = (N, 2)
    # + b: broadcasting (2,) onto (N, 2)
    logits = X @ w + b
    return logits  # CHANGED: was `return torch.sigmoid(logits)`


def loss_fn(logits, y):
    """
    Cross-entropy loss using PyTorch's built-in.
    
    CHANGED from sigmoid version:
    - Replaced manual BCE formula with F.cross_entropy
    - F.cross_entropy:
      * Takes raw LOGITS (not probabilities)
      * Internally does log_softmax + negative log-likelihood
      * Numerically stable (no need for the eps clamping we did before)
      * Returns mean loss across batch
    """
    return F.cross_entropy(logits, y)  # CHANGED: replaced the entire BCE block


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
    logits = model(X, w, b)        # CHANGED: variable name "logits" instead of "probs"
    loss = loss_fn(logits, y)      # CHANGED: pass logits, not probs
    
    # Backward pass (no change)
    loss.backward()
    
    # Manual update (no change)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    # Logging
    with torch.no_grad():
        # CHANGED: prediction is argmax over class dimension, not threshold at 0.5
        # logits.argmax(dim=-1): for each example, pick the class with highest logit
        predictions = logits.argmax(dim=-1)
        
        # CHANGED: accuracy comparison no longer needs .float() on predictions
        # because both are long tensors of class indices
        accuracy = (predictions == y).float().mean().item()
        losses.append(loss.item())
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:<8}{loss.item():<12.4f}{accuracy:<12.4f}")

print(f"\nFinal w:\n{w.data}")  # CHANGED: now a matrix
print(f"Final b: {b.data.tolist()}")


# =====================================================================
# PART 5: Visualize the decision boundary
# =====================================================================
xx, yy = torch.meshgrid(
    torch.linspace(-3, 3, 100),
    torch.linspace(-3, 3, 100),
    indexing='xy'
)
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    grid_logits = model(grid, w, b)  # CHANGED: variable name "logits"
    
    # CHANGED: need F.softmax to convert logits to probabilities for visualization
    # In the sigmoid version, model() already returned probs.
    # Here, model() returns logits, so we apply softmax explicitly when needed.
    grid_probs = F.softmax(grid_logits, dim=-1)
    
    # CHANGED: pick column 1 to get P(y=1), since softmax outputs probs for both classes
    grid_class1_prob = grid_probs[:, 1]

# Reshape back to grid
grid_class1_prob = grid_class1_prob.reshape(xx.shape)

# Plot (no change to this section)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contourf(xx, yy, grid_class1_prob, levels=20, cmap='RdBu', alpha=0.7)
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', label='y=1', edgecolors='white', s=40)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', edgecolors='white', s=40)
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')
axes[0].set_title('Decision Boundary (color = P(y=1))')
axes[0].legend()

axes[1].plot(losses)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True, alpha=0.3)

plt.savefig('phase1/day4_result_softmax.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved visualization to phase1/day4_result_softmax.png")


# =====================================================================
# PART 6: Information theory analysis (mostly unchanged)
# =====================================================================
print("\n" + "="*60)
print("INFORMATION THEORY CONNECTION")
print("="*60)

with torch.no_grad():
    final_logits = model(X, w, b)            # CHANGED: name change
    final_loss = loss_fn(final_logits, y).item()  # CHANGED: pass logits

# Compute label entropy
# CHANGED: y is now long, so use float() for the mean calculation
p_class1 = (y == 1).float().mean().item()
label_entropy = -(p_class1 * np.log(p_class1) + (1-p_class1) * np.log(1-p_class1))

print(f"Cross-entropy of predictions: {final_loss:.4f} nats")
print(f"Entropy of labels: {label_entropy:.4f} nats")
print(f"KL divergence (approx): {final_loss - label_entropy:.4f} nats")
print()
print("Shannon: the lowest achievable cross-entropy IS the label entropy.")
print("Our cross-entropy > label entropy → there's still room to improve.")

print("\n" + "="*60)
print("✓ Day 4 softmax version complete!")
print("="*60)

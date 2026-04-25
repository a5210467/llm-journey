"""
This is a placeholder for self-written code. You can replace this with your own code or logic as needed.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PART 1: Generate data
# =====================================================================
N_PER_CLASS = 100

pos_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([0.5, 0.5])
neg_points = torch.randn(N_PER_CLASS, 2) * 0.5 + torch.tensor([-0.5, -0.5])

pos_labels = torch.ones(N_PER_CLASS, dtype=torch.long)
neg_labels = torch.zeros(N_PER_CLASS, dtype=torch.long)

X = torch.cat([pos_points, neg_points], dim=0) 
y = torch.cat([pos_labels, neg_labels], dim=0)

perm = torch.randperm(X.shape[0]) #randomly shuffle the data

X = X[perm]
y = y[perm] #shuffle the labels in the same way as the data

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"y dtype: {y.dtype}  # CHANGED: now torch.long, was torch.float")
print(f"Number of class-1 examples: {(y == 1).sum().item()}")  # CHANGED: was y.sum()
print(f"Number of class-0 examples: {(y == 0).sum().item()}")  # CHANGED: was (1-y).sum()

# =====================================================================
# PART 2: Initialize parameters
# =====================================================================

# CHANGED: w is now a MATRIX of shape (n_features, n_classes), not a vector
# Each column gives the weights for one class.
# Sigmoid version: w shape was (2,) — one set of weights, output 1 logit
# Softmax version: w shape is (2, 2) — n_classes sets of weights, output n_classes logits

w = torch.randn(2, 2, requires_grad=True)  # CHANGED: was torch.randn(2, ...)

# CHANGED: b is now a VECTOR of shape (n_classes,), not (1,)
# One bias per class.

b = torch.randn(2, requires_grad=True)  # CHANGED: was torch.randn(1, ...)

print(f"Initial w: {w}")
print(f"Initial b: {b}")

# =====================================================================
# PART 3: Forward pass (compute logits and loss)
# =====================================================================
def model(X,w,b):
    return X @ w + b  # Matrix multiplication + broadcasted bias

def loss_fn(logits, y):
    return F.cross_entropy(logits, y)  # CHANGED: use cross-entropy loss

# =====================================================================
# PART 4: Training loop
# =====================================================================
learning_rate = 0.1
n_epochs = 100
losses = []

print(f"\nTraining for {n_epochs} epochs...")
print(f"{'Epoch':<8}{'Loss':<12}{'Accuracy':<12}")
print("-" * 32)

for epoch in range(n_epochs):
    logits = model(X, w, b)
    loss = loss_fn(logits, y)

    loss.backward()

    # Manually update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    with torch.no_grad():
        predicted_classes = torch.argmax(logits, dim=1)
        accuracy = (predicted_classes == y).float().mean().item()
        losses.append(loss.item())

    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:<8}{loss.item():<12.4f}{accuracy:<12.4f}")

print(f"\nFinal w:\n{w.data}")  # CHANGED: now a matrix
print(f"Final b: {b.data.tolist()}")    


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

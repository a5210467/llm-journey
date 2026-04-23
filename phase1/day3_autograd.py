"""
Day 3 — Autograd
Verify your understanding of gradients by computing them two ways.
"""

import torch
import math

print("="*70)
print("SECTION 3: Autograd — The Heart of Deep Learning")
print("="*70)

# === Example 1: Simple gradient ===
print("\n--- Example 1: y = x^2 ---")
print("Mathematical gradient: dy/dx = 2x")

x = torch.tensor(3.0, requires_grad=True)  # KEY: requires_grad=True
y = x ** 2
print(f"\nx = {x.item()}")
print(f"y = x^2 = {y.item()}")

y.backward()  # This computes the gradient
print(f"x.grad (autograd): {x.grad.item()}")
print(f"Expected: 2x = {2 * x.item()}")
print(f"Match: {x.grad.item() == 2 * x.item()}")

# === Example 2: Your homework problem from the plan ===
print("\n--- Example 2: f(x) = sin(x^2) * exp(x) at x=1.5 ---")
print("Math:  f'(x) = exp(x) * (sin(x^2) + 2x*cos(x^2))")

x = torch.tensor(1.5, requires_grad=True)
f = torch.sin(x ** 2) * torch.exp(x)
f.backward()

# Hand-computed gradient
hand_grad = math.exp(1.5) * (math.sin(1.5**2) + 2 * 1.5 * math.cos(1.5**2))

print(f"\nf({x.item()}) = {f.item():.6f}")
print(f"Autograd:   {x.grad.item():.6f}")
print(f"Hand-calc:  {hand_grad:.6f}")
print(f"Difference: {abs(x.grad.item() - hand_grad):.2e}")

# === Example 3: Gradients through a chain ===
print("\n--- Example 3: Chain rule in action ---")
print("Compute dL/dw where L = (wx - y)^2 (linear regression loss)")

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)  # input (no grad needed)
y = torch.tensor(7.0)  # target (no grad needed)

prediction = w * x
loss = (prediction - y) ** 2

print(f"\nw = {w.item()}, x = {x.item()}, y = {y.item()}")
print(f"prediction = w*x = {prediction.item()}")
print(f"loss = (pred - y)^2 = {loss.item()}")

loss.backward()
print(f"\ndL/dw (autograd): {w.grad.item()}")
print(f"Expected: 2*x*(wx - y) = 2*{x.item()}*({prediction.item()} - {y.item()}) = "
      f"{2 * x.item() * (prediction.item() - y.item())}")

# === Example 4: Gradients accumulate ===
print("\n--- Example 4: Gradients ACCUMULATE (important!) ---")
x = torch.tensor(2.0, requires_grad=True)

# First backward pass
y1 = x ** 2
y1.backward()
print(f"After y1.backward(): x.grad = {x.grad.item()}")

# Second backward pass — gradients ADD
y2 = x ** 3
y2.backward()
print(f"After y2.backward(): x.grad = {x.grad.item()}")
print("  Notice: grad accumulated! That's why training loops do optimizer.zero_grad()")


# To reset:
x.grad.zero_()
print(f"After x.grad.zero_(): x.grad = {x.grad.item()}")

# === Example 5: Detaching ===
print("\n--- Example 5: detach() and torch.no_grad() ---")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

# detach() removes from gradient graph
y_detached = y.detach()
print(f"y requires grad: {y.requires_grad}")
print(f"y_detached requires grad: {y_detached.requires_grad}")

# torch.no_grad() is a context where grads aren't tracked
with torch.no_grad():
    z = x ** 2
print(f"z (computed under no_grad) requires grad: {z.requires_grad}")

print("\nUse cases:")
print("  - .detach() when you want a value without breaking the graph upstream")
print("  - torch.no_grad() during inference/evaluation (faster, less memory)")

print("\n" + "="*70)
print("Autograd is doing calculus for you. Trust it but understand it.")
print("="*70)

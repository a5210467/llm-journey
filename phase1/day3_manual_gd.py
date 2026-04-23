"""
Manual gradient descent — find minimum of f(x) = (x - 3)^2
True minimum is at x = 3.
"""
import torch

# Start from x = 0
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

print(f"Starting x = {x.item()}")
print(f"True minimum at x = 3\n")

for step in range(20):
    # Forward
    f = (x - 3) ** 2
    
    # Backward
    f.backward()
    
    # Manual update (don't use optimizer)
    with torch.no_grad():  # don't track this update in gradient graph
        x -= learning_rate * x.grad
        #print(f"learning_rate={learning_rate:.2f}, gradient={x.grad.item():.4f}")
        x.grad.zero_()  # reset gradient
    
    if step % 2 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"\nFinal x = {x.item():.4f} (should be close to 3)")

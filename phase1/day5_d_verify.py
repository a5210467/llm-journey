# day5_verify.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from day5_c_adam import MyAdam

torch.manual_seed(42)
N, D, K = 256, 8, 3
X = torch.randn(N, D)
y = torch.randint(0, K, (N,))

def make_model():
    torch.manual_seed(0)
    return nn.Linear(D, K, bias=True)

def train(opt_cls, n_steps=50):
    model = make_model()
    opt = opt_cls(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    losses, weights = [], []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        weights.append(model.weight.detach().clone())
    return losses, torch.stack(weights)

L_mine,  W_mine  = train(MyAdam)
L_torch, W_torch = train(torch.optim.Adam)

max_w_diff = (W_mine - W_torch).abs().max().item()
max_l_diff = max(abs(a - b) for a, b in zip(L_mine, L_torch))
print(f"max |W_mine - W_torch| = {max_w_diff:.2e}")
print(f"max |L_mine - L_torch| = {max_l_diff:.2e}")
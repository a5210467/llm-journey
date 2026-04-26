# day5_momentum.py
import torch
import matplotlib.pyplot as plt

H = torch.diag(torch.tensor([1.0, 25.0]))
def grad_f(x): return H @ x

def run_momentum(x0, lr, beta, n_steps):
    x = x0.clone()
    m = torch.zeros_like(x)
    traj = [x.clone()]
    for _ in range(n_steps):
        g = grad_f(x)
        m = beta * m + (1 - beta) * g
        x = x - lr * m
        traj.append(x.clone())
    return torch.stack(traj)

x0 = torch.tensor([1.0, 1.0])
for (lr, beta) in [(0.04, 0.0), (0.04, 0.9), (0.04, 0.99)]:
    traj = run_momentum(x0, lr, beta, 60)
    plt.plot(traj[:,0], traj[:,1], '.-', label=f'lr={lr}, β={beta}')

plt.scatter([0],[0], c='k', marker='*', s=80, label='optimum')
plt.legend(); plt.grid(True); plt.xlabel('x'); plt.ylabel('y')
plt.title('Adding momentum'); plt.show()
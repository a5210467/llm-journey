import torch
import matplotlib.pyplot as plt

# f(x,y) = 1/2 (x^2 + 25 y^2). A ravine: gentle in x, steep in y.
H = torch.diag(torch.tensor([1.0, 25.0]))
def grad_f(x): return H @ x

def run_sgd(x0, lr, n_steps):
    x = x0.clone()
    traj = [x.clone()]
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        traj.append(x.clone())
    return torch.stack(traj)

x0 = torch.tensor([1.0, 1.0])
# Three regimes:
# lr=0.02:  small enough to be stable, painfully slow along x
# lr=0.077: ~optimal for SGD on this problem
# lr=0.078: near stability boundary — watch y-direction zigzag wildly
for lr in [0.02, 0.077, 0.078]:
    traj = run_sgd(x0, lr, 60)
    plt.plot(traj[:,0], traj[:,1], '.-', label=f'lr={lr}')

plt.scatter([0],[0], c='k', marker='*', s=80, label='optimum')
plt.legend(); plt.grid(True); plt.xlabel('x'); plt.ylabel('y')
plt.title('SGD on a ravine'); plt.show()
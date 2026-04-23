import torch

a = torch.ones(3, 4)
b = torch.ones(4)
c = torch.ones(3, 1)
d = torch.ones(2, 3, 4)

# What's the shape of:
# (a + b) = 2 * torch.one(3,4)?
# (a + c) = 2 * torch.one(3,4)?
# (d + a) = 2 * torch.one(2,3,4)?
# (d + b) = 2 * torch.one(2,3,4)?
# (a + d.sum(dim=0)) = 3 * torch.one(2,3,4)?

print(a + b == 2 * torch.ones(3, 4))
print(a + c == 2 * torch.ones(3, 4))
print(d + a == 2 * torch.ones(2, 3, 4))
print(d + b == 2 * torch.ones(2, 3, 4))
print(a + d.sum(dim=0) == 3 * torch.ones(3, 4))
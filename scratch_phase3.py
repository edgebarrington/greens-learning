import torch
import numpy as np

from models.constrained_kernel import ConstrainedKernel
from data.data_loader import PoissonDataset, DataLoader

# -------------------------------
# Kernel sanity check
# -------------------------------

kernel = ConstrainedKernel(hidden_dim=32, num_layers=2)

x = torch.linspace(0, 1, 20)
xp = torch.linspace(0, 1, 20)

G1 = kernel(x, xp)
G2 = kernel(xp, x)

print("Max symmetry error:", torch.max(torch.abs(G1 - G2)).item())

x0 = torch.zeros_like(x)
x1 = torch.ones_like(x)

print("Boundary max (x=0):", torch.max(torch.abs(kernel(x0, xp))).item())
print("Boundary max (x=1):", torch.max(torch.abs(kernel(x1, xp))).item())
print("Boundary max (x'=0):", torch.max(torch.abs(kernel(x, x0))).item())
print("Boundary max (x'=1):", torch.max(torch.abs(kernel(x, x1))).item())

# -------------------------------
# Data loader sanity check
# -------------------------------

train_data = PoissonDataset("data/datasets/train.npz")
loader = DataLoader(train_data, batch_size=8, shuffle=True, seed=0)

batch = next(iter(loader))

print("Batch f shape:", batch["f"].shape)
print("Batch u shape:", batch["u"].shape)
print("Boundary max (batch u):", abs(batch["u"][:, [0, -1]]).max())
# -------------------------------
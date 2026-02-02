import torch
import torch.nn as nn
from models.kernel_mlp import KernelMLP

class NoSymmetryKernel(nn.Module):
    """
    Kernel with boundary conditions but NO symmetry enforcement.
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.base_kernel = KernelMLP(hidden_dim, num_layers)

    def forward(self, x, xp):
        envelope = x * (1 - x) * xp * (1 - xp)
        return envelope * self.base_kernel(x, xp)

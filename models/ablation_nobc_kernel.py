import torch
import torch.nn as nn
from models.kernel_mlp import KernelMLP


class NoBCKernel(nn.Module):
    """
    Symmetric kernel WITHOUT boundary condition enforcement.
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.base_kernel = KernelMLP(hidden_dim, num_layers)

    def forward(self, x, xp):
        g_x_xp = self.base_kernel(x, xp)
        g_xp_x = self.base_kernel(xp, x)
        g_sym = 0.5 * (g_x_xp + g_xp_x)
        return g_sym

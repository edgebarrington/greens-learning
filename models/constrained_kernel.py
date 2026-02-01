import torch
import torch.nn as nn

from models.kernel_mlp import KernelMLP


class ConstrainedKernel(nn.Module):
    """
    Physics-constrained Green's function kernel.

    Enforces:
    - Symmetry: G(x, x') = G(x', x)
    - Dirichlet BCs via multiplicative envelope
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.base_kernel = KernelMLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(self, x, xp):
        """
        Parameters
        ----------
        x : tensor of shape (M,)
        xp : tensor of shape (M,)

        Returns
        -------
        G : tensor of shape (M,)
            Symmetric, boundary-constrained kernel values.
        """
        g_x_xp = self.base_kernel(x, xp)
        g_xp_x = self.base_kernel(xp, x)

        g_sym = 0.5 * (g_x_xp + g_xp_x)

        envelope = x * (1.0 - x) * xp * (1.0 - xp)

        return envelope * g_sym

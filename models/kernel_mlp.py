import torch
import torch.nn as nn


class KernelMLP(nn.Module):
    """
    Raw MLP parameterization of a kernel g(x, x').
    No symmetry or boundary conditions enforced here.
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()

        layers = []
        in_dim = 2

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x, xp):
        """
        Parameters
        ----------
        x : tensor of shape (M,)
        xp : tensor of shape (M,)

        Returns
        -------
        g : tensor of shape (M,)
        """
        inp = torch.stack([x, xp], dim=-1)
        out = self.net(inp)
        return out.squeeze(-1)

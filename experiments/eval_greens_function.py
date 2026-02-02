import torch
import numpy as np

from models.constrained_kernel import ConstrainedKernel
from solvers.analytic_greens import make_grid, greens_matrix


def compute_learned_greens(kernel, x):
    """
    Compute learned Green's matrix G(x_i, x_j).
    """
    N = len(x)

    x_i = x.view(N, 1).expand(N, N).reshape(-1)
    x_j = x.view(1, N).expand(N, N).reshape(-1)

    with torch.no_grad():
        G_flat = kernel(x_i, x_j)

    return G_flat.view(N, N)


if __name__ == "__main__":
    # Grid
    N = 100
    x_np, h = make_grid(N)
    x = torch.tensor(x_np, dtype=torch.float32)

    # Load trained kernel
    kernel = ConstrainedKernel(hidden_dim=64, num_layers=3)
    kernel.load_state_dict(torch.load("checkpoints/kernel_final.pt"))
    kernel.eval()

    # Learned Green's function
    G_learned = compute_learned_greens(kernel, x).numpy()

    # Analytic Green's function
    G_true = greens_matrix(x_np)

    # Relative Frobenius error
    rel_error = np.linalg.norm(G_learned - G_true) / np.linalg.norm(G_true)

    print("Relative Frobenius error (Green's function):", rel_error)

# Plotting
import matplotlib.pyplot as plt

# Plot analytic vs learned Green's function
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(G_true, origin="lower", extent=[0,1,0,1])
axes[0].set_title("Analytic Green's Function")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(G_learned, origin="lower", extent=[0,1,0,1])
axes[1].set_title("Learned Green's Function")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(G_learned - G_true, origin="lower", extent=[0,1,0,1])
axes[2].set_title("Difference (Learned - True)")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

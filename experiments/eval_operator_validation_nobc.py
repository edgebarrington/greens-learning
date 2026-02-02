import torch
import numpy as np

from models.ablation_nobc_kernel import NoBCKernel
from data.data_loader import PoissonDataset, DataLoader

@torch.no_grad()
def evaluate_operator(kernel, dataset_path, batch_size=32):
    dataset = PoissonDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = torch.tensor(dataset.x, dtype=torch.float32)
    h = x[1] - x[0]

    rel_errors = []

    for batch_np in loader:
        batch = {
            "x": x,
            "f": torch.tensor(batch_np["f"], dtype=torch.float32),
            "u": torch.tensor(batch_np["u"], dtype=torch.float32),
        }

        # Build Green's matrix once per batch
        N = len(x)
        x_i = x.view(N, 1).expand(N, N).reshape(-1)
        x_j = x.view(1, N).expand(N, N).reshape(-1)

        G = kernel(x_i, x_j).view(N, N)

        # Operator application
        u_pred = torch.matmul(batch["f"], G.T) * h

        # Relative L2 error per sample
        diff = u_pred - batch["u"]
        num = torch.norm(diff, dim=1)
        denom = torch.norm(batch["u"], dim=1)

        rel_errors.append((num / denom).cpu().numpy())
    # check boundary leakage
    boundary_max = torch.max(torch.abs(u_pred[:, [0, -1]])).item()
    print("Boundary violation max |u(0)|, |u(1)|:", boundary_max)

    rel_errors = np.concatenate(rel_errors)
    return rel_errors


if __name__ == "__main__":
    # Load trained kernel
    kernel = NoBCKernel(hidden_dim=64, num_layers=3)
    kernel.load_state_dict(torch.load("checkpoints/kernel_nobc.pt"))
    kernel.eval()


    rel_errors = evaluate_operator(
        kernel,
        dataset_path="data/datasets/val.npz",
        batch_size=32,
    )

    print("Validation relative L2 error:")
    print("  mean :", rel_errors.mean())
    print("  median:", np.median(rel_errors))
    print("  90% quantile:", np.quantile(rel_errors, 0.9))


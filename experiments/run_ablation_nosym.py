import torch
import os

from models.ablation_nosym_kernel import NoSymmetryKernel
from data.data_loader import PoissonDataset, DataLoader
from training.loss import operator_loss

import torch.optim as optim


def train_nosym_kernel(
    dataset_path="data/datasets/train.npz",
    num_epochs=30,
    batch_size=16,
    lr=1e-3,
):
    dataset = PoissonDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, seed=0)

    x = torch.tensor(dataset.x, dtype=torch.float32)
    h = x[1] - x[0]

    kernel = NoSymmetryKernel(hidden_dim=64, num_layers=3)
    optimizer = optim.Adam(kernel.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        nb = 0

        for batch_np in loader:
            batch = {
                "x": x,
                "f": torch.tensor(batch_np["f"], dtype=torch.float32),
                "u": torch.tensor(batch_np["u"], dtype=torch.float32),
            }

            optimizer.zero_grad()
            loss = operator_loss(kernel, batch, h)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            nb += 1

        print(f"[NoSym] Epoch {epoch+1:03d} | Loss {epoch_loss/nb:.6e}")

    torch.save(kernel.state_dict(), "checkpoints/kernel_nosym.pt")
    print("Saved NoSym kernel")


if __name__ == "__main__":
    train_nosym_kernel()

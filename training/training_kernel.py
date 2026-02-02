import torch
import torch.optim as optim

from models.constrained_kernel import ConstrainedKernel
from data.data_loader import PoissonDataset, DataLoader
from training.loss import operator_loss

import os
os.makedirs("checkpoints", exist_ok=True)

def train_kernel(
    dataset_path,
    num_epochs=20,
    batch_size=16,
    lr=1e-3,
    hidden_dim=64,
    num_layers=3,
):
    # Load dataset
    dataset = PoissonDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, seed=0)

    # Convert grid to torch
    x = torch.tensor(dataset.x, dtype=torch.float32)
    h = x[1] - x[0]

    # Initialize kernel
    kernel = ConstrainedKernel(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    optimizer = optim.Adam(kernel.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

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
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6e}")

    torch.save(kernel.state_dict(), "checkpoints/kernel_final.pt")
    print("Saved trained kernel to checkpoints/kernel_final.pt")
    return kernel

from training.training_kernel import train_kernel

if __name__ == "__main__":
    train_kernel(
        dataset_path="data/datasets/train.npz",
        num_epochs=30,
        batch_size=16,
        lr=1e-3,
        hidden_dim=64,
        num_layers=3,
    )

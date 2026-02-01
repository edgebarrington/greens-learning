import torch


def operator_loss(kernel, batch, h):
    """
    Compute solution-level loss induced by the learned kernel operator.

    Parameters
    ----------
    kernel : nn.Module
        Constrained kernel G_theta(x, x').
    batch : dict
        Batch from DataLoader with keys 'x', 'f', 'u'.
        f, u have shape (B, N).
    h : float
        Grid spacing.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
    x = batch["x"]                # (N,)
    f = batch["f"]                # (B, N)
    u_true = batch["u"]           # (B, N)

    B, N = f.shape

    # Build all (x_i, x_j) pairs
    x_i = x.view(N, 1).expand(N, N).reshape(-1)
    x_j = x.view(1, N).expand(N, N).reshape(-1)

    # Evaluate kernel on grid
    G_flat = kernel(x_i, x_j)     # (N*N,)
    G = G_flat.view(N, N)         # (N, N)

    # Apply operator: u_pred = G @ f^T * h
    u_pred = torch.matmul(f, G.T) * h  # (B, N)

    # Mean squared error
    loss = torch.mean((u_pred - u_true) ** 2)

    return loss

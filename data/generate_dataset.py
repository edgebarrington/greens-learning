import numpy as np
from pathlib import Path

from data.generate_forcing import generate_sine_forcing
from solvers.spectral_solver import spectral_poisson
from solvers.analytic_greens import make_grid


def generate_dataset(
    split_name,
    num_samples,
    N=100,
    K=10,
    p=2.0,
    seed=0,
    out_dir="data/datasets"
):
    """
    Generate a dataset of (f, u) pairs for the Poisson problem.

    Parameters
    ----------
    split_name : str
        Name of the dataset split ('train', 'val', 'test').
    num_samples : int
        Number of samples to generate.
    N : int
        Number of grid points.
    K : int
        Maximum sine mode for forcing.
    p : float
        Decay exponent for forcing coefficients.
    seed : int
        Random seed for reproducibility.
    out_dir : str
        Output directory for .npz files.
    """
    rng = np.random.default_rng(seed)

    x, h = make_grid(N)

    f_all = np.zeros((num_samples, N))
    u_all = np.zeros((num_samples, N))
    coeffs_all = np.zeros((num_samples, K))

    for i in range(num_samples):
        f, coeffs = generate_sine_forcing(
            x,
            K=K,
            p=p,
            seed=rng.integers(1e9)
        )

        u = spectral_poisson(f, x)

        f_all[i] = f
        u_all[i] = u
        coeffs_all[i] = coeffs

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_path = Path(out_dir) / f"{split_name}.npz"
    np.savez(
        out_path,
        x=x,
        f=f_all,
        u=u_all,
        coeffs=coeffs_all,
        meta=dict(
            N=N,
            K=K,
            p=p,
            seed=seed,
            solver="spectral_poisson"
        )
    )

    print(f"Saved {split_name} dataset to {out_path}")

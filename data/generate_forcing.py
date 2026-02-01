import numpy as np

def generate_sine_forcing(
    x,
    K=10,
    p=2.0,
    seed=None
):
    """
    Generate a smooth forcing function using a truncated sine series:
        f(x) = sum_{k=1}^K a_k sin(k pi x)

    Coefficients a_k are drawn from N(0, k^{-2p}).

    Parameters
    ----------
    x : ndarray of shape (N,)
        Grid points in [0,1].
    K : int
        Maximum sine mode.
    p : float
        Decay exponent controlling smoothness.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    f : ndarray of shape (N,)
        Forcing function evaluated on x.
    coeffs : ndarray of shape (K,)
        Sine coefficients a_k.
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(x)
    f = np.zeros(N)

    k_vals = np.arange(1, K + 1)

    # Variance decay: sigma_k ~ k^{-p}
    sigma = k_vals ** (-p)

    coeffs = np.random.normal(loc=0.0, scale=sigma)

    for k, a_k in zip(k_vals, coeffs):
        f += a_k * np.sin(np.pi * k * x)

    return f, coeffs

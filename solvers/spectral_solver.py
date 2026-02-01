import numpy as np

def spectral_poisson(f, x):
    """
    Solve -u'' = f on (0,1) with u(0)=u(1)=0
    using a sine-series spectral method.

    Parameters
    ----------
    f : ndarray of shape (N,)
        Forcing evaluated on the grid.
    x : ndarray of shape (N,)
        Grid points in [0,1].

    Returns
    -------
    u : ndarray of shape (N,)
        Spectral solution including boundary points.
    """
    N = len(x)

    # Interior grid (exclude boundaries)
    x_inner = x[1:-1]
    f_inner = f[1:-1]

    M = N - 2  # number of interior points
    k = np.arange(1, M + 1)

    # Build sine basis matrix
    S = np.sin(np.pi * np.outer(x_inner, k))

    # Project f onto sine modes
    f_hat = (2.0 / (M + 1)) * (S.T @ f_inner)

    # Solve in spectral space
    u_hat = f_hat / (np.pi * k) ** 2

    # Reconstruct u
    u_inner = S @ u_hat

    # Pad boundaries
    u = np.zeros(N)
    u[1:-1] = u_inner

    return u

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def finite_difference_poisson(f, h):
    """
    Solve -u'' = f on (0,1) with u(0)=u(1)=0
    using second-order finite differences.

    Parameters
    ----------
    f : ndarray of shape (N,)
        Forcing evaluated on the grid.
        Boundary values are assumed to be zero.
    h : float
        Grid spacing.

    Returns
    -------
    u : ndarray of shape (N,)
        Numerical solution including boundary points.
    """
    N = len(f)

    # Interior points only (exclude boundaries)
    f_inner = f[1:-1]

    main_diag = 2.0 * np.ones(N - 2)
    off_diag = -1.0 * np.ones(N - 3)

    A = diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr"
    ) / (h ** 2)

    u_inner = spsolve(A, f_inner)

    # Pad with boundary conditions
    u = np.zeros(N)
    u[1:-1] = u_inner

    return u

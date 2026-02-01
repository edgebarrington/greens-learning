import numpy as np
from solvers.analytic_greens import make_grid, greens_matrix

x, h = make_grid(100)
G = greens_matrix(x)

print("Symmetry error:", np.max(np.abs(G - G.T)))
print("Boundary row max:", np.max(np.abs(G[0, :])))

# ----- Phase 1.2: Discrete convolution test -----

# Forcing function: f(x) = sin(pi x)
f = np.sin(np.pi * x)

# Discrete convolution u_i = sum_j G_ij f_j h
u_numeric = G @ f * h

# Analytic solution: u(x) = sin(pi x) / pi^2
u_true = np.sin(np.pi * x) / (np.pi ** 2)

# Relative L2 error
rel_error = np.linalg.norm(u_numeric - u_true) / np.linalg.norm(u_true)

print("Relative L2 error (convolution):", rel_error)

# ----- Phase 1.3: Finite difference solver -----

from solvers.finite_difference import finite_difference_poisson

# Same forcing function
f = np.sin(np.pi * x)

# Finite difference solution
u_fd = finite_difference_poisson(f, h)

# Analytic solution
u_true = np.sin(np.pi * x) / (np.pi ** 2)

# Relative error
rel_error_fd = np.linalg.norm(u_fd - u_true) / np.linalg.norm(u_true)

print("Relative L2 error (finite difference):", rel_error_fd)

# ----- Phase 1.4: Spectral solver -----

from solvers.spectral_solver import spectral_poisson

# Same forcing
f = np.sin(np.pi * x)

# Spectral solution
u_spec = spectral_poisson(f, x)

# Analytic solution
u_true = np.sin(np.pi * x) / (np.pi ** 2)

# Relative error
rel_error_spec = np.linalg.norm(u_spec - u_true) / np.linalg.norm(u_true)

print("Relative L2 error (spectral):", rel_error_spec)

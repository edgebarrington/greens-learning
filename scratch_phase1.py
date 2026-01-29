import numpy as np
from solvers.analytic_greens import make_grid, greens_matrix

x, h = make_grid(100)
G = greens_matrix(x)

print("Symmetry error:", np.max(np.abs(G - G.T)))
print("Boundary row max:", np.max(np.abs(G[0, :])))

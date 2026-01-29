import numpy as np

def make_grid(N):
    x = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    return x, h

def greens_function(x, xp):
    return np.where(x <= xp, x * (1 - xp), xp * (1 - x))

def greens_matrix(x):
    X, XP = np.meshgrid(x, x, indexing="ij")
    return greens_function(X, XP)

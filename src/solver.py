"""Finite-difference solver for 2D steady heat conduction on a unit plate.

Solves  div(k grad T) + q = 0  with fixed temperatures on the four edges.
Conductivity varies across the plate: k(x) = k0 * exp(g * x), so a plate with
g != 0 conducts better on one side (a simple graded material).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 64  # interior grid points per side -> 64 x 64 = 4096 unknowns


def solve(k0, g, q, t_top, t_bottom, t_left, t_right, n=N):
    """Return the (n, n) temperature field. Row 0 is the top edge, column 0 the left edge."""
    h = 1.0 / (n + 1)
    x = h * np.arange(1, n + 1)                    # x position of each column of nodes
    k_c = np.tile(k0 * np.exp(g * x), (n, 1))              # k at the nodes
    k_w = np.tile(k0 * np.exp(g * (x - h / 2)), (n, 1))    # k on each node's west face
    k_e = np.tile(k0 * np.exp(g * (x + h / 2)), (n, 1))    # k on each node's east face

    idx = np.arange(n * n).reshape(n, n)
    rows = [idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()]
    cols = [idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()]
    vals = [-(k_w + k_e + 2 * k_c).ravel(),       # centre
            k_e[:, :-1].ravel(), k_w[:, 1:].ravel(),   # east, west neighbours
            k_c[:-1, :].ravel(), k_c[1:, :].ravel()]   # south, north neighbours
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * n, n * n))

    b = np.full((n, n), -q * h**2)
    b[0, :] -= k_c[0, :] * t_top
    b[-1, :] -= k_c[-1, :] * t_bottom
    b[:, 0] -= k_w[:, 0] * t_left
    b[:, -1] -= k_e[:, -1] * t_right
    return spla.spsolve(A, b.ravel()).reshape(n, n)

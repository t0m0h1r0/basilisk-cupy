"""Spectrally split damping ``gamma Q^T M Q`` built from the CCD Laplacian (docs/notes/42) -- an evaluation.

``s = L_h / lambda_max``, ``L_h = -h^2 (D2_x (+) D2_y)`` with the CCD second derivative (Neumann closure on
walls, periodic otherwise), ``P = (I + alpha s^m)^{-1}``, ``Q = I - P``. Dense matrices: small grids only.
:func:`attach` adds the implicit sub-step ``(M + dt gamma Q^T M Q) u = M u`` (``M`` the cell density) to
CenteredNS at the hook ``viscous_term``.
"""

from __future__ import annotations

import numpy as np

from basilisk_cupy.discretization.ccd import ccd_matrices


def laplacian_1d(n: int, kind: str = "neumann") -> np.ndarray:
    """``-h^2 D2`` of the CCD (dimensionless)."""
    D2 = ccd_matrices(n, 1.0, kind, kind)[1]
    return -D2


def filters(nx: int, ny: int, alpha: float = 10.0, m: int = 4, kinds=("neumann", "neumann")):
    """``(P, Q, s)`` on the flattened ``(nx, ny)`` cell array (C order, index i * ny + j)."""
    Lx, Ly = laplacian_1d(nx, kinds[0]), laplacian_1d(ny, kinds[1])
    lam = max(np.abs(np.linalg.eigvals(Lx)).max(), 1e-30) + max(np.abs(np.linalg.eigvals(Ly)).max(), 1e-30)
    L = np.kron(Lx, np.eye(ny)) + np.kron(np.eye(nx), Ly)
    s = L / lam
    S = np.linalg.matrix_power(s, m)
    P = np.linalg.inv(np.eye(nx * ny) + alpha * S)
    return P, np.eye(nx * ny) - P, s


def attach(ns, gamma: float, alpha: float = 10.0, m: int = 4, kinds=("neumann", "neumann")):
    grid = ns.grid
    nx, ny = grid.n
    _, Q, _ = filters(nx, ny, alpha, m, kinds)
    cache = {}

    def hook(_ns):
        dt = float(_ns.clock.dt)
        rho = _ns.rho if np.ndim(_ns.rho) else np.full(grid.n, float(_ns.rho))
        Md = np.asarray(rho, dtype=float).ravel()
        key = (dt, Md.tobytes())
        if key not in cache:
            cache.clear()
            A = np.diag(Md) + dt * gamma * Q.T @ (Md[:, None] * Q)
            cache[key] = np.linalg.inv(A) * Md[None, :]
        Ainv = cache[key]
        for c in _ns.u:
            c.interior = (Ainv @ np.asarray(c.interior).ravel()).reshape(grid.n)
        _ns.u.fill_ghosts()

    ns.on("viscous_term", hook)
    return Q

"""ACLS re-initialisation (Desjardins et al. 2008; Chiodi & Desjardins 2017) on the CLS field."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.cls import CLSInterface, _acls_flux_divergence, psi_from_phi


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 4])
def test_planar_profile_is_an_exact_discrete_equilibrium_for_2017(theta):
    """A planar H_eps profile at any angle: the 2017 flux balances to round-off (the face gradient of
    phi_map is exact for a linear function); the 2008 flux leaves 8.1e-4 / 7.1e-4 / 1.9e-3 (times h)."""
    n = 64
    h, eps = 1.0 / n, 1.5 / n
    x = (np.arange(n) + 0.5) * h - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    nx, ny = np.cos(theta), np.sin(theta)
    d = X * nx + Y * ny
    q = psi_from_phi(d, eps)
    core = (np.abs(d) < 3 * eps) & (np.abs(X) < 0.4) & (np.abs(Y) < 0.4)
    args = (np.full_like(q, nx), np.full_like(q, ny), eps, h)
    new = np.abs(_acls_flux_divergence(q, *args, "acls2017", BoundaryConditions())[core]).max() * h
    old = np.abs(_acls_flux_divergence(q, *args, "acls2008", BoundaryConditions())[core]).max() * h
    assert new < 1e-14 and old > 7e-4


def _circle(n, backend):
    R, cx, cy = 10.0 / n, 0.0123, -0.0217
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    return CLSInterface.from_level_set(grid, R - xp.sqrt((x - cx) ** 2 + (y - cy) ** 2), BoundaryConditions())


@pytest.mark.parametrize("method,change", [("acls2008", 6.36e-3), ("acls2017", 3.21e-4), ("compression", 2.86e-4)])
def test_static_circle_hundred_reinitialisations(backend, method, change):
    """Circle of diameter 20 cells, N = 64, 100 re-initialisations of one pseudo-step: max change of psi
    (1000: 3.2e-2 / 1.1e-3 / 1.1e-3; the 2017 step is a quarter of the others); mass to round-off."""
    cls = _circle(64, backend)
    q0 = to_numpy(cls.psi.interior).copy()
    m0 = cls.mass()
    for _ in range(100):
        cls.reinitialize(1, method=method)
    assert float(np.abs(to_numpy(cls.psi.interior) - q0).max()) == pytest.approx(change, rel=0.03)
    assert abs(cls.mass() / m0 - 1) < 1e-13


def test_translation_curvature_with_acls2017_every_step(backend):
    """Periodic translation, 4 pseudo-steps after every step, N = 32 / 64 / 128: curvature error (max over
    |phi| < eps) 0.066 / 0.015 / 0.007 (compression--diffusion 0.202 / 0.014 / 0.003; ACLS 2008 0.33 /
    0.35 / 0.16)."""
    n, R = 64, 0.2
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - xp.sqrt(x * x + y * y), BoundaryConditions.periodic(2))
    uf = FaceField(grid, [xp.ones(grid.face_shape(0)), 0.5 * xp.ones(grid.face_shape(1))])
    for _ in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
        cls.reinitialize(4, method="acls2017")
    xn, yn = to_numpy(x), to_numpy(y)
    r = np.sqrt(xn * xn + (((yn + 1.0) % 1.0) - 0.5) ** 2)
    k = to_numpy(cls.curvature())
    core = np.abs(R - r) < cls.eps
    assert float(np.abs(k - 1.0 / r)[core].max()) * R == pytest.approx(0.015, rel=0.05)

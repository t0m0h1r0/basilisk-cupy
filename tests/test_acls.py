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


@pytest.mark.parametrize("theta,imbalance", [(0.0, 4.09e-4), (np.pi / 4, 6.58e-5)])
def test_fccd_acls_planar_imbalance(theta, imbalance):
    """2008 form with FCCD operators on a planar profile, N = 64: max |div F| h = 4.09e-4 / 3.78e-4 / 6.58e-5
    at 0 / 0.3 / pi/4 rad (2nd order: 8.05e-4 / 7.13e-4 / 1.91e-3; the 2017 form: round-off)."""
    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    nx, ny = np.cos(theta), np.sin(theta)
    d = x * nx + y * ny
    cls = CLSInterface.from_level_set(grid, d, BoundaryConditions())
    core = (np.abs(d) < 3 * cls.eps) & (np.abs(x) < 0.35) & (np.abs(y) < 0.35)
    fn = cls._acls_face_normals(np.full_like(d, nx), np.full_like(d, ny))
    div = cls._acls_fccd_divergence(cls.psi.interior, fn)
    assert float(np.abs(div[core]).max()) / n == pytest.approx(imbalance, rel=0.02)


def test_fccd_acls_static_circle(backend):
    """Circle of 20 cells, N = 64, 100 / 1000 re-initialisations of one pseudo-step: 2.76e-4 / 7.17e-4,
    curvature error 0.003 / 0.013 (compression--diffusion 1.1e-3 / 0.018 after 1000)."""
    cls = _circle(64, backend)
    q0 = to_numpy(cls.psi.interior).copy()
    for _ in range(100):
        cls.reinitialize(1, method="acls-fccd")
    assert float(np.abs(to_numpy(cls.psi.interior) - q0).max()) == pytest.approx(2.76e-4, rel=0.03)


@pytest.mark.parametrize("factor,converges", [(4.0, True), (16.0, False)])
def test_fccd_acls_backward_euler(factor, converges):
    """Backward Euler by defect correction (2nd-order operator div(eps n_a^2 grad) without the cross and
    compression terms), first pseudo-step from a profile twice too thick: 8 corrections reduce the residual
    by 740 at 4 times the explicit dtau (1.7e-2 -> 2.3e-5), by 28 at 8, and stall at 16
    (diverge at 64)."""
    n = 64
    R = 10.0 / n
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    r = np.hypot(x - 0.0123, y + 0.0217)
    cls = CLSInterface.from_level_set(grid, R - r, BoundaryConditions())
    cls.psi.interior = 0.5 * (1 + np.tanh((R - r) / (4 * cls.eps)))  # twice too thick
    cls.psi.fill_ghosts()
    st = []
    cls.reinitialize(1, method="acls-fccd", acls_options={"implicit": True, "dtau_factor": factor, "dc_iters": 8,
                                                          "stats": st})
    assert (st[0][-1] / st[0][0] < 2e-3) == converges


def test_fccd_acls_translation(backend):
    """Translation, 4 pseudo-steps after every step, N = 32 / 64 / 128: curvature error 0.125 / 0.013 /
    0.005, psi L1 1.31e-3 / 2.88e-4 / 9.45e-5 (compression--diffusion 0.202 / 0.014 / 0.003)."""
    n, R = 64, 0.2
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - xp.sqrt(x * x + y * y), BoundaryConditions.periodic(2))
    uf = FaceField(grid, [xp.ones(grid.face_shape(0)), 0.5 * xp.ones(grid.face_shape(1))])
    for _ in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
        cls.reinitialize(4, method="acls-fccd")
    xn, yn = to_numpy(x), to_numpy(y)
    r = np.sqrt(xn * xn + (((yn + 1.0) % 1.0) - 0.5) ** 2)
    k = to_numpy(cls.curvature())
    core = np.abs(R - r) < cls.eps
    assert float(np.abs(k - 1.0 / r)[core].max()) * R == pytest.approx(0.013, rel=0.08)


@pytest.mark.parametrize("method,stable", [("compression", True), ("acls-fccd", False)])
def test_capillary_mode_needs_tangential_diffusion(backend, method, stable):
    """GFM static drop, density ratio 100, N = 64, 1.4 times tension.h's step, re-initialisation after every
    step: compression--diffusion (isotropic diffusion) stays stable, ACLS (diffusion along n only) grows."""
    from cases.capillary_dt_scan import growth

    r = growth(64, 0.01, 1.4, steps=800, backend=backend, reinit_every=1, reinit_method=method)
    assert r["stable"] == stable

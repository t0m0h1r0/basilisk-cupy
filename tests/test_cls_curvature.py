"""Curvature from the CLS field psi (TwoPhaseFlow paper eq. curvature_psi_2d)."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.cls import CLSInterface

R, CX, CY = 0.2, -0.0123, 0.0217


def _drop(n, backend, bc=None, shift=(0.0, 0.0)):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    dx, dy = x - CX - shift[0], y - CY - shift[1]
    if bc is not None:
        dx, dy = dx - xp.round(dx), dy - xp.round(dy)
    r = xp.sqrt(dx * dx + dy * dy)
    return CLSInterface.from_level_set(grid, R - r, bc or BoundaryConditions()), to_numpy(r)


def _core_error(cls, r, **kw):
    """Max |kappa - 1/r| R over |phi| < eps (kappa of the level set through the cell is 1/r)."""
    k = to_numpy(cls.curvature(**kw))
    core = np.abs(R - r) < cls.eps
    return float(np.abs(k - 1.0 / r)[core].max()) * R


def test_drop_sign_and_band(backend):
    cls, r = _drop(64, backend)
    k = to_numpy(cls.curvature())
    q = to_numpy(cls.psi.interior)
    band = (q > 0.01) & (q < 0.99)
    assert np.all(k[~band] == 0.0) and np.all(k[band & (np.abs(R - r) < cls.eps)] > 0)


def test_exact_profile_error_grows_like_one_over_h(backend):
    """psi = H_eps(R - r) with eps = 1.5 h: the profile is 3 cells wide at every N, so the CCD error of
    the psi derivatives is a fixed fraction and kappa's error grows like 1/eps. Measured (max over
    |phi| < eps, times R): 6.31e-4 / 1.21e-3 / 2.39e-3 / 4.78e-3 for N = 64 / 128 / 256 / 512."""
    e = [_core_error(_drop(n, backend)[0], _drop(n, backend)[1]) for n in (64, 128)]
    assert e[0] == pytest.approx(6.31e-4, rel=0.02) and e[1] == pytest.approx(1.21e-3, rel=0.02)


def test_phi_source_is_far_more_accurate_on_the_exact_profile(backend):
    """The same formula on phi = H_eps^{-1}(psi) (smooth): 4.9e-7 at N = 128 (psi: 1.21e-3)."""
    cls, r = _drop(128, backend)
    assert _core_error(cls, r, source="phi") < 1e-6


def test_code_variant_dccd_filter(backend):
    """TwoPhaseFlow's DCCD filter (eps_d = 0.05) on the derivatives: 1.25e-2 at N = 128, 10x the paper's."""
    cls, r = _drop(128, backend)
    assert _core_error(cls, r, code_variant=True) == pytest.approx(1.25e-2, rel=0.02)


def test_periodic_shift_by_whole_cells(backend):
    n = 32
    bc = BoundaryConditions.periodic(2)
    a, _ = _drop(n, backend, bc)
    b, _ = _drop(n, backend, bc, shift=(0.5, 13 / n))
    ka, kb = to_numpy(a.curvature()), to_numpy(b.curvature())
    np.testing.assert_allclose(np.roll(ka, (n // 2, 13), axis=(0, 1)), kb, atol=1e-9)


@pytest.mark.parametrize("reinit,every,expected", [(None, 0, 0.611), ("xi_sdf", 10, 11.2), ("compression", 10, 0.0372),
                                                   ("compression", 1, 0.01388)])
def test_after_transport(backend, reinit, every, expected):
    """Periodic translation by (1, 0.5) over t = 1 (CLS stage A + B), N = 32 / 64 / 128, max error over
    |phi| < eps: 0.394 / 0.611 / 0.805 without re-initialisation (no convergence); every 10 steps
    with xi-SDF 4.12 / 11.2 / 33.6, with compression--diffusion 0.134 / 0.0372 / 0.0167; every step
    with compression--diffusion 0.202 / 0.0139 / 0.00305. The error comes from the transported
    interface (grid-scale position noise 2.2e-4 at every N without re-initialisation), not from the
    curvature formula (docs/notes/17)."""
    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - xp.sqrt(x * x + y * y), BoundaryConditions.periodic(2))
    uf = FaceField(grid, [xp.ones(grid.face_shape(0)), 0.5 * xp.ones(grid.face_shape(1))])
    for s in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
        if reinit and (s + 1) % every == 0:
            cls.reinitialize(method=reinit)
    xn, yn = to_numpy(x), to_numpy(y)
    r = np.sqrt(xn * xn + (((yn + 1.0) % 1.0) - 0.5) ** 2)
    assert _core_error(cls, r) == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("order,expected", [(4, 0.1042), (6, 0.00850)])
def test_after_transport_corrected_divergence(backend, order, expected):
    """The translation above with the corrected divergence (``divergence="fccd"``), no re-initialisation:
    0.0788 / 0.104 / 0.119 (4th-order face values), 0.0221 / 0.00850 / 0.00881 (6th order) at
    N = 32 / 64 / 128 (plain: 0.394 / 0.611 / 0.805)."""
    from basilisk_cupy.discretization.ccd import FCCDFaceComplex

    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - xp.sqrt(x * x + y * y), BoundaryConditions.periodic(2),
                                      divergence="fccd", ops=FCCDFaceComplex(order=order))
    uf = FaceField(grid, [xp.ones(grid.face_shape(0)), 0.5 * xp.ones(grid.face_shape(1))])
    for _ in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
    xn, yn = to_numpy(x), to_numpy(y)
    r = np.sqrt(xn * xn + (((yn + 1.0) % 1.0) - 0.5) ** 2)
    assert _core_error(cls, r) == pytest.approx(expected, rel=0.02)

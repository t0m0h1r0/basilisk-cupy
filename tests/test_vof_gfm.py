"""VOF with the ghost-fluid pressure jump on the PLIC interface (TwoPhase(..., pressure_jump=True))."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS


def _drop(n, backend, rho2=1.0, pressure_jump=True):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.0625 - x * x - y * y), BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, pressure_jump=pressure_jump)
    tp.add_surface_tension(1.0)
    return grid, ns, f, tp


def test_plic_crossings_on_a_circle(backend):
    """The crossings of the PLIC lines with the centre-to-centre segments lie on the circle to 2.4e-4
    (0.016 h, N = 64; fraction() itself is O(h^2))."""
    n = 64
    grid, ns, f, tp = _drop(n, backend)
    h = 1.0 / n
    for ax in range(2):
        hl, hr, cut, th = (to_numpy(a) for a in tp._plic_cut(ax))
        i, j = np.nonzero(cut)
        if ax == 0:
            xc, yc = -0.5 + (i - 0.5 + th[i, j]) * h, -0.5 + (j + 0.5) * h
        else:
            xc, yc = -0.5 + (i + 0.5) * h, -0.5 + (j - 0.5 + th[i, j]) * h
        assert len(i) == 64 and np.abs(np.hypot(xc, yc) - 0.25).max() < 2.5e-4


@pytest.mark.parametrize("pj,jump,umax", [(True, 4.0031, 4.08e-3), (False, 4.0138, 4.23e-3)])
def test_static_drop(backend, pj, jump, umax):
    """N = 64, 10 steps: jump 4.0031 with the GFM jump (iforce.h: 4.0138); N = 32 / 128: 4.0279 / 3.9920
    (4.0379 / 4.0043). Parasitic velocities are set by the height-function curvature, not by the force."""
    grid, ns, f, tp = _drop(64, backend, pressure_jump=pj)
    xp = grid.xp
    x, y = grid.meshgrid()
    r = xp.sqrt(x * x + y * y)
    ns.start()
    for _ in range(10):
        ns.step()
    j = float(xp.mean(ns.p.interior[r < 0.1]) - xp.mean(ns.p.interior[r > 0.4]))
    u = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
    assert j == pytest.approx(jump, abs=2e-4) and u == pytest.approx(umax, rel=0.03)


def test_static_drop_density_ratio_10_reaches_rest(backend):
    """400 steps, tension.h's time step: 3.9e-8 (ratios 100 / 1000: 1.7e-8 / 4.3e-9; iforce.h 1.8e-9)."""
    grid, ns, f, tp = _drop(64, backend, rho2=0.1)
    m0 = float(f.interior.sum())
    ns.start()
    for _ in range(400):
        ns.step()
    u = max(float(grid.xp.max(grid.xp.abs(c.interior))) for c in ns.u)
    assert math.isfinite(u) and u < 1e-7 and abs(float(f.interior.sum()) / m0 - 1) < 1e-7


def test_capillary_wave(backend):
    """capwave.c, 8 / 16 / 32 / 64 points per wavelength: 0.1362 / 0.0215 / 0.00668 / 0.00159
    (iforce.h 0.1386 / 0.0243 / 0.00638 / 0.00232)."""
    from cases import capillary_wave as cw

    assert cw.run(16, backend, pressure_jump=True)["error"] == pytest.approx(0.13615, rel=1e-3)


def test_gravity_wave(backend):
    """gravity.c (density ratio 10, reduced gravity as a jump), 8 / 16 / 32 / 64 points per wavelength:
    0.1272 / 0.0287 / 0.00701 / 0.00483 (iforce.h 0.1267 / 0.0318 / 0.00809 / 0.00456)."""
    from cases import gravity_wave as gw

    assert gw.run(32, backend, pressure_jump=True)["error"] == pytest.approx(0.028697, rel=1e-3)

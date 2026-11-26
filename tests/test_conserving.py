"""Momentum-conserving VOF (Basilisk navier-stokes/conserving.h) and centered.h's wall pressure (docs/notes/71)."""


import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.linear.lexicographic import set_relaxation
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.conserving_compare import run


def test_wall_pressure_balances_gravity():
    """A fluid at rest under a uniform acceleration between walls stays at rest (centered.h's neumann_pressure)."""
    grid = CartesianGrid((16, 16))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10, DT=1e-2)
    ns.a[1] = -9.81 + 0.0 * ns.a[1]
    ns.start()
    for _ in range(20):
        ns.step()
    assert max(float(np.abs(c.interior).max()) for c in ns.u) < 1e-10


@pytest.mark.xdist_group("lexicographic")
def test_rayleigh_taylor_against_basilisk():
    """rt.c (conserving.h) on a uniform 64^2 grid, t = 0.1 (31 steps): Basilisk's C build gives max|u.y| 0.2258136,
    kinetic energy 0.0101207, sum f 2048.000527 (tools/basilisk/rt_conserving.c)."""
    set_relaxation("lexicographic")
    try:
        out, dts = run(True)
    finally:
        set_relaxation("redblack")
    s = out[0.1]
    rho = np.clip(s["f"], 0, 1) * (1.225 - 0.1694) + 0.1694
    ke = float(np.sum(rho * (s["ux"] ** 2 + s["uy"] ** 2)) * (4 / 64) ** 2 / 2)
    assert len(dts) == 92
    assert float(np.abs(s["uy"]).max()) == pytest.approx(0.2258136, abs=2e-5)
    assert ke == pytest.approx(0.0101207, rel=1e-4)
    assert float(s["f"].sum()) == pytest.approx(2048.000527, abs=1e-5)


def test_conserving_options():
    grid = CartesianGrid((8, 8))
    ns = CenteredNS(grid)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.5), BoundaryConditions())
    with pytest.raises(NotImplementedError):
        TwoPhase(ns, f, conserving=True, vof_first=True)
    TwoPhase(CenteredNS(grid), f, rho1=2.0, rho2=1.0, conserving=True)


def test_ccd_cmin_keeps_interfacial_cells_out():
    """``conserving_ccd_cmin`` (docs/notes/75): cells with a phase fraction below it keep vof.h's gradient; the rest of
    the run is differentiated by CCD (exact for a quadratic concentration with the extrapolation closures)."""
    from basilisk_cupy.interface.vof import _ccd_concentration_gradient

    n = 12
    x = np.arange(n, dtype=float)
    c = np.ones(n)
    c[:2] = 0.0
    c[2] = 0.7  # an interfacial cell at the start of the phase's run
    fq = 0.01 * x ** 2
    cp = np.pad(c[:, None], 1, mode="edge")
    tp = np.pad((c * fq)[:, None], 1, mode="edge")
    base = np.full((n, 1), -1.0)
    g50 = _ccd_concentration_gradient(np, cp, tp, 0, False, base, False)[:, 0]
    g99 = _ccd_concentration_gradient(np, cp, tp, 0, False, base, False, cmin=0.99)[:, 0]
    assert np.allclose(g50[2:], 0.02 * x[2:], atol=1e-10)
    assert g99[2] == -1.0
    assert np.allclose(g99[3:], 0.02 * x[3:], atol=1e-10)
    g1, g2 = _ccd_concentration_gradient(np, cp, tp, 0, False, base, False, True, 0.99)  # "ccd2": also f''
    assert np.allclose(g1[:, 0], g99) and np.allclose(g2[3:, 0], 0.02, atol=1e-10) and np.all(g2[:3, 0] == 0.0)
    grid = CartesianGrid((8, 8))
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.5), BoundaryConditions())
    tp2 = TwoPhase(CenteredNS(grid), f, rho1=2.0, rho2=1.0, conserving=True, conserving_gradient="ccd",
                   conserving_ccd_cmin=0.99)
    assert tp2.conserving_ccd_cmin == 0.99

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


def test_strip_moments():
    """``geometry.strip_moments`` (docs/notes/77): the area agrees with ``rectangle_fraction`` to round-off and the
    first moments with a fine sampling of the region."""
    from basilisk_cupy.interface.geometry import rectangle_fraction, strip_moments

    rng = np.random.default_rng(1)
    nx, ny = rng.normal(size=200), rng.normal(size=200)
    nrm = np.abs(nx) + np.abs(ny)
    nx, ny = nx / nrm, ny / nrm
    nx[:10], ny[10:20] = 0.0, 0.0
    al, C = rng.uniform(-0.6, 0.6, 200), rng.uniform(0.0, 1.0, 200)
    A, MX, MY = strip_moments(nx, ny, al, -0.5, C - 0.5)
    assert np.abs(A - rectangle_fraction(nx, ny, al, -0.5, -0.5, C - 0.5, 0.5) * C).max() < 1e-13
    for k in (5, 15, 60, 200 - 1):
        xs, ys = np.linspace(-0.5, C[k] - 0.5, 2001), np.linspace(-0.5, 0.5, 2001)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside = nx[k] * X + ny[k] * Y <= al[k]
        dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
        assert MX[k] == pytest.approx((inside * X).sum() * dA, abs=2e-4)
        assert MY[k] == pytest.approx((inside * Y).sum() * dA, abs=2e-4)


def test_geometric_flux_full_cells():
    """``VOFTracer(geometric=True)`` (docs/notes/77) only changes interfacial upwind cells: with no interface the
    concentration moves exactly as with vof.h's flux."""
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection

    n = 16
    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi = 2.0 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    out = []
    for geo in (False, True):
        c = CellField.from_interior(grid, np.ones((n, n)), bc)
        t = CellField.from_interior(grid, phi.copy(), bc)
        itf = VOFInterface(c, [VOFTracer(t, geometric=geo)])
        uf = FaceField(grid)
        uf.components[0][...] = 1.0
        uf.components[1][...] = -0.3
        for i in range(6):
            vof_advection([itf], uf, 0.4 / n, i)
        out.append(np.asarray(t.interior).copy())
    assert np.array_equal(out[0], out[1])


@pytest.mark.parametrize("conserving,c_value", [(False, 1.095545e-02), (True, 6.348679e-02)])
def test_taylor_green_against_basilisk(conserving, c_value):
    """Inviscid Taylor-Green on 32^2, t = 0.5 (docs/notes/77): max|u - u0| of the plain centred solver and of
    conserving.h with one phase everywhere, against the C build (tools/basilisk/tg_conserving.c). conserving.h's
    error is 6 times the plain solver's (first order in time)."""
    from cases.pure_advection import coupled, tg_cfl

    r = tg_cfl("centered-conserving", 32, 0.5) if conserving else coupled("single", 32)
    assert r["du_max"] == pytest.approx(c_value, rel=2e-2)


@pytest.mark.parametrize("complementary", [False, True])
def test_constant_concentration_at_half(complementary):
    """A constant concentration (t1 = c, t2 = 1 - c) moved by a divergence-free, non-uniform velocity: vof.h's
    dilation term takes c > 1/2 for t1 and c < 1/2 for t2, so a cell with c = 1/2 exactly dilates neither and t2/(1 - c)
    drifts by 1.7e-2 in one step (Basilisk's own behaviour); ``complementary=True`` (c <= 1/2 for t2) keeps it to
    round-off (docs/notes/77)."""
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection
    from cases.pure_advection import face_velocity

    n = 32
    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    c0 = np.zeros((n, n))
    c0[:, :12] = 1.0
    c0[:, 12] = 0.5  # a flat interface through the centres of row 12
    c = CellField.from_interior(grid, c0.copy(), bc)
    t1 = CellField.from_interior(grid, c0.copy(), bc)
    t2 = CellField.from_interior(grid, 1.0 - c0, bc)
    itf = VOFInterface(c, [VOFTracer(t1), VOFTracer(t2, inverse=True, complementary=complementary)])
    u, v = face_velocity("vortex", n, 0.0, 0.5)
    uf = FaceField(grid)
    uf.components[0][...] = u
    uf.components[1][...] = v
    vof_advection([itf], uf, 0.25 / n, 0)
    C, T2 = np.asarray(c.interior), np.asarray(t2.interior)
    err = np.abs(np.where(1.0 - C > 1e-12, T2 / np.maximum(1.0 - C, 1e-300), 1.0) - 1.0).max()
    if complementary:
        assert err < 1e-14
    else:
        assert err == pytest.approx(1.66e-2, rel=0.05)


@pytest.mark.parametrize("variant", ["centered-vof", "centered-conserving", "centered-conserving-ccd",
                                     "centered-conserving-geo"])
def test_free_stream(variant):
    """Free-stream preservation (docs/notes/77, P0-1): a drop at density ratio 1000 with no surface tension carried by
    a uniform oblique flow keeps the velocity to round-off."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from cases.convection_stability import conserving_options

    n = 16
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2), bc)
    TwoPhase(ns, f, rho1=1.0, rho2=1e-3, **conserving_options(variant))
    ns.u[0].interior = 1.0 + 0.0 * ns.u[0].interior
    ns.u[1].interior = 0.37 + 0.0 * ns.u[1].interior
    ns.start()
    for _ in range(20):
        ns.step()
    assert float(np.abs(to_numpy(ns.u[0].interior) - 1.0).max()) < 1e-11
    assert float(np.abs(to_numpy(ns.u[1].interior) - 0.37).max()) < 1e-11

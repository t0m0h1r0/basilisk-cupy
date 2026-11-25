"""Step 5: IMEX Runge--Kutta stepper (ARS(3,4,3) / ARS(4,4,3)): explicit convection with imaginary-axis stability."""

import math
import warnings

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.time.basilisk_centered import STAGES
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS
from basilisk_cupy.time.imex_rk import TABLEAUX, IMEXRKNS, stability_function

from .conftest import order


@pytest.mark.parametrize("name", ["ars343", "ars443"])  # 3rd-order tableaux (the others: test_two_phase_imex_rk)
def test_tableaux(name):
    t = TABLEAUX[name]()
    AE, AI, b, c = (np.array(v, float) for v in (t.AE, t.AI, t.bE, t.c))
    # explicit part: 3rd-order conditions; both parts share c; implicit part stiffly accurate SDIRK
    assert b.sum() == pytest.approx(1.0) and b @ c == pytest.approx(0.5)
    assert b @ c ** 2 == pytest.approx(1 / 3, abs=1e-9) and b @ AE @ c == pytest.approx(1 / 6, abs=1e-9)
    assert np.allclose(AE.sum(axis=1), c, atol=1e-9) and np.allclose(AI.sum(axis=1), c, atol=1e-9)
    assert np.allclose(np.diag(AI)[1:], t.gamma)
    assert t.explicit_stiffly_accurate == (name == "ars443")  # ARS(3,4,3) needs the final projection
    bI = AI[-1]  # implicit weights: the last implicit row (stiffly accurate)
    assert bI.sum() == pytest.approx(1.0) and bI @ c == pytest.approx(0.5)
    # the axis limits stored on the tableau are those of its stability function
    ys = np.linspace(1e-3, 4.0, 4000)
    imag = next(y for y in ys if abs(stability_function(t, 1j * y)) > 1 + 1e-12)
    xs = np.linspace(-4.0, 0.0, 4001)
    real = min(x for x in xs if abs(stability_function(t, x)) <= 1 + 1e-12)
    assert imag == pytest.approx(t.imag_limit, abs=2e-3) and -real == pytest.approx(t.real_limit, abs=2e-3)


def _taylor_green(cls, n, dt, t_end, *, backend, mu=0.01, stokes=False, conv="central", face="fccd", noise=0.0, **kw):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": face, "cell": "ccd" if face == "fccd" else "fd2"},
                           "terms": {"convection": conv, "viscosity": "ccd" if face == "fccd" else "basilisk"}})
    ns = cls(grid, periodic=(0, 1), mu=mu, ops=scheme.face, convection=scheme.convection, viscous=scheme.viscous,
             fixed_dt=dt, tolerance=1e-11, stokes=stokes, **kw)
    rng = np.random.default_rng(0)
    ex = [-xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y), xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)]
    for a in range(2):
        ns.u[a].interior = ex[a] + noise * xp.asarray(rng.standard_normal(grid.n))
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(t_end)
    decay = math.exp(-8 * math.pi ** 2 * mu * t_end)
    return ns, max(float(xp.max(xp.abs(ns.u[a].interior - decay * ex[a]))) for a in range(2))


@pytest.mark.parametrize("tableau", ["ars343", "ars443"])
def test_stokes_third_order_in_time(backend, tableau):
    ks, t_end = [5, 10, 20], 0.25
    ref, _ = _taylor_green(IMEXRKNS, 16, t_end / 160, t_end, backend=backend, stokes=True, tableau=tableau)
    errs = []
    for k in ks:
        ns, _ = _taylor_green(IMEXRKNS, 16, t_end / k, t_end, backend=backend, stokes=True, tableau=tableau)
        errs.append(max(float(abs(to_numpy(ns.u[a].interior - ref.u[a].interior)).max()) for a in range(2)))
    assert min(order(errs, ks)) > 2.9


def test_inviscid_stability_beyond_ext2(backend):
    """Inviscid Taylor--Green with noise, FCCD central flux, N = 32, t = 1: EXT2 blows up at CFL 0.7
    (measured: 1.1e-4 at 0.3, 1.6e-2 at 0.5), ARS(3,4,3) stays accurate at CFL 1.5."""
    _, rk = _taylor_green(IMEXRKNS, 32, 1.5 / 32, 1.0, backend=backend, mu=0.0, noise=1e-6)
    _, bdf2 = _taylor_green(IMEXBDF2NS, 32, 0.7 / 32, 1.0, backend=backend, mu=0.0, noise=1e-6)
    assert rk < 1e-3
    assert not bdf2 < 0.5


def test_adaptive_step_uses_the_spectral_bound(backend):
    ns, _ = _taylor_green(IMEXRKNS, 16, None, 0.05, backend=backend, conv="uccd6", cfl=10.0)
    imag, real = ns.convection.spectrum(ns.u, ns.uf)
    bound = ns.safety * min(ns.tableau.imag_limit / imag, ns.tableau.real_limit / real)
    assert ns.clock.dt <= bound * 1.05 and ns.clock.dt > 0.5 * bound


@pytest.mark.parametrize("tableau", ["ars343", "ars443"])
def test_hooks_run_in_basilisk_order(tableau):
    ns = IMEXRKNS(CartesianGrid((8, 8)), periodic=(0, 1), fixed_dt=0.01, tableau=tableau)
    seen = []
    for stage in STAGES:
        ns.on(stage, lambda s, stage=stage: seen.append(stage))
    ns.start()
    seen.clear()
    ns.step()
    assert [s for i, s in enumerate(seen) if s not in seen[:i]] == list(STAGES)


def test_lid_cavity_with_walls(backend):
    from basilisk_cupy import Dirichlet

    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: Dirichlet(0.0) for w in walls}
    tang["top"] = Dirichlet(1.0)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "central", "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                  convection=scheme.convection, viscous=scheme.viscous, fixed_dt=1.0 / 64)  # CFL 1
    ns.start(events=[1.1])
    ns.run(1.1)
    # fine-grid value E(1.1) ~ 0.00969 (fd2 IMEX-BDF2, 256^2); FCCD + CCD viscosity + BDF2 at CFL 0.25: 0.009211
    # (4th-order D_f; N = 128 / 256: 0.009651 / 0.009688)
    assert ns.kinetic_energy() == pytest.approx(0.009326, rel=0.01)
    assert ns.max_divergence() < 2e-5  # measured 1.16e-5 (cell state)

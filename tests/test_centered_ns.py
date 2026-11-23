"""Step 3: Basilisk-compatible centred Navier--Stokes, validated against Basilisk's own reference outputs."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import vector_bcs
from basilisk_cupy.fields import VectorField
from basilisk_cupy.linear.viscosity import viscosity
from basilisk_cupy.terms.convection.bcg import advection, minmod
from basilisk_cupy.time.basilisk_centered import STAGES, CenteredNS
from basilisk_cupy.time.stepping import Clock, dtnext, timestep
from cases import lid_cavity, taylor_green

from .conftest import order


# -- time-step control ------------------------------------------------------------
def test_dtnext_hits_events_exactly():
    clock = Clock(events=[0.1, 0.2])
    t = 0.0
    hit = []
    while t < 0.2 - 1e-12:
        clock.t = t
        dt = dtnext(0.03, clock)
        t = clock.tnext
        hit.append(t)
        assert dt <= 0.03 * (1 + 1e-9)
    assert any(abs(h - 0.1) < 1e-14 for h in hit) and abs(hit[-1] - 0.2) < 1e-14


def test_timestep_smooths_increases():
    grid = CartesianGrid((16, 16))
    uf = FaceField.constant(grid, 1.0)
    clock = Clock()
    first = timestep(uf, 1e30, 0.8, clock)  # from previous = 0
    assert first == pytest.approx(0.8 / 16 / 11)
    clock.t = 1.0
    second = timestep(uf, 1e30, 0.8, clock)
    assert first < second < 0.8 / 16


# -- BCG advection -----------------------------------------------------------------
def test_bcg_conserves_and_converges(backend):
    """Gaussian advected one period along the diagonal: exact conservation, second order."""
    from basilisk_cupy import CellField
    errs, ns_ = [], [32, 64, 128]
    for n in ns_:
        grid = CartesianGrid((n, n), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        f0 = xp.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)
        f = CellField.from_interior(grid, f0, BoundaryConditions.periodic(2))
        uf = FaceField.constant(grid, 1.0)
        for _ in range(2 * n):
            advection([f], uf, 0.5 / n)
        assert float(xp.sum(f.interior)) == pytest.approx(float(xp.sum(f0)), rel=1e-12)
        errs.append(float(xp.max(xp.abs(f.interior - f0))))
    assert min(order(errs, ns_)) > 1.8


def test_limited_gradients_are_bounded(backend):
    from basilisk_cupy import CellField
    grid = CartesianGrid((32, 32), backend=backend)
    xp = grid.xp
    x, _ = grid.meshgrid()
    f = CellField.from_interior(grid, xp.where(x < 0.5, 1.0, 0.0), BoundaryConditions.periodic(2))
    uf = FaceField.constant(grid, 1.0)
    for _ in range(20):
        advection([f], uf, 0.4 / 32, limiter=minmod)
    assert float(f.interior.max()) <= 1.0 + 1e-12 and float(f.interior.min()) >= -1e-12


# -- implicit viscosity ------------------------------------------------------------------
def test_viscosity_matches_discrete_decay(backend):
    """u.x = sin(2 pi y), u.y = 0: one backward-Euler step decays by 1/(1 + dt mu lambda_h)."""
    n, mu, dt = 32, 0.1, 0.05
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    _, y = grid.meshgrid()
    u = VectorField.zeros(grid, vector_bcs(2, (0, 1)))
    u[0].interior = xp.sin(2 * math.pi * y)
    stats = viscosity(u, FaceField.constant(grid, mu), 1.0, dt, tolerance=1e-10)
    lam = 4 * math.sin(math.pi / n) ** 2 / grid.h[1] ** 2
    expected = xp.sin(2 * math.pi * y) / (1 + dt * mu * lam)
    assert stats.converged
    np.testing.assert_allclose(to_numpy(u[0].interior), to_numpy(expected), atol=1e-9)
    assert float(xp.max(xp.abs(u[1].interior))) < 1e-9


# -- solver plumbing ------------------------------------------------------------------
def test_hooks_run_in_basilisk_order():
    ns = CenteredNS(CartesianGrid((8, 8)), periodic=(0, 1))
    seen = []
    for stage in STAGES:
        ns.on(stage, lambda s, stage=stage: seen.append(stage))
    ns.on("projection", lambda s: seen.append("projection-late"))  # registered later -> runs first
    ns.start()
    seen.clear()
    ns.step()
    assert seen[:9] == list(STAGES[:9])
    assert seen[9:] == ["projection-late", "projection", "end_timestep"]


# -- Basilisk reference cases --------------------------------------------------------------
def test_taylor_green_matches_basilisk(backend):
    """src/test/taylor-green.c at t = 2 against taylor-green.ref (N = 32, 64)."""
    ns_, avg, rms = [32, 64], [], []
    for n in ns_:
        a, r, _ = taylor_green.run(n, backend)
        ra, rr, _ = taylor_green.REFERENCE[n]
        assert a == pytest.approx(ra, rel=0.01)
        assert r == pytest.approx(rr, rel=0.02)
        avg.append(a)
        rms.append(r)
    assert min(order(avg, ns_)) > 2.7 and min(order(rms, ns_)) > 2.7


def test_lid_cavity_matches_basilisk_log(backend):
    """src/test/lid.c, N = 64, Re = 1000: energy log up to t = 1.1 against lid.gpu.ref."""
    _, rows = lid_cavity.run(64, 1.1, backend, log=None)
    assert len(rows) == len(lid_cavity.REFERENCE)
    for (t, e, du), (rt, re_, rdu) in zip(rows, lid_cavity.REFERENCE):
        assert t == pytest.approx(rt, abs=1e-12)
        assert e == pytest.approx(re_, rel=1e-3)
        assert du == pytest.approx(rdu, rel=0.02)


@pytest.mark.slow
def test_lid_cavity_steady_state_and_ghia():
    """Full run: stops at t = 44.0 like lid.ref; centreline profiles within 0.02 of Ghia et al."""
    ns, rows = lid_cavity.run(64, 60.0, log=None)
    t_ref, e_ref = lid_cavity.REFERENCE_STEADY
    assert rows[-1][0] == pytest.approx(t_ref, abs=1e-9)
    assert rows[-1][1] == pytest.approx(e_ref, rel=1e-3)
    xprof, yprof = lid_cavity.profiles(ns)
    assert max(abs(a - b) for _, a, b in xprof + yprof) < 0.02

"""The ghost cells of the conserving momentum transport (docs/notes/112 D, issue #146, audit #149).

conserving.h builds the ghosts of the phase momenta q = c rho u from the step-start velocity u (``boundary_q1_x``).
After the first sweep of the split VOF advection q has changed and u has not: the second sweep's ghosts are stale,
at periodic boundaries and at walls alike, and the transport is first order in time. ``TwoPhase`` now takes them
from the momentum's current velocity (q1 + q2) / rho(f); ``conserving_stale_ghosts=True`` keeps conserving.h's.

Transport only (``TwoPhase._vof`` called directly, exact face velocities): no pressure solve."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase, _MomentumField
from basilisk_cupy.time.basilisk_centered import CenteredNS
from basilisk_cupy.time.stepping import Clock

TWO_PI = 2 * math.pi
BOUNDARIES = {"periodic": (0, 1), "walls": (), "periodic-x": (0,), "periodic-y": (1,)}


def _fbc(periodic):
    bc = BoundaryConditions()
    for ax in periodic:
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    return bc


def _setup(n, periodic, f0=None, ratio=1.0, stale=False, source=False):
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=periodic)
    f = CellField.from_interior(grid, np.zeros((n, n)) if f0 is None else f0, _fbc(periodic))
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, conserving=True, conserving_source=source,
                  conserving_stale_ghosts=stale)
    return grid, ns, tp


def _stream_faces(n, psi):
    """Face velocities from a stream function at the vertices: discretely divergence-free."""
    h = 1.0 / n
    xv = np.arange(n + 1) * h
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    P = psi(X, Y)
    return (P[:, 1:] - P[:, :-1]) / h, -(P[1:, :] - P[:-1, :]) / h


def _cells(n, fun):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return fun(X, Y)


def _vof(ns, tp, dt, i, ufx, ufy):
    xp = ns.grid.xp
    ns.uf.components[0][...] = xp.asarray(ufx)
    ns.uf.components[1][...] = xp.asarray(ufy)
    ns.clock = Clock(dt=dt, i=i)
    tp._vof(ns)


# ------------------------------------------------------------------------------- 1, 3: fully periodic, both sweep orders
def _periodic_step(n, shift, i, stale):
    """One transport step of a two-phase state (a drop, density ratio 1000) in a periodic box, the data shifted by
    ``shift`` cells; the result shifted back. A periodic scheme does not see the shift."""
    psi = lambda x, y: (np.sin(TWO_PI * x) * np.sin(TWO_PI * y) + 0.3 * np.cos(TWO_PI * (x + 2 * y))) / TWO_PI  # noqa: E731
    sx, sy = shift[0] / n, shift[1] / n
    grid0 = CartesianGrid((n, n))
    f0 = to_numpy(fraction(grid0, lambda x, y: 0.2 ** 2 - (((x + sx) % 1.0) - 0.3) ** 2 - (((y + sy) % 1.0) - 0.4) ** 2,
                           refine=8))
    grid, ns, tp = _setup(n, (0, 1), f0=f0, ratio=1000.0, stale=stale)
    ufx, ufy = _stream_faces(n, lambda x, y: psi(x + sx, y + sy))
    ns.u[0].interior = grid.xp.asarray(_cells(n, lambda x, y: np.sin(TWO_PI * (x + sx)) * np.cos(TWO_PI * (y + sy)) + 0.5))
    ns.u[1].interior = grid.xp.asarray(_cells(n, lambda x, y: -np.cos(TWO_PI * (x + sx)) * np.sin(TWO_PI * (y + sy)) + 0.2))
    ns.u.fill_ghosts()
    rho0 = to_numpy(tp.rho(tp.f.interior))
    mom0 = [float((rho0 * to_numpy(c.interior)).sum()) for c in ns.u.components]
    _vof(ns, tp, 0.4 / n / 1.6, i, ufx, ufy)
    rho1 = to_numpy(tp.rho(tp.f.interior))
    mom1 = [float((rho1 * to_numpy(c.interior)).sum()) for c in ns.u.components]
    back = lambda a: np.roll(to_numpy(a), shift, axis=(0, 1))  # noqa: E731
    return [back(c.interior) for c in ns.u.components], back(tp.f.interior), mom0, mom1


@pytest.mark.parametrize("i", [0, 1])                             # i = 0: x then y; i = 1: y then x
def test_periodic_transport_does_not_see_where_the_boundary_is(i):
    n = 32
    u_a, f_a, _, _ = _periodic_step(n, (0, 0), i, stale=False)
    u_b, f_b, _, _ = _periodic_step(n, (5, 11), i, stale=False)
    assert np.abs(f_a - f_b).max() < 1e-12
    assert max(np.abs(a - b).max() for a, b in zip(u_a, u_b)) < 1e-12   # (1e-2 with the mixture's velocity in the
    #                                    ghosts of both phases: after a sweep the phases of a cell differ in velocity)
    # conserving.h's ghosts: the result depends on where the periodic boundary cuts the flow (recorded, not fixed)
    s_a, _, _, _ = _periodic_step(n, (0, 0), i, stale=True)
    s_b, _, _, _ = _periodic_step(n, (5, 11), i, stale=True)
    assert max(np.abs(a - b).max() for a, b in zip(s_a, s_b)) > 1e-5


# -------------------------------------------------- 4, 5: the two sides of a periodic face use one flux (conservation)
# (the conservation does not tell the two ghost definitions apart; the shift test above does)
@pytest.mark.parametrize("i", [0, 1])
def test_periodic_transport_conserves_the_momentum_of_a_two_phase_state(i):
    _, _, m0, m1 = _periodic_step(32, (0, 0), i, stale=False)
    assert max(abs(a - b) for a, b in zip(m0, m1)) < 1e-11 * 32 ** 2
    _, _, s0, s1 = _periodic_step(32, (0, 0), i, stale=True)     # also with conserving.h's ghosts: the flux of a
    assert max(abs(a - b) for a, b in zip(s0, s1)) < 1e-11 * 32 ** 2   # periodic face is one array entry for both sides


@pytest.mark.parametrize("name", list(BOUNDARIES))
def test_two_phase_free_stream_is_kept_with_every_boundary(name):
    """A drop of density ratio 1000 carried by a uniform flow along the periodic direction(s): u stays uniform."""
    n, periodic = 32, BOUNDARIES[name]
    U = (1.0 if 0 in periodic else 0.0, 0.37 if 1 in periodic else 0.0)
    grid0 = CartesianGrid((n, n))
    f0 = to_numpy(fraction(grid0, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
    grid, ns, tp = _setup(n, periodic, f0=f0, ratio=1000.0)
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
    ns.u.fill_ghosts()
    ufx, ufy = np.full((n + 1, n), U[0]), np.full((n, n + 1), U[1])
    for i in range(6):
        _vof(ns, tp, 0.3 / n, i, ufx, ufy)
    assert max(float(np.abs(to_numpy(ns.u[ax].interior) - U[ax]).max()) for ax in range(2)) < 1e-12


# --------------------------------------------- 2: periodic and walls mixed; the ghosts at the start of a step are the same
@pytest.mark.parametrize("name", list(BOUNDARIES))
def test_first_sweep_ghosts_are_those_of_conserving_h(name):
    """Before the first sweep the momentum's current velocity is the step-start u: the ghosts of both definitions
    agree at every boundary (to round-off). Only the second sweep differs."""
    n, periodic = 16, BOUNDARIES[name]
    grid0 = CartesianGrid((n, n))
    f0 = to_numpy(fraction(grid0, lambda x, y: 0.3 ** 2 - (x - 0.5) ** 2 - (y - 0.2) ** 2, refine=8))   # touches y = 0
    grid, ns, tp = _setup(n, periodic, f0=f0, ratio=1000.0)
    xp = grid.xp
    ns.u[0].interior = xp.asarray(_cells(n, lambda x, y: np.sin(TWO_PI * x) * np.cos(math.pi * y) + 0.3))
    ns.u[1].interior = xp.asarray(_cells(n, lambda x, y: np.cos(TWO_PI * x) * np.sin(math.pi * y)))
    fc = xp.minimum(xp.maximum(tp.f.interior, 0.0), 1.0)
    for u in ns.u.components:
        old = _MomentumField(grid, bc=u.bc, name="q1").bind(tp.f, u, tp.rho1, False)
        q1 = _MomentumField(grid, bc=u.bc, name="q1").bind(tp.f, u, tp.rho1, False)
        q2 = _MomentumField(grid, bc=u.bc, name="q2").bind(tp.f, u, tp.rho2, True)
        for q, c, rho in ((old, fc, tp.rho1), (q1, fc, tp.rho1), (q2, 1.0 - fc, tp.rho2)):
            q.interior = c * rho * u.interior
        scratch = CellField(grid, bc=u.bc)
        q1.current(q1, q2, tp.rho, scratch)
        old.fill_ghosts()
        q1.fill_ghosts()
        assert float(xp.abs(old.data - q1.data).max()) < 1e-13


# ------------------------------------------------------------------- audit #149: walls give the same first order in time
def _taylor_green_transport(n, periodic, stale, cfl=0.4, t_end=0.25):
    """The steady Taylor-Green flow (also a solution between slip walls), exact face velocity and exact pressure
    acceleration g (in the face value and after the transport): max |u - u0| at t_end (docs/notes/112, section 3)."""
    grid, ns, tp = _setup(n, periodic, stale=stale, source=True)
    xp = grid.xp
    ufx, ufy = _stream_faces(n, lambda x, y: np.sin(TWO_PI * x) * np.sin(TWO_PI * y) / TWO_PI)
    u0 = [_cells(n, lambda x, y: np.sin(TWO_PI * x) * np.cos(TWO_PI * y)),
          _cells(n, lambda x, y: -np.cos(TWO_PI * x) * np.sin(TWO_PI * y))]
    g = [_cells(n, lambda x, y: math.pi * np.sin(2 * TWO_PI * x)), _cells(n, lambda x, y: math.pi * np.sin(2 * TWO_PI * y))]
    for ax in range(2):
        ns.u[ax].interior = xp.asarray(u0[ax])
    ns.u.fill_ghosts()
    steps = int(math.ceil(t_end / (cfl / n) - 1e-9))
    dt = t_end / steps
    for i in range(steps):
        for ax in range(2):
            ns.g[ax].interior = xp.asarray(g[ax])
        ns.g.fill_ghosts()
        _vof(ns, tp, dt, i, ufx, ufy)
        for ax in range(2):
            ns.u[ax].interior = ns.u[ax].interior + dt * xp.asarray(g[ax])
        ns.u.fill_ghosts()
    return max(float(np.abs(to_numpy(ns.u[ax].interior) - u0[ax]).max()) for ax in range(2))


@pytest.mark.parametrize("name", list(BOUNDARIES))
def test_transport_is_second_order_with_every_boundary_and_first_order_with_stale_ghosts(name):
    periodic = BOUNDARIES[name]
    new = [_taylor_green_transport(n, periodic, stale=False) for n in (32, 64)]
    old = [_taylor_green_transport(n, periodic, stale=True) for n in (32, 64)]
    assert new[0] / new[1] > 3.5                                  # second order: periodic, walls and mixed
    assert old[0] / old[1] < 2.7 and old[1] > 2.0 * new[1]        # conserving.h's ghosts: first order, walls too

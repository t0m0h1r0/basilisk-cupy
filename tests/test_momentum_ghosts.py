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


# ------------------------------------------- B's requests on PR #154: no-slip walls, slivers, an interface on the wall
def _noslip_setup(n, stale):
    """Periodic in x, NO-SLIP walls in y (u.t = dirichlet(0) on both)."""
    from basilisk_cupy.boundary import Dirichlet

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0,), tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)})
    f = CellField.from_interior(grid, np.zeros((n, n)), _fbc((0,)))
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0, conserving=True, conserving_source=True, conserving_stale_ghosts=stale)
    return grid, ns, tp


def _noslip_transport(n, stale, cfl=0.4, t_end=0.25):
    """A steady manufactured state between no-slip walls: psi = sin(2 pi x) sin^2(pi y) / (2 pi), so that
    u = sin(2 pi x) sin(2 pi y) / 2 and v = -cos(2 pi x) sin^2(pi y) both vanish on the walls; the source is
    g = (u . grad) u (in the face value and after the transport), with which u is a steady solution of
    u_t + div(u u) = g. Exact face velocities; max |u - u0| at t_end."""
    grid, ns, tp = _noslip_setup(n, stale)
    xp = grid.xp
    ufx, ufy = _stream_faces(n, lambda x, y: np.sin(TWO_PI * x) * np.sin(math.pi * y) ** 2 / TWO_PI)
    U = lambda x, y: 0.5 * np.sin(TWO_PI * x) * np.sin(TWO_PI * y)  # noqa: E731
    V = lambda x, y: -np.cos(TWO_PI * x) * np.sin(math.pi * y) ** 2  # noqa: E731
    Ux = lambda x, y: math.pi * np.cos(TWO_PI * x) * np.sin(TWO_PI * y)  # noqa: E731
    Uy = lambda x, y: math.pi * np.sin(TWO_PI * x) * np.cos(TWO_PI * y)  # noqa: E731
    Vx = lambda x, y: TWO_PI * np.sin(TWO_PI * x) * np.sin(math.pi * y) ** 2  # noqa: E731
    Vy = lambda x, y: -math.pi * np.cos(TWO_PI * x) * np.sin(TWO_PI * y)  # noqa: E731
    u0 = [_cells(n, U), _cells(n, V)]
    g = [_cells(n, lambda x, y: U(x, y) * Ux(x, y) + V(x, y) * Uy(x, y)),
         _cells(n, lambda x, y: U(x, y) * Vx(x, y) + V(x, y) * Vy(x, y))]
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


def test_no_slip_walls_second_order_and_first_order_with_stale_ghosts():
    new = [_noslip_transport(n, stale=False) for n in (32, 64)]
    old = [_noslip_transport(n, stale=True) for n in (32, 64)]
    assert new[0] / new[1] > 3.5
    assert old[0] / old[1] < 3.0 and old[1] > 2.0 * new[1]     # 4.24e-3 / 1.56e-3 against 2.69e-3 / 6.56e-4


def _stripe(n, eps, k):
    """Phase 1 fills a band of columns that ends with the volume fraction eps (a sliver) in column k; k = n - 1 is
    the last column before the periodic boundary, a smaller k is the same state shifted into the interior."""
    f = np.zeros((n, n))
    f[n // 4: n - 1, :] = 1.0
    f[n - 1, :] = eps
    return np.roll(f, k - (n - 1), axis=0)


@pytest.mark.parametrize("eps", [1e-12, 1e-10, 2e-10, 1e-8, 1e-4])
def test_slivers_at_the_boundary_free_stream_conservation_and_shift(eps):
    """A cell with a tiny volume of the heavy phase (below, at and above the threshold c > 1e-10 of the phase's own
    velocity) in the last column before the periodic boundary, density ratio 1000."""
    n = 32
    psi = lambda x, y: y * 0.3 + x * 0.0 + np.sin(TWO_PI * x) * np.sin(TWO_PI * y) / TWO_PI  # noqa: E731
    res = {}
    for k in (n - 1, n - 9):                                     # the sliver at the boundary, and 8 cells inside
        sx = (n - 1 - k) / n
        # free stream: six steps of the uniform flow (1, 0.37)
        grid, ns, tp = _setup(n, (0, 1), f0=_stripe(n, eps, k), ratio=1000.0)
        for ax, U in enumerate((1.0, 0.37)):
            ns.u[ax].interior = U + 0.0 * ns.u[ax].interior
        ns.u.fill_ghosts()
        for i in range(6):
            _vof(ns, tp, 0.3 / n, i, np.full((n + 1, n), 1.0), np.full((n, n + 1), 0.37))
        assert max(float(np.abs(to_numpy(ns.u[ax].interior) - U).max()) for ax, U in enumerate((1.0, 0.37))) < 1e-11
        # a non-uniform flow: the momentum is conserved, and the result does not depend on where the boundary is
        grid, ns, tp = _setup(n, (0, 1), f0=_stripe(n, eps, k), ratio=1000.0)
        ufx, ufy = _stream_faces(n, lambda x, y: psi(x + sx, y))
        ns.u[0].interior = grid.xp.asarray(_cells(n, lambda x, y: 0.3 + np.sin(TWO_PI * (x + sx)) * np.cos(TWO_PI * y)))
        ns.u[1].interior = grid.xp.asarray(_cells(n, lambda x, y: -np.cos(TWO_PI * (x + sx)) * np.sin(TWO_PI * y)))
        ns.u.fill_ghosts()
        m0 = [float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(c.interior)).sum()) for c in ns.u.components]
        for i in range(2):
            _vof(ns, tp, 0.25 / n, i, ufx, ufy)
        m1 = [float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(c.interior)).sum()) for c in ns.u.components]
        assert max(abs(a - b) for a, b in zip(m0, m1)) < 1e-11 * n ** 2
        res[k] = [np.roll(to_numpy(c.interior), n - 1 - k, axis=0) for c in ns.u.components]
    assert max(np.abs(a - b).max() for a, b in zip(res[n - 1], res[n - 9])) < 1e-11


@pytest.mark.parametrize("walls", ["slip", "no-slip"])
def test_interface_on_a_wall_free_stream_and_momentum_through_both_sweeps(walls):
    """A drop cut by the bottom wall (periodic in x), density ratio 1000: the uniform flow along the wall stays
    uniform with slip walls, and the x-momentum of a non-uniform flow tangent to the walls is conserved by the
    transport (no flux through the walls), over six steps with both sweep orders."""
    from basilisk_cupy.boundary import Dirichlet

    n = 32
    grid0 = CartesianGrid((n, n))
    f0 = to_numpy(fraction(grid0, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - y ** 2, refine=8))

    def make():
        grid = CartesianGrid((n, n))
        kw = {} if walls == "slip" else {"tangential": {"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}}
        ns = CenteredNS(grid, periodic=(0,), **kw)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, _fbc((0,))), rho1=1.0, rho2=1e-3, conserving=True)
        return grid, ns, tp

    if walls == "slip":
        grid, ns, tp = make()
        ns.u[0].interior = 1.0 + 0.0 * ns.u[0].interior
        ns.u.fill_ghosts()
        for i in range(6):
            _vof(ns, tp, 0.3 / n, i, np.full((n + 1, n), 1.0), np.zeros((n, n + 1)))
        assert float(np.abs(to_numpy(ns.u[0].interior) - 1.0).max()) < 1e-12
        assert float(np.abs(to_numpy(ns.u[1].interior)).max()) < 1e-12
    grid, ns, tp = make()
    amp = 1.0 if walls == "slip" else 0.0                        # no-slip: the flow itself vanishes on the walls
    psi = lambda x, y: amp * y + 0.2 * np.sin(TWO_PI * x) * (np.sin(math.pi * y) / math.pi if walls == "slip"  # noqa: E731
                                                              else np.sin(math.pi * y) ** 2 / TWO_PI)
    ufx, ufy = _stream_faces(n, psi)
    ns.u[0].interior = grid.xp.asarray(0.5 * (ufx[1:, :] + ufx[:-1, :]))
    ns.u[1].interior = grid.xp.asarray(0.5 * (ufy[:, 1:] + ufy[:, :-1]))
    ns.u.fill_ghosts()
    m0 = float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[0].interior)).sum())
    for i in range(6):
        _vof(ns, tp, 0.3 / n, i, ufx, ufy)
    m1 = float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[0].interior)).sum())
    assert abs(m1 - m0) < 1e-11 * n ** 2
    assert np.isfinite(to_numpy(ns.u[0].interior)).all() and float(np.abs(to_numpy(ns.u[0].interior)).max()) < 3.0


# ------------------------------------------------- audit #156: a cell with c exactly 1/2 and the free stream
@pytest.mark.parametrize("rho", [(1e-3, 1.0), (1.0, 1.0), (1.0, 1e-3)])
@pytest.mark.parametrize("frac", [0.5, 0.5 + 1e-9, 0.3])
def test_free_stream_with_a_cell_at_exactly_one_half(rho, frac):
    """A band parallel to the grid whose interfaces cut the cells at ``frac`` (0.5: c = 1/2 exactly), a uniform
    momentum carried by a non-uniform divergence-free face velocity, ten steps. With vof.h's dilation rule neither
    phase dilates in the c = 1/2 cells and the uniform velocity is lost (the auditor's table: 1.04e-2 / 2.40e-3 /
    1.19e-5 at N = 32 for the three density pairs); the default keeps it to round-off."""
    n, U = 32, (1.0, 0.37)

    def run(vof_half):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1))
        i = np.arange(n)
        a, b = n // 4 + frac, 3 * n // 4 + frac
        col = np.clip(np.minimum(i + 1, b) - np.maximum(i, a), 0.0, 1.0)
        f = CellField.from_interior(grid, np.repeat(col[:, None], n, axis=1), _fbc((0, 1)))
        tp = TwoPhase(ns, f, rho1=rho[0], rho2=rho[1], conserving=True, conserving_vof_half=vof_half)
        ufx, ufy = _stream_faces(n, lambda x, y: np.sin(TWO_PI * x) * np.sin(TWO_PI * y) / TWO_PI
                                 + np.cos(TWO_PI * x) * np.sin(TWO_PI * y) / (2 * TWO_PI))
        for ax in range(2):
            ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
        ns.u.fill_ghosts()
        dt = 0.4 / n / max(np.abs(ufx).max(), np.abs(ufy).max())
        worst = 0.0
        for k in range(10):
            _vof(ns, tp, dt, k, ufx, ufy)
            worst = max(worst, max(float(np.abs(to_numpy(ns.u[ax].interior) - U[ax]).max()) for ax in range(2)))
        return worst

    assert run(False) < 1e-11                                    # round-off (ten steps, density ratio 1000)
    old = run(True)
    if frac == 0.5:
        assert old > 5e-6                                        # vof.h's rule: recorded, not fixed
    else:
        assert old < 1e-11

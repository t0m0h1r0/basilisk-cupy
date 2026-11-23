"""Step 4c: IMEX-BDF2 with incremental pressure correction (TwoPhaseFlow chapter 11, single-phase path)."""

import math

import numpy as np
import pytest

from basilisk_cupy.backend import to_numpy
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField, Neumann
from basilisk_cupy.discretization import FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.projection.approximate import project
from basilisk_cupy.time.basilisk_centered import STAGES, CenteredNS
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS

from .conftest import order


def taylor_green(cls, n, dt, t_end, *, mu=0.01, stokes=False, ops=None, backend="numpy"):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    kw = dict(periodic=(0, 1), mu=mu, stokes=stokes, ops=ops, tolerance=1e-10)
    ns = cls(grid, fixed_dt=dt, **kw) if cls is IMEXBDF2NS else cls(grid, DT=dt, cfl=100.0, **kw)
    ns.u[0].interior = -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y)
    ns.u[1].interior = xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    ns.run(t_end)
    return ns


def self_convergence(cls, n, t_end, ks, kref, **kw):
    ref = taylor_green(cls, n, t_end / kref, t_end, **kw)
    errs = []
    for k in ks:
        ns = taylor_green(cls, n, t_end / k, t_end, **kw)
        errs.append(max(float(abs(ns.u[a].interior - ref.u[a].interior).max()) for a in range(2)))
    return errs


@pytest.mark.parametrize("ops", [FD2FaceComplex, FCCDFaceComplex])
def test_incremental_and_full_pressure_projection_coincide(backend, ops):
    """Why IPC alone changes nothing for constant density: Basilisk starts from p^n."""
    ops = ops()
    grid = CartesianGrid((32, 32), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    xf, yf = grid.face_meshgrid(0)
    xg, yg = grid.face_meshgrid(1)
    uf = FaceField(grid, [xp.sin(3 * xf) * xp.cos(2 * yf), xp.cos(xg * yg)])
    uf[0][0] = uf[0][-1] = 0.0
    uf[1][:, 0] = uf[1][:, -1] = 0.0
    bc = BoundaryConditions.uniform(2, Neumann(0.0))
    pn = xp.cos(2 * math.pi * x) * xp.cos(math.pi * y)
    dt = 0.01
    p = CellField.from_interior(grid, pn.copy(), bc)
    full = uf.copy()
    project(full, p, None, dt, ops=ops)
    pnf = CellField.from_interior(grid, pn.copy(), bc)
    pnf.fill_ghosts()
    inc = FaceField(grid, [uf[a] - dt * ops.grad_f(pnf, a) for a in range(2)])
    dp = CellField(grid, bc=bc)
    project(inc, dp, None, dt, ops=ops)
    for a in range(2):
        np.testing.assert_allclose(to_numpy(full[a]), to_numpy(inc[a]), atol=1e-12)
    d = p.interior - (pn + dp.interior)
    assert float(abs(d - d.mean()).max()) < 1e-12


def test_bdf2_viscous_part_is_second_order(backend):
    """Unsteady Stokes: BDF2 + IPC is 2nd order in time, Basilisk's backward-Euler viscosity 1st."""
    ks = [10, 20, 40]
    bdf2 = self_convergence(IMEXBDF2NS, 16, 0.5, ks, 320, stokes=True, backend=backend)
    assert min(order(bdf2, ks)) > 1.95
    basilisk = self_convergence(CenteredNS, 16, 0.5, ks, 320, stokes=True, backend=backend)
    assert max(order(basilisk, ks)) < 1.2
    assert bdf2[-1] < basilisk[-1] / 20


def test_bdf2_navier_stokes_second_order_with_fccd(backend):
    """With FCCD the O(dt h^4) projection coupling is small enough to see 2nd order (fd2: O(dt h^2))."""
    ks = [10, 20, 40]
    errs = self_convergence(IMEXBDF2NS, 32, 0.25, ks, 320, ops=FCCDFaceComplex(), backend=backend)
    assert min(order(errs, ks)) > 1.8


def test_hooks_run_in_basilisk_order():
    ns = IMEXBDF2NS(CartesianGrid((8, 8)), periodic=(0, 1), fixed_dt=0.01)
    seen = []
    for stage in STAGES:
        ns.on(stage, lambda s, stage=stage: seen.append(stage))
    ns.start()
    seen.clear()
    ns.step()
    assert seen == list(STAGES)


def test_lid_cavity_early_energy(backend):
    from basilisk_cupy import Dirichlet
    from cases import lid_cavity

    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: Dirichlet(0.0) for w in walls}
    tang["top"] = Dirichlet(1.0)
    ns = IMEXBDF2NS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, fixed_dt=0.01)
    times = [round(0.1 * k, 10) for k in range(1, 12)]
    ns.start(events=times)
    ns.run(1.1)
    assert ns.kinetic_energy() == pytest.approx(lid_cavity.REFERENCE[-1][1], rel=0.01)
    assert ns.max_divergence() < 1e-2

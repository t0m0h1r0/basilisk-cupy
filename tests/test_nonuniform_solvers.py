"""Step 4f-2: solvers on mapped grids — multigrid with line relaxation, Poisson/DC, projection, viscosity."""

import math
import warnings

import numpy as np
import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann, poisson
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.fields import VectorField
from basilisk_cupy.grid.mapping import MonitorMapping, SineMapping, TanhMapping
from basilisk_cupy.linear.multigrid import restrict_residual
from basilisk_cupy.projection.approximate import project
from basilisk_cupy.terms.viscous import CCDViscosity, FD2Stress, FD2Viscosity

from .conftest import order

X, Y = sp.symbols("x y")
XY = (X, Y)
MONITOR = MonitorMapping(lambda x: 1 + 3 * np.exp(-((x - 0.3) / 0.08) ** 2))
NS_ = [16, 32, 64]


def _fn(expr, xp):
    f = sp.lambdify(XY, expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


def _poisson(kind, maps, ops, n, backend):
    periodic = kind == "periodic"
    expr = sp.sin(X + 2 * Y) + sp.cos(2 * X - Y) if periodic else sp.sin(sp.pi * X) * sp.sin(2 * sp.pi * Y) + X * X * Y
    grid = CartesianGrid((n, n), length=(2 * math.pi,) * 2 if periodic else (1.0, 1.0), mapping=maps, backend=backend)
    xp = grid.xp
    gx, gy = _fn(sp.diff(expr, X), xp), _fn(sp.diff(expr, Y), xp)
    if periodic:
        bc = BoundaryConditions.periodic(2)
    elif kind == "dirichlet":
        bc = BoundaryConditions.uniform(2, Dirichlet(_fn(expr, xp)))
    else:
        bc = (BoundaryConditions().set("left", Neumann(lambda x, y: -gx(x, y))).set("right", Neumann(gx))
              .set("bottom", Neumann(lambda x, y: -gy(x, y))).set("top", Neumann(gy)))
    b = ops.div_f(FaceField(grid, [gx(*grid.face_meshgrid(0)), gy(*grid.face_meshgrid(1))]),
                  periodic=bc.periodic_axes(2))  # projection setting
    a = CellField(grid, bc=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = poisson(a, b, ops=ops, tolerance=1e-10 * float(abs(b).max()), nitermax=100)
    err = a.interior - _fn(expr, xp)(*grid.meshgrid())
    if kind != "dirichlet":
        vol = grid.cell_volumes()
        err = err - (err * vol).sum() / (vol + 0 * err).sum()
    return float(abs(to_numpy(err)).max()), stats


CASES = [("dirichlet", (TanhMapping(2.0), MONITOR)), ("neumann", (TanhMapping(2.0), TanhMapping(1.5))),
         ("periodic", (SineMapping(0.5), SineMapping(0.4, 0.3)))]


@pytest.mark.parametrize("kind,maps", CASES)
# fccd, Dirichlet, tanh x monitor: 6.5e-3, 5.6e-5, 4.5e-6 (orders 6.84, 3.64; pre-asymptotic)
@pytest.mark.parametrize("opsname,min_order,max_cycles", [("fd2", 1.9, 20), ("fccd", 3.6, 20)])
def test_poisson_on_mapped_grids(backend, kind, maps, opsname, min_order, max_cycles):
    """fd2 multigrid: 2nd order; FCCD residual + fd2 multigrid (defect correction): 4th order."""
    ops = FD2FaceComplex() if opsname == "fd2" else FCCDFaceComplex()
    errs, its = [], []
    for n in NS_:
        e, stats = _poisson(kind, maps, ops, n, backend)
        assert stats.converged
        errs.append(e)
        its.append(stats.i)
    assert min(order(errs, NS_)) > min_order and max(its) <= max_cycles


@pytest.mark.parametrize("beta", [2.0, 3.0])
def test_line_relaxation_handles_strong_stretching(backend, beta):
    """Point red/black on tanh(beta=2) x tanh(beta=2), 64^2: 52 cycles; zebra lines: ~10 as on a uniform grid."""
    grid = CartesianGrid((64, 64), mapping=(TanhMapping(beta), TanhMapping(beta)), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
    stats = poisson(a, xp.sin(math.pi * x) * xp.sin(math.pi * y), tolerance=1e-10)
    assert stats.converged and stats.i <= 12


def test_residual_restriction_is_volume_weighted(backend):
    grid = CartesianGrid((16, 8), mapping=(TanhMapping(2.0), SineMapping(0.5)), backend=backend)
    r = grid.xp.asarray(np.random.default_rng(0).random(grid.n))
    coarse = restrict_residual(r, grid)
    assert float((grid.cell_volumes() * r).sum()) == pytest.approx(float((grid.coarsen().cell_volumes() * coarse).sum()),
                                                                       rel=1e-13)


def test_fccd_neumann_wall_face_is_the_datum(backend):
    """The wall face gradient equals the Neumann datum exactly (compatibility of pure-Neumann problems)."""
    grid = CartesianGrid((32, 32), mapping=(TanhMapping(2.0), TanhMapping(1.5)), backend=backend)
    f = CellField.from_interior(grid, grid.meshgrid()[0] ** 2,
                                BoundaryConditions().set("left", Neumann(0.7)).set("right", Neumann(-1.3)))
    f.fill_ghosts()
    g = FCCDFaceComplex().grad_f(f, 0)
    assert float(abs(to_numpy(g[0] + 0.7)).max()) < 1e-14 and float(abs(to_numpy(g[-1] + 1.3)).max()) < 1e-14


@pytest.mark.parametrize("opsname", ["fd2", "fccd"])
def test_projection_on_a_mapped_grid(backend, opsname):
    ops = FD2FaceComplex() if opsname == "fd2" else FCCDFaceComplex()
    grid = CartesianGrid((32, 32), mapping=(TanhMapping(2.0), MONITOR), backend=backend)
    xp = grid.xp
    rng = np.random.default_rng(0)
    uf = FaceField(grid, [xp.asarray(rng.standard_normal(grid.face_shape(a))) for a in range(2)])
    uf[0][0] = uf[0][-1] = 0.0
    uf[1][:, 0] = uf[1][:, -1] = 0.0
    p = CellField(grid, bc=BoundaryConditions.uniform(2, Neumann(0.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project(uf, p, None, 1.0, ops=ops, tolerance=1e-10)
    assert float(abs(to_numpy(ops.div_f(uf, periodic=(False, False)))).max()) < 1e-8


def test_fd2_stress_interior_second_order_on_a_mapped_grid(backend):
    U = [sp.sin(X + 2 * Y), sp.cos(X - Y) * X]
    target = [sum(sp.diff(sp.diff(U[a], XY[b]) + sp.diff(U[b], XY[a]), XY[b]) for b in range(2)) for a in range(2)]
    errs = []
    for n in (16, 32, 64, 128):
        grid = CartesianGrid((n, n), mapping=(TanhMapping(2.0), TanhMapping(1.5)), backend=backend)
        u = VectorField([CellField.from_interior(grid, _fn(U[a], grid.xp)(*grid.meshgrid()),
                                                 BoundaryConditions.uniform(2, Dirichlet(_fn(U[a], grid.xp))))
                         for a in range(2)])
        out = FD2Stress()(u, 1.0)
        # wall cells carry the O(1) local error of Basilisk's Dirichlet ghost (also on a uniform grid)
        errs.append(max(float(abs(to_numpy(out[a] - _fn(target[a], grid.xp)(*grid.meshgrid()))[1:-1, 1:-1]).max())
                        for a in range(2)))
    assert min(order(errs, [16, 32, 64, 128])) > 1.8


@pytest.mark.parametrize("cls", [CCDViscosity, FD2Viscosity])
@pytest.mark.parametrize("kappa", [0.1, 1000.0])
def test_viscous_defect_correction_on_a_mapped_grid(backend, cls, kappa):
    """A_L on a mapped grid: componentwise scalar Helmholtz (line-relaxed fd2 multigrid)."""
    grid = CartesianGrid((32, 32), mapping=(TanhMapping(2.0), TanhMapping(2.0)), backend=backend)
    rng = np.random.default_rng(1)
    u = VectorField([CellField.from_interior(grid, grid.xp.asarray(rng.random(grid.n)),
                                             BoundaryConditions.uniform(2, Dirichlet(0.0))) for _ in range(2)])
    stats = cls(maxiter=100).solve(u, FaceField.constant(grid, 1.0), 1.0, kappa * grid.h[0] ** 2, tolerance=1e-10)
    assert stats.converged and stats.i <= 45

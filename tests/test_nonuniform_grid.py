"""Step 4f-1: mapped (non-uniform) grids — maps, metric, CCD/FCCD/fd2 operators (TwoPhaseFlow paper §10)."""

import math

import numpy as np
import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import CCDCellDerivative, FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.discretization.fd2 import FD2CellDerivative
from basilisk_cupy.fields import VectorField
from basilisk_cupy.grid.hierarchy import build_levels
from basilisk_cupy.grid.mapping import MonitorMapping, SineMapping, TanhMapping, UniformMapping
from basilisk_cupy.terms.viscous import CCDStress

from .conftest import order

X, Y = sp.symbols("x y")
XY = (X, Y)
F_WALL = sp.sin(3 * X + 1) * sp.cos(2 * Y + sp.Rational(1, 2)) + X * Y
F_PERIODIC = sp.sin(X + 2 * Y) + sp.cos(2 * X - Y)
NS_ = [16, 32, 64, 128]
MONITOR = MonitorMapping(lambda x: 1 + 3 * np.exp(-((x - 0.3) / 0.08) ** 2))


def _fn(expr, xp):
    f = sp.lambdify(XY, expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


def _field(n, maps, kind, backend):
    periodic = kind == "periodic"
    expr = F_PERIODIC if periodic else F_WALL
    grid = CartesianGrid((n, n), length=(2 * math.pi,) * 2 if periodic else (1.0, 1.2), origin=(0.0, -0.1),
                         mapping=maps, backend=backend)
    xp = grid.xp
    if periodic:
        bc = BoundaryConditions.periodic(2)
    elif kind == "dirichlet":
        bc = BoundaryConditions.uniform(2, Dirichlet(_fn(expr, xp)))
    else:
        bc = (BoundaryConditions().set("left", Neumann(_fn(-sp.diff(expr, X), xp)))
              .set("right", Neumann(_fn(sp.diff(expr, X), xp)))
              .set("bottom", Neumann(_fn(-sp.diff(expr, Y), xp))).set("top", Neumann(_fn(sp.diff(expr, Y), xp))))
    f = CellField.from_interior(grid, _fn(expr, xp)(*grid.meshgrid()), bc)
    f.fill_ghosts()
    return grid, f, expr


def _err(a, b):
    return float(abs(to_numpy(a - b)).max())


# -- maps -------------------------------------------------------------------------------
@pytest.mark.parametrize("mapping", [TanhMapping(2.0), TanhMapping(3.0, side="lower"), TanhMapping(3.0, side="upper"),
                                     SineMapping(0.5), SineMapping(0.4, 0.3), MONITOR])
def test_maps_are_increasing_with_exact_derivative(mapping):
    s = np.linspace(0.0, 1.0, 2001)
    m = mapping.unit(s)
    assert m[0] == pytest.approx(0.0, abs=1e-14) and m[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(m) > 0)
    fd = (mapping.unit(s[1:-1] + 1e-6) - mapping.unit(s[1:-1] - 1e-6)) / 2e-6
    assert np.allclose(fd, mapping.derivative(s[1:-1]), rtol=1e-6, atol=1e-8)


def test_ghost_extension_mirrors_walls_and_translates_periodic_axes():
    t, p = TanhMapping(2.0), SineMapping(0.5, 0.2)
    s = np.array([0.1, 0.3])
    assert np.allclose(t.extended(-s), -t.unit(s)) and np.allclose(t.extended(1 + s), 2 - t.unit(1 - s))
    assert np.allclose(p.extended(s + 1), p.unit(s) + 1) and np.allclose(p.extended(s - 1), p.unit(s) - 1)


def test_monitor_map_equidistributes():
    """Every cell carries the same integral of the monitor (Boole quadrature + Hermite inverse)."""
    faces = MONITOR.unit(np.linspace(0.0, 1.0, 65))
    x = np.linspace(0.0, 1.0, 2_000_001)
    w = MONITOR.omega(x)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))])
    per_cell = np.diff(np.interp(faces, x, cum))
    assert per_cell.std() / per_cell.mean() < 1e-9


def test_grid_equality_follows_the_map():
    g = CartesianGrid((8, 8), mapping=(TanhMapping(2.0), None))
    assert g == CartesianGrid((8, 8), mapping=(TanhMapping(2.0), None))
    assert g != CartesianGrid((8, 8), mapping=(TanhMapping(2.5), None))
    assert CartesianGrid((8, 8), mapping=(UniformMapping(), None)).is_uniform
    other = MonitorMapping(lambda x: 1 + 3 * np.exp(-((x - 0.3) / 0.08) ** 2))
    assert CartesianGrid((8,), mapping=(MONITOR,)) != CartesianGrid((8,), mapping=(other,))


def test_coarse_grids_nest_in_the_fine_one():
    fine = CartesianGrid((16, 8), mapping=(TanhMapping(2.0), SineMapping(0.5)))
    coarse = fine.coarsen()
    assert np.allclose(to_numpy(coarse.faces(0)), to_numpy(fine.faces(0))[::2], atol=1e-15)
    assert np.allclose(to_numpy(coarse.centers(0)), to_numpy(fine.faces(0))[1::2], atol=1e-15)


# -- metric and geometric conservation --------------------------------------------------------
def test_uniform_map_gives_the_uniform_metric_and_bitwise_operators(backend):
    a, f, _ = _field(32, None, "dirichlet", backend)
    b, g, _ = _field(32, (UniformMapping(), UniformMapping()), "dirichlet", backend)
    assert a == b and b.is_uniform
    for op in (lambda c: CCDCellDerivative().d2(c, 1), lambda c: FCCDFaceComplex().grad_f(c, 0),
               lambda c: FD2FaceComplex().grad_f(c, 1)):
        np.testing.assert_array_equal(to_numpy(op(f)), to_numpy(op(g)))


@pytest.mark.parametrize("maps", [(TanhMapping(2.0), TanhMapping(1.5)), (MONITOR, TanhMapping(3.0, side="lower"))])
def test_geometric_conservation(backend, maps):
    """d x / d x = 1 and d c / d x = 0 to round-off for CCD, FCCD and fd2 (metric from the same operators)."""
    grid = CartesianGrid((64, 48), length=(1.0, 1.2), origin=(0.0, -0.1), mapping=maps, backend=backend)
    assert float(to_numpy(grid.metric(0).width).sum()) == pytest.approx(1.0, abs=1e-14)
    for a in range(2):
        coord = grid.meshgrid()[a]
        f = CellField.from_interior(grid, coord, BoundaryConditions.uniform(2, Dirichlet(lambda *c, a=a: c[a])))
        f.fill_ghosts()
        assert _err(CCDCellDerivative().d1(f, a), 1.0) < 1e-11
        assert float(abs(to_numpy(CCDCellDerivative().d2(f, a))).max()) < 1e-9
        assert _err(FCCDFaceComplex().grad_f(f, a), 1.0) < 1e-11
        assert _err(FD2FaceComplex().grad_f(f, a), 1.0) < 1e-12
        const = CellField.from_interior(grid, 0 * coord + 2.5, BoundaryConditions.uniform(2, Neumann(0.0)))
        const.fill_ghosts()
        # the xi-space round-off of the dense CCD operator (~2e-11 here, also on a uniform grid) times J
        j_max = float(to_numpy(grid.metric(a).J).max())
        assert float(abs(to_numpy(CCDCellDerivative().d1(const, a))).max()) < 5e-11 * j_max


# -- operator orders ------------------------------------------------------------------------
CASES = [("dirichlet", (TanhMapping(2.0), TanhMapping(1.5))), ("neumann", (TanhMapping(2.0), TanhMapping(1.5))),
         ("periodic", (SineMapping(0.5), SineMapping(0.4, 0.3)))]


@pytest.mark.parametrize("kind,maps", CASES)
def test_ccd_orders_on_mapped_grids(backend, kind, maps):
    ccd = CCDCellDerivative()
    errs = {k: [] for k in ("d1x", "d2x", "d1y", "d2y", "d11")}
    for n in NS_:
        grid, f, expr = _field(n, maps, kind, backend)
        x, y = grid.meshgrid()
        ex = lambda e: _fn(e, grid.xp)(x, y)  # noqa: E731
        errs["d1x"].append(_err(ccd.d1(f, 0), ex(sp.diff(expr, X))))
        errs["d2x"].append(_err(ccd.d2(f, 0), ex(sp.diff(expr, X, 2))))
        errs["d1y"].append(_err(ccd.d1(f, 1), ex(sp.diff(expr, Y))))
        errs["d2y"].append(_err(ccd.d2(f, 1), ex(sp.diff(expr, Y, 2))))
        errs["d11"].append(_err(ccd.d11(f, 0, 1), ex(sp.diff(expr, X, Y))))
    # walls: compact closures, f' 4th / f'' 3rd / d11 3rd order at the wall cells (measured >= 4.02 / 2.82 / 3.05)
    walls = kind != "periodic"
    for key, min_order in (("d1x", 3.9 if walls else 5.9), ("d1y", 3.9 if walls else 5.9), ("d2x", 2.7 if walls else 5.9),
                           ("d2y", 2.7 if walls else 5.9), ("d11", 2.9 if walls else 5.2)):
        assert min(order(errs[key][1:], NS_[1:])) > min_order, (key, errs[key])


@pytest.mark.parametrize("kind,maps", CASES)
def test_fccd_orders_on_mapped_grids(backend, kind, maps):
    fccd = FCCDFaceComplex()
    eg, ev, ec = [], [], []
    for n in NS_:
        grid, f, expr = _field(n, maps, kind, backend)
        xf, yf = grid.face_meshgrid(0)
        xg, yg = grid.face_meshgrid(1)
        x, y = grid.meshgrid()
        eg.append(_err(fccd.grad_f(f, 0), _fn(sp.diff(expr, X), grid.xp)(xf, yf)))
        ev.append(_err(fccd.interp_f(f, 1), _fn(expr, grid.xp)(xg, yg)))
        ec.append(_err(fccd.face_to_cell(_fn(expr, grid.xp)(xf, yf), 0, periodic=kind == "periodic"),
                       _fn(expr, grid.xp)(x, y)))
    for errs in (eg, ev, ec):
        assert min(order(errs[1:], NS_[1:])) > 3.8


@pytest.mark.parametrize("kind,maps", CASES)
def test_fd2_orders_on_mapped_grids(backend, kind, maps):
    """Finite-volume fd2: gradient over the centre distance, divergence over the cell width: 2nd order."""
    fd2 = FD2FaceComplex()
    eg, ed = [], []
    for n in NS_:
        grid, f, expr = _field(n, maps, kind, backend)
        xf, yf = grid.face_meshgrid(0)
        xg, yg = grid.face_meshgrid(1)
        g = fd2.grad_f(f, 0)
        exact = _fn(sp.diff(expr, X), grid.xp)(xf, yf)
        if kind != "periodic":  # wall faces use the ghost values
            g, exact = g[1:-1], exact[1:-1]
        eg.append(_err(g, exact))
        flux = FaceField(grid, [_fn(sp.diff(expr, X), grid.xp)(xf, yf), _fn(sp.diff(expr, Y), grid.xp)(xg, yg)])
        ed.append(_err(fd2.div_f(flux), _fn(sp.diff(expr, X, 2) + sp.diff(expr, Y, 2), grid.xp)(*grid.meshgrid())))
    for errs in (eg, ed):
        assert min(order(errs, NS_)) > 1.9


def test_fd2_cell_derivatives_on_a_mapped_grid(backend):
    errs = []
    for n in NS_:
        grid, f, expr = _field(n, (TanhMapping(2.0), TanhMapping(1.5)), "neumann", backend)
        errs.append(_err(FD2CellDerivative().d2(f, 1), _fn(sp.diff(expr, Y, 2), grid.xp)(*grid.meshgrid())))
    assert min(order(errs, NS_)) > 1.9


def test_ccd_stress_on_a_mapped_grid(backend):
    U = [sp.sin(X + 2 * Y), sp.cos(X - Y) * X]
    target = [sum(sp.diff(sp.diff(U[a], XY[b]) + sp.diff(U[b], XY[a]), XY[b]) for b in range(2)) for a in range(2)]
    errs = []
    for n in NS_:
        grid = CartesianGrid((n, n), mapping=(TanhMapping(2.0), MONITOR), backend=backend)
        comps = [CellField.from_interior(grid, _fn(U[a], grid.xp)(*grid.meshgrid()),
                                         BoundaryConditions.uniform(2, Dirichlet(_fn(U[a], grid.xp)))) for a in range(2)]
        u = VectorField(comps)
        u.fill_ghosts()
        out = CCDStress()(u, 1.0)
        errs.append(max(_err(out[a], _fn(target[a], grid.xp)(*grid.meshgrid())) for a in range(2)))
    assert min(order(errs[1:], NS_[1:])) > 3.9  # wall cells (compact closures): measured 4.08, 4.26


def test_wall_clustering_resolves_a_boundary_layer(backend):
    """f = exp(-x/0.02): at N = 32 a tanh map lowers the CCD derivative error 400-fold (1.4e-2 -> 3.3e-5)."""
    errs = {}
    for name, mp in (("uniform", None), ("tanh", (TanhMapping(3.0, side="lower"),))):
        grid = CartesianGrid((32,), mapping=mp, backend=backend)
        xp = grid.xp
        x, = grid.meshgrid()
        f = CellField.from_interior(grid, xp.exp(-x / 0.02), BoundaryConditions.uniform(1, Dirichlet(lambda x: xp.exp(-x / 0.02))))
        errs[name] = _err(CCDCellDerivative().d1(f, 0) * 0.02, -xp.exp(-x / 0.02))
    assert errs["tanh"] < errs["uniform"] / 400


def test_solvers_not_yet_ported_refuse_mapped_grids():
    from basilisk_cupy.terms.convection.uccd6 import UCCD6Convection

    from basilisk_cupy.linear.viscosity import ViscosityProblem

    grid = CartesianGrid((16, 16), mapping=(TanhMapping(2.0), None))
    assert len(build_levels(grid)) == 4  # multigrid works on mapped grids (step 4f-2)
    with pytest.raises(NotImplementedError):  # Basilisk's viscosity stencil needs a single Delta
        ViscosityProblem(VectorField.zeros(grid, [BoundaryConditions(), BoundaryConditions()]),
                         FaceField.constant(grid, 1.0), 1.0, 0.1)
    with pytest.raises(NotImplementedError):
        UCCD6Convection().tendency(VectorField.zeros(grid, [BoundaryConditions(), BoundaryConditions()]))
    with pytest.raises(ValueError):
        grid.cell_volume()
    assert to_numpy(grid.cell_volumes()).sum() == pytest.approx(1.0)

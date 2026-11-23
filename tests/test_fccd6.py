"""FCCD face value and gradient of order 6: quintic Hermite of the two cell jets (f, f', f'') of one CCD solve."""

import math

import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FCCDFaceComplex
from basilisk_cupy.grid.mapping import SineMapping, TanhMapping

from .conftest import order
from .test_ccd import exact, field

NS_ = [16, 32, 64, 128]


@pytest.mark.parametrize("kind", ["periodic", "dirichlet", "neumann"])
def test_sixth_order_away_from_walls(backend, kind):
    """Periodic: 6th order (N = 128: gradient 3.8e-12, value 3.0e-13; order 4: 2.1e-8, 7.6e-8).
    Walls: the wall faces keep the 4th-order closures; the faces away from them are 6th order."""
    ops = FCCDFaceComplex(order=6)
    eg, ev = [], []
    for n in NS_:
        grid, f = field(n, kind, backend)
        xp = grid.xp
        xf, yf = grid.face_meshgrid(0)
        sl = slice(None) if kind == "periodic" else slice(n // 4, 3 * n // 4)
        eg.append(float(abs(to_numpy((ops.grad_f(f, 0) - exact(xp, xf, yf, dx=1))[sl])).max()))
        ev.append(float(abs(to_numpy((ops.interp_f(f, 0) - exact(xp, xf, yf))[sl])).max()))
    for errs in (eg, ev):
        assert min(order(errs[:3], NS_[:3])) > 5.8, errs


def test_order_four_is_unchanged_and_order_is_checked():
    with pytest.raises(ValueError):
        FCCDFaceComplex(order=5)
    assert FCCDFaceComplex().order == 4


def test_mapped_periodic_sixth_order_with_its_own_face_metric(backend):
    """J_face from the 6th-order face gradient of the coordinates (4th-order J_face: stuck at 4.31)."""
    X, Y = sp.symbols("x y")
    expr = sp.sin(X + 2 * Y) + sp.cos(2 * X - Y)
    errs = []
    for n in NS_:
        grid = CartesianGrid((n, n), length=(2 * math.pi,) * 2, mapping=(SineMapping(0.5), SineMapping(0.4, 0.3)),
                             backend=backend)
        xp = grid.xp
        fn = sp.lambdify((X, Y), expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
        dfn = sp.lambdify((X, Y), sp.diff(expr, X), modules=[{"sin": xp.sin, "cos": xp.cos}])
        f = CellField.from_interior(grid, fn(*grid.meshgrid()), BoundaryConditions.periodic(2))
        errs.append(float(abs(to_numpy(FCCDFaceComplex(order=6).grad_f(f, 0) - dfn(*grid.face_meshgrid(0)))).max()))
    assert min(order(errs[1:], NS_[1:])) > 5.8  # measured 6.12, 6.03


@pytest.mark.parametrize("order_", [4, 6])
def test_geometric_conservation(backend, order_):
    grid = CartesianGrid((64, 48), length=(1.0, 1.2), origin=(0.0, -0.1), mapping=(TanhMapping(2.0), TanhMapping(1.5)),
                         backend=backend)
    for a in range(2):
        coord = grid.meshgrid()[a]
        f = CellField.from_interior(grid, coord, BoundaryConditions.uniform(2, Dirichlet(lambda *c, a=a: c[a])))
        f.fill_ghosts()
        assert float(abs(to_numpy(FCCDFaceComplex(order=order_).grad_f(f, a) - 1.0)).max()) < 1e-12

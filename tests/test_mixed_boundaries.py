"""Periodic and wall axes together (channel: x periodic, y walls) for the CCD-based operators and NS."""

import math
import warnings

import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic
from basilisk_cupy.discretization import CCDCellDerivative, FCCDFaceComplex
from basilisk_cupy.fields import VectorField
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.terms.convection.uccd6 import UCCD6Convection
from basilisk_cupy.terms.viscous import CCDStress
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS

from .conftest import order
from .test_ccd_viscosity import _solve

X, Y = sp.symbols("x y")
XY = (X, Y)
U = [sp.sin(X + 2 * Y) + sp.Rational(3, 10) * sp.cos(2 * X - Y), sp.cos(X + Y) * sp.sin(2 * X) + sp.Rational(1, 5)]
SKEW = [-sp.Rational(1, 2) * sum(U[k] * sp.diff(U[j], XY[k]) + sp.diff(U[k] * U[j], XY[k]) for k in range(2))
        for j in range(2)]
STRESS = [sum(sp.diff(sp.diff(U[a], XY[b]) + sp.diff(U[b], XY[a]), XY[b]) for b in range(2)) for a in range(2)]
NS_ = [16, 32, 64, 128]


def _fn(expr, xp):
    f = sp.lambdify(XY, expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


def _channel(n, backend):
    grid = CartesianGrid((n, n), length=(2 * math.pi, 2.0), origin=(0.0, -0.2), backend=backend)
    exact = [_fn(e, grid.xp) for e in U]
    comps = []
    for j in range(2):
        bc = (BoundaryConditions().set("left", Periodic()).set("right", Periodic())
              .set("bottom", Dirichlet(exact[j])).set("top", Dirichlet(exact[j])))
        comps.append(CellField.from_interior(grid, exact[j](*grid.meshgrid()), bc))
    u = VectorField(comps)
    u.fill_ghosts()
    return grid, u, FaceField(grid, [exact[k](*grid.face_meshgrid(k)) for k in range(2)])


def _errors(evaluate, target, backend):
    errs = []
    for n in NS_:
        grid, u, uf = _channel(n, backend)
        x, y = grid.meshgrid()
        out = evaluate(grid, u, uf)
        errs.append(max(float(abs(to_numpy(o - _fn(t, grid.xp)(x, y))).max()) for o, t in zip(out, target)))
    return errs


def test_ccd_derivatives_in_a_channel(backend):
    ccd = CCDCellDerivative()
    for axis, deriv, min_order in ((0, 1, 5.9), (1, 1, 3.9), (0, 2, 5.9), (1, 2, 2.9)):  # wall axis: 4.00 / 3.00
        errs = _errors(lambda g, u, uf: [ccd.derivatives(u[0], axis)[deriv - 1]],
                       [sp.diff(U[0], XY[axis], deriv)], backend)
        assert min(order(errs, NS_)) > min_order


def test_fccd_wall_normal_gradient_in_a_channel(backend):
    errs = []
    for n in NS_[:3]:
        grid, u, uf = _channel(n, backend)
        xf, yf = grid.face_meshgrid(1)
        errs.append(float(abs(to_numpy(FCCDFaceComplex().grad_f(u[0], 1) - _fn(sp.diff(U[0], Y), grid.xp)(xf, yf))).max()))
    assert min(order(errs, NS_[:3])) > 3.9


def test_uccd6_in_a_channel(backend):
    errs = _errors(lambda g, u, uf: UCCD6Convection().tendency(u, uf), SKEW, backend)
    assert min(order(errs, NS_)) > 4.0  # 7.05, 6.57, 4.28: the wall-cell D1 (4th order) shows at N = 128


@pytest.mark.parametrize("form", ["split", "layered"])
def test_ccd_stress_in_a_channel(backend, form):
    errs = _errors(lambda g, u, uf: CCDStress(form=form)(u, 1.0), STRESS, backend)
    assert min(order(errs, NS_)) > 2.9  # wall cells: 3rd order (measured 3.01)


@pytest.mark.parametrize("kappa", [0.1, 1000.0])
def test_viscous_defect_correction_periodic(backend, kappa):
    stats = _solve("split", 32, kappa, backend, walls=False)
    assert stats.converged and stats.i <= 30


def _channel_decay(backend, n, viscosity, nu=0.05, t_end=0.5):
    """u = sin(pi y) exp(-nu pi^2 t), v = 0: exact NS solution between no-slip walls (convection vanishes)."""
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "uccd6", "viscosity": viscosity}})
    ns = IMEXBDF2NS(grid, periodic=(0,), tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)},
                    uf_normal={"bottom": 0.0, "top": 0.0}, mu=nu, ops=scheme.face, convection=scheme.convection,
                    viscous=scheme.viscous, tolerance=1e-12, fixed_dt=t_end / round(2 * t_end * n * n))
    ns.u[0].interior = xp.sin(math.pi * y) + 0 * x
    ns.start(events=[t_end])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(t_end)
    ex = xp.sin(math.pi * y) * math.exp(-nu * math.pi ** 2 * t_end)
    return max(float(xp.max(xp.abs(ns.u[0].interior - ex))), float(xp.max(xp.abs(ns.u[1].interior))))


def test_channel_decay_with_ccd_viscosity(backend):
    """N = 16: CCD viscosity 1.1e-6 (N = 32 / 64: 4.0e-8 / 2.0e-9), fd2 viscosity 6.2e-4 (2nd order in space)."""
    ccd = _channel_decay(backend, 16, "ccd")
    fd2 = _channel_decay(backend, 16, "basilisk")
    assert ccd < 2e-6 and fd2 > 100 * ccd

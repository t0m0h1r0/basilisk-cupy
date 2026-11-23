"""4th-order conservative face divergence of FCCD (``FCCDFaceComplex.div_f`` / ``face_flux``).

The plain face difference of point values is ``F_x + h^2/24 F_xxx``: 2nd order. FCCD differences
the flux ``F - h^2/24 d_nn F`` (face-line CCD), the dual of its face gradient: 4th order, and the
sum over the cells still telescopes. The standard Taylor--Green vortex (``kx = ky``) hides the
2nd-order term (``u_xxx + v_yyy = 0``); ``psi = sin(a x) sin(b y)`` with ``a != b`` shows it.
"""

import math
import warnings

import numpy as np
import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.grid.mapping import MonitorMapping, SineMapping, TanhMapping
from basilisk_cupy.projection.approximate import project
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.time.imex_rk import IMEXRKNS

from .conftest import order

X, Y = sp.symbols("x y")
MONITOR = MonitorMapping(lambda x: 1 + 3 * np.exp(-((x - 0.3) / 0.08) ** 2))
NS_ = [16, 32, 64, 128]


def _fn(expr, xp):
    f = sp.lambdify((X, Y), expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


CASES = [  # (name, maps, periodic); measured FCCD orders (16..128): >= 4.02, 3.44 / 5.63 / 3.97, >= 3.98, >= 3.71
    ("walls", None, False), ("walls, tanh x monitor", (TanhMapping(2.0), MONITOR), False),
    ("periodic", None, True), ("periodic, sine x sine", (SineMapping(0.5), SineMapping(0.4, 0.3)), True)]


def _errors(maps, periodic, backend):
    fx, fy = ((sp.sin(X + 2 * Y), sp.cos(2 * X - Y)) if periodic
              else (sp.sin(3 * X + 1) * sp.cos(2 * Y + 0.5), sp.cos(X - 2 * Y) * X))
    high, plain = [], []
    for n in NS_:
        grid = CartesianGrid((n, n), length=(2 * math.pi,) * 2 if periodic else (1.0, 1.0), mapping=maps,
                             backend=backend)
        xp = grid.xp
        flux = FaceField(grid, [_fn(fx, xp)(*grid.face_meshgrid(0)), _fn(fy, xp)(*grid.face_meshgrid(1))])
        exact = _fn(sp.diff(fx, X) + sp.diff(fy, Y), xp)(*grid.meshgrid())
        high.append(float(abs(to_numpy(FCCDFaceComplex().div_f(flux, periodic=(periodic,) * 2) - exact)).max()))
        plain.append(float(abs(to_numpy(FD2FaceComplex().div_f(flux) - exact)).max()))
    return high, plain


@pytest.mark.parametrize("name,maps,periodic", CASES)
def test_fourth_order_divergence(backend, name, maps, periodic):
    high, plain = _errors(maps, periodic, backend)
    assert min(order(high[1:], NS_[1:])) > 3.9, (name, high)
    assert max(order(plain[1:], NS_[1:])) < 2.1 and high[-1] < plain[-1] / 30


def test_divergence_is_the_face_difference_of_the_conservative_flux(backend):
    grid = CartesianGrid((16, 12), mapping=(TanhMapping(2.0), None), backend=backend)
    rng = np.random.default_rng(0)
    flux = FaceField(grid, [grid.xp.asarray(rng.standard_normal(grid.face_shape(a))) for a in range(2)])
    ops = FCCDFaceComplex()
    per = (False, True)
    flux[1][:, -1] = flux[1][:, 0]  # periodic in y: face n is face 0
    hat = FaceField(grid, [ops.face_flux(flux[a], a, per[a]) for a in range(2)])
    # sum over the cells of div * (h / J) telescopes to the conservative wall fluxes
    vol = (grid.h[0] / grid.metric(0).along(grid.metric(0).J, 2)) * grid.h[1]
    total = float((ops.div_f(flux, periodic=per) * vol).sum())
    walls = float(((hat[0][-1] - hat[0][0]) * grid.h[1]).sum())
    assert total == pytest.approx(walls, abs=1e-12 * float(abs(to_numpy(hat[0])).max()) * 16)


def test_projection_of_an_exact_field_moves_it_by_h4(backend):
    """psi = sin(2 pi x) sin(4 pi y): the plain face difference moves the exact faces by O(h^2)."""
    moved = {"fccd": [], "plain": []}
    for n in (16, 32, 64):
        grid = CartesianGrid((n, n), backend=backend)
        xp = grid.xp
        a, b = 2 * math.pi, 4 * math.pi
        u = lambda x, y: b / b * xp.sin(a * x) * xp.cos(b * y)  # noqa: E731
        v = lambda x, y: -a / b * xp.cos(a * x) * xp.sin(b * y)  # noqa: E731
        for name, ops in (("fccd", FCCDFaceComplex()), ("plain", FD2FaceComplex())):
            uf = FaceField(grid, [u(*grid.face_meshgrid(0)), v(*grid.face_meshgrid(1))])
            ref = [uf[k].copy() for k in range(2)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                project(uf, CellField(grid, bc=BoundaryConditions.periodic(2)), None, 1.0, ops=ops, tolerance=1e-13)
            moved[name].append(max(float(abs(to_numpy(uf[k] - ref[k])).max()) for k in range(2)))
    # measured: fccd 1.7e-4 / 1.1e-5 / 6.8e-7, plain 7.7e-3 / 1.9e-3 / 4.8e-4
    assert min(order(moved["fccd"], [16, 32, 64])) > 3.9 and max(order(moved["plain"], [16, 32, 64])) < 2.1


def _taylor_green(n, backend, ka=1, kb=2, mu=0.01, t_end=0.25):
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    a, b = 2 * math.pi * ka, 2 * math.pi * kb
    s = max(a, b)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "central", "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, periodic=(0, 1), mu=mu, ops=scheme.face, convection=scheme.convection,
                  viscous=scheme.viscous, fixed_dt=t_end / round(t_end * n), tolerance=1e-12, face_native=True,
                  reconstruction="ccd")
    U = lambda x, y: b / s * xp.sin(a * x) * xp.cos(b * y)  # noqa: E731
    V = lambda x, y: -a / s * xp.cos(a * x) * xp.sin(b * y)  # noqa: E731
    x, y = grid.meshgrid()
    ns.u[0].interior = U(x, y)
    ns.u[1].interior = V(x, y)
    ns.p.interior = (b * b * xp.cos(2 * a * x) + a * a * xp.cos(2 * b * y)) / (4 * s * s)
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(t_end)
    d = math.exp(-mu * (a * a + b * b) * t_end)
    return max(float(xp.max(xp.abs(ns.u[0].interior - d * U(x, y)))), float(xp.max(xp.abs(ns.u[1].interior - d * V(x, y)))))


def test_taylor_green_with_unequal_wavenumbers_is_fourth_order(backend):
    """Face state, IMEX-RK, CFL 1: 1.39e-3 / 1.25e-4 / 8.49e-6 (N = 16 / 32 / 64); the plain face
    difference gave 1.14e-2 / 3.08e-3 / 7.84e-4 (2nd order)."""
    errs = [_taylor_green(n, backend) for n in (32, 64)]
    assert math.log2(errs[0] / errs[1]) > 3.7 and errs[1] < 2e-5

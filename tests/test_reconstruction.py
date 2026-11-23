"""High-order face-to-cell reconstruction R_h (discretization/reconstruction.py) for the face-native state."""

import math
import warnings

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid, Dirichlet
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FCCDFaceComplex
from basilisk_cupy.discretization.reconstruction import (HERMITE, _WALL_I, _WALL_II, CompactFaceToCell, FaceToCell,
                                                        face_ccd_matrices, reconstruction_matrix)
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.time.imex_rk import IMEXRKNS

from .conftest import order


@pytest.mark.parametrize("periodic", [True, False])
def test_compact_ccd_reconstruction_sixth_order(backend, periodic):
    """Face-line CCD jet + quintic Hermite midpoint: 6th order, more accurate than the 6-point stencil."""
    xp = np.asarray if backend == "numpy" else __import__("cupy").asarray
    errs, lag, ns = [], [], [16, 32, 64, 128]
    for n in ns:
        L = 2 * math.pi if periodic else 1.0
        xf, xc = np.arange(n + 1) * L / n, (np.arange(n) + 0.5) * L / n
        f = (lambda x: np.sin(x + 1)) if periodic else (lambda x: np.sin(3 * x + 1) * np.exp(x))
        errs.append(float(abs(to_numpy(CompactFaceToCell()(xp(f(xf)), 0, periodic=periodic)) - f(xc)).max()))
        lag.append(float(abs(to_numpy(FaceToCell(6)(xp(f(xf)), 0, periodic=periodic)) - f(xc)).max()))
    if periodic:
        assert min(order(errs, ns)) > 5.9 and errs[-1] < lag[-1] / 100  # measured 3.0e-13 vs 6.8e-11
    else:  # compact wall rows (nodes 0..2): 5th order at walls
        assert min(order(errs[1:], ns[1:])) > 4.8


def test_face_line_ccd_derivatives(backend):
    """Compact wall rows: f'' 3rd order, and through the implicit h f''_0 of row I f' 4th order;
    both enter R_h as O(h^5)."""
    e1, e2, ns = [], [], [32, 64, 128]
    for n in ns:
        h = 1.0 / n
        D1, D2 = face_ccd_matrices(n, h, False)
        x = np.arange(n + 1) * h
        f = np.sin(3 * x + 1) * np.exp(x)
        fp = (3 * np.cos(3 * x + 1) + np.sin(3 * x + 1)) * np.exp(x)
        fpp = (6 * np.cos(3 * x + 1) - 8 * np.sin(3 * x + 1)) * np.exp(x)
        e1.append(abs(D1 @ f - fp).max())
        e2.append(abs(D2 @ f - fpp).max())
    assert min(order(e1, ns)) > 3.8 and min(order(e2, ns)) > 2.8


@pytest.mark.skipif(__import__("importlib").util.find_spec("sympy") is None, reason="sympy not installed")
def test_face_ccd_coefficients_regenerate():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "derive_face_ccd", Path(__file__).resolve().parents[1] / "tools" / "derive_face_ccd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.derive()
    assert tuple(out["I"]["implicit"]) == _WALL_I["implicit"] and tuple(out["I"]["explicit"]) == _WALL_I["explicit"]
    assert tuple(out["II"]["implicit"]) == _WALL_II["implicit"] and tuple(out["II"]["explicit"]) == _WALL_II["explicit"]
    assert tuple(out["hermite"]) == HERMITE


def test_interior_weights():
    np.testing.assert_allclose(reconstruction_matrix(12, 6, True)[5, 3:9] * 256, [3, -25, 150, 150, -25, 3])
    np.testing.assert_allclose(reconstruction_matrix(12, 4, True)[5, 4:8] * 16, [-1, 9, 9, -1])


@pytest.mark.parametrize("p", [4, 6, 8])
@pytest.mark.parametrize("periodic", [True, False])
def test_exact_for_polynomials_and_order_p(backend, p, periodic):
    xp = np.asarray if backend == "numpy" else __import__("cupy").asarray
    R = FaceToCell(p)
    if not periodic:  # polynomials of degree p - 1 are reproduced, the wall rows included
        n = 12
        xf, xc = np.arange(n + 1) / n, (np.arange(n) + 0.5) / n
        poly = sum((k + 1) * xf ** k for k in range(p))
        exact = sum((k + 1) * xc ** k for k in range(p))
        assert float(abs(to_numpy(R(xp(poly), 0, periodic=False)) - exact).max()) < 1e-10
    errs, ns = [], [16, 32, 64, 128]
    for n in ns:
        L = 2 * math.pi if periodic else 1.0
        xf, xc = np.arange(n + 1) * L / n, (np.arange(n) + 0.5) * L / n
        f = (lambda x: np.sin(x + 1)) if periodic else (lambda x: np.sin(3 * x + 1) * np.exp(x))
        errs.append(float(abs(to_numpy(R(xp(f(xf)), 0, periodic=periodic)) - f(xc)).max()))
    assert min(order(errs[:3], ns[:3])) > p - 0.5


def test_four_points_are_fccd_face_to_cell(backend):
    grid = CartesianGrid((16, 12), backend=backend)
    faces = grid.xp.asarray(np.random.default_rng(0).random(grid.face_shape(0)))
    for periodic in (True, False):
        np.testing.assert_allclose(to_numpy(FaceToCell(4)(faces, 0, periodic=periodic)),
                                   to_numpy(FCCDFaceComplex().face_to_cell(faces, 0, periodic=periodic)), atol=1e-14)


def _taylor_green(n, backend, reconstruction):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "central", "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, periodic=(0, 1), mu=0.01, ops=scheme.face, convection=scheme.convection,
                  viscous=scheme.viscous, fixed_dt=0.25 / round(0.25 * n), tolerance=1e-11, face_native=True,
                  reconstruction=reconstruction)
    ex = [lambda x, y: -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y),
          lambda x, y: xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)]
    for a in range(2):
        ns.u[a].interior = ex[a](x, y)
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[0.25])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(0.25)
    d = math.exp(-8 * math.pi ** 2 * 0.01 * 0.25)
    cells = max(float(xp.max(xp.abs(ns.u[a].interior - d * ex[a](x, y)))) for a in range(2))
    faces = max(float(xp.max(xp.abs(ns.uf[a] - d * ex[a](*grid.face_meshgrid(a))))) for a in range(2))
    return cells, faces


def test_high_order_reconstruction_lowers_the_cell_floor(backend):
    """Face state, IMEX-RK + FCCD, CFL 1, N = 32: cells 3.5e-5 with FCCD's 4th-order face_to_cell, 1.3e-5 with
    the 6-point Lagrange R_h; cells and faces are then at the same level (the faces' 4th order)."""
    c4, _ = _taylor_green(32, backend, None)
    c6, f6 = _taylor_green(32, backend, 6)
    cc, fc = _taylor_green(32, backend, "ccd")
    assert c6 < c4 / 2 and c6 == pytest.approx(f6, rel=0.2)
    assert cc < c4 / 2 and cc == pytest.approx(fc, rel=0.2)  # the floor is now the 4th-order face operators


def test_lid_cavity_with_walls(backend):
    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: Dirichlet(0.0) for w in walls}
    tang["top"] = Dirichlet(1.0)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "central", "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                  convection=scheme.convection, viscous=scheme.viscous, fixed_dt=1.0 / 64, face_native=True,
                  reconstruction="ccd")
    ns.start(events=[1.1])
    ns.run(1.1)
    assert ns.kinetic_energy() == pytest.approx(0.009257, rel=0.01)  # 4th-order R_h, same run: 0.009257
    assert ns.max_divergence() < 1e-5

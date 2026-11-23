"""Step 4a: CCD / FCCD on the cell-centred grid (convergence orders, consistency, swapping into the solver)."""

import importlib.util
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann, poisson
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import CCDCellDerivative, FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.scheme import build_scheme

from .conftest import order

K = (3.0, 2.0)
PHASE = (1.0, 0.5)


def exact(xp, x, y, dx=0, dy=0):
    """Derivatives of sin(3x + 1) cos(2y + 0.5)."""
    def part(t, k, ph, d, fn):
        # fn: 0 = sin, 1 = cos; derivative d
        idx = (fn + d) % 4
        val = [xp.sin, xp.cos, lambda z: -xp.sin(z), lambda z: -xp.cos(z)][idx](k * t + ph)
        return k ** d * val
    return part(x, K[0], PHASE[0], dx, 0) * part(y, K[1], PHASE[1], dy, 1)


def make_bc(kind: str, xp):
    if kind == "periodic":
        return BoundaryConditions.periodic(2)
    if kind == "dirichlet":
        return BoundaryConditions.uniform(2, Dirichlet(lambda x, y: exact(xp, x, y)))
    # Neumann conditions hold the outward gradient
    return (BoundaryConditions()
            .set("left", Neumann(lambda x, y: -exact(xp, x, y, dx=1)))
            .set("right", Neumann(lambda x, y: exact(xp, x, y, dx=1)))
            .set("bottom", Neumann(lambda x, y: -exact(xp, x, y, dy=1)))
            .set("top", Neumann(lambda x, y: exact(xp, x, y, dy=1))))


def field(n, kind, backend):
    # periodic: use an exactly periodic function on the unit square
    length = (2 * math.pi / K[0], 2 * math.pi / K[1]) if kind == "periodic" else (1.0, 1.0)
    grid = CartesianGrid((n, n), length=length, backend=backend)
    x, y = grid.meshgrid()
    return grid, CellField.from_interior(grid, exact(grid.xp, x, y), make_bc(kind, grid.xp))


NS = [16, 32, 64]


# Walls: compact closures (first two cells), f' 4th and f'' 3rd order at the wall cells (measured 4.05 / 3.05);
# the cross derivative d11 differentiates a derivative (extrapolating closure, f' 3rd): 2.96.
@pytest.mark.parametrize("kind,min_d1,min_d2,min_d11", [("periodic", 5.8, 5.8, 4.7), ("dirichlet", 3.9, 2.9, 2.8),
                                                        ("neumann", 3.9, 2.9, 2.8)])
def test_ccd_orders(backend, kind, min_d1, min_d2, min_d11):
    ccd = CCDCellDerivative()
    e1, e2, e11 = [], [], []
    for n in NS:
        grid, f = field(n, kind, backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        d1, d2 = ccd.derivatives(f, 0)
        e1.append(float(xp.abs(d1 - exact(xp, x, y, dx=1)).max()))
        e2.append(float(xp.abs(d2 - exact(xp, x, y, dx=2)).max()))
        e11.append(float(xp.abs(ccd.d11(f, 0, 1) - exact(xp, x, y, dx=1, dy=1)).max()))
    assert min(order(e1, NS)) > min_d1
    assert min(order(e2, NS)) > min_d2
    assert min(order(e11, NS)) > min_d11


def test_ccd_along_second_axis_and_3d(backend):
    ccd = CCDCellDerivative()
    errs = []
    for n in (8, 16, 32):
        grid = CartesianGrid((n, n, n), backend=backend)
        xp = grid.xp
        x, y, z = grid.meshgrid()
        f = CellField.from_interior(grid, xp.sin(x + 2 * y + 3 * z),
                                    BoundaryConditions.uniform(3, Dirichlet(lambda x, y, z: xp.sin(x + 2 * y + 3 * z))))
        errs.append(float(abs(ccd.d1(f, 2) - 3 * xp.cos(x + 2 * y + 3 * z)).max()))
    assert min(order(errs, [8, 16, 32])) > 3.9  # wall cells: 4th order (measured 4.00)


@pytest.mark.parametrize("kind", ["periodic", "dirichlet", "neumann"])
def test_fccd_face_operators_fourth_order(backend, kind):
    ops = FCCDFaceComplex()
    eg, ev, ec = [], [], []
    for n in NS:
        grid, f = field(n, kind, backend)
        xp = grid.xp
        xf, yf = grid.face_meshgrid(0)
        eg.append(float(xp.abs(ops.grad_f(f, 0) - exact(xp, xf, yf, dx=1)).max()))
        ev.append(float(xp.abs(ops.interp_f(f, 0) - exact(xp, xf, yf)).max()))
        x, y = grid.meshgrid()
        ec.append(float(xp.abs(ops.face_to_cell(exact(xp, xf, yf), 0, periodic=kind == "periodic")
                               - exact(xp, x, y)).max()))
    for errs in (eg, ev, ec):
        assert min(order(errs, NS)) > 3.8


def test_fccd_face_to_cell_requires_periodicity():
    grid = CartesianGrid((8, 8))
    with pytest.raises(ValueError):
        FCCDFaceComplex().face_to_cell(np.zeros(grid.face_shape(0)), 0)


def _consistent_poisson(ops, n, kind, backend="numpy"):
    """div(grad a) = D_f(exact face gradient): the setting of a projection."""
    pi = math.pi
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    ex = lambda x, y: xp.sin(pi * x) * xp.sin(2 * pi * y) + x * x * y  # noqa: E731
    gx = lambda x, y: pi * xp.cos(pi * x) * xp.sin(2 * pi * y) + 2 * x * y  # noqa: E731
    gy = lambda x, y: 2 * pi * xp.sin(pi * x) * xp.cos(2 * pi * y) + x * x  # noqa: E731
    if kind == "dirichlet":
        bc = BoundaryConditions.uniform(2, Dirichlet(ex))
    else:
        bc = (BoundaryConditions().set("left", Neumann(lambda x, y: -gx(x, y))).set("right", Neumann(gx))
              .set("bottom", Neumann(lambda x, y: -gy(x, y))).set("top", Neumann(gy)))
    b = ops.div_f(FaceField(grid, [gx(*grid.face_meshgrid(0)), gy(*grid.face_meshgrid(1))]))
    a = CellField(grid, bc=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = poisson(a, b, ops=ops, tolerance=1e-11, nitermax=400)
    x, y = grid.meshgrid()
    err = a.interior - ex(x, y)
    if kind == "neumann":
        err = err - err.mean()
    return float(abs(err).max()), stats


@pytest.mark.parametrize("kind", ["dirichlet", "neumann"])
def test_fccd_poisson_defect_correction_fourth_order(backend, kind):
    """Residual with FCCD, relaxation with fd2: the projection Poisson problem converges at 4th order."""
    ns_ = [16, 32, 64]
    fccd = [_consistent_poisson(FCCDFaceComplex(), n, kind, backend) for n in ns_]
    fd2 = [_consistent_poisson(FD2FaceComplex(), n, kind, backend) for n in ns_]
    assert all(s.converged for _, s in fccd)
    assert min(order([e for e, _ in fccd], ns_)) > 3.9
    assert max(order([e for e, _ in fd2], ns_)) < 2.1


def test_fccd_swaps_into_centered_ns(backend):
    """Taylor--Green with FCCD face operators (projection, centred gradient, face velocity)."""
    from cases import taylor_green

    errs = {}
    for name, ops in (("fd2", FD2FaceComplex()), ("fccd", FCCDFaceComplex())):
        ns = taylor_green.setup(32, backend)
        ns.ops = ops
        ns.start(events=[2.0])
        ns.run(2.0)
        errs[name] = taylor_green.error_norms(ns)
        assert ns.max_divergence() < 1e-4
    assert errs["fccd"][1] < errs["fd2"][1] / 3.0


def test_scheme_registry_builds_ccd():
    scheme = build_scheme({"grid": {"cells": [16, 16]}, "operators": {"face": "fccd", "cell": "ccd"}})
    assert scheme.face.name == "fccd" and scheme.cell.name == "ccd"


@pytest.mark.skipif(importlib.util.find_spec("sympy") is None, reason="sympy not installed")
def test_closure_coefficients_regenerate(tmp_path):
    """The committed coefficients are exactly what tools/derive_ccd_closures.py derives."""
    root = Path(__file__).resolve().parents[1]
    target = root / "basilisk_cupy" / "discretization" / "ccd_coefficients.py"
    before = target.read_text()
    try:
        subprocess.run([sys.executable, str(root / "tools" / "derive_ccd_closures.py")], check=True,
                       capture_output=True)
        assert target.read_text() == before
    finally:
        target.write_text(before)


def test_ccd_operators_match_on_backends(backend):
    grid, f = field(32, "dirichlet", backend)
    d1 = to_numpy(CCDCellDerivative().d1(f, 1))
    grid_np, f_np = field(32, "dirichlet", "numpy")
    np.testing.assert_allclose(d1, to_numpy(CCDCellDerivative().d1(f_np, 1)), rtol=1e-12, atol=1e-12)

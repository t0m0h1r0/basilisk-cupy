"""Step 4b: GCR-accelerated defect correction (FCCD residual, fd2 multigrid)."""

import math
import warnings

import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann, poisson
from basilisk_cupy.discretization import FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.backend import array_module
from basilisk_cupy.linear.defect_correction import defect_correction

PI = math.pi


def ex(x, y):
    xp = array_module(x)
    return xp.sin(PI * x) * xp.sin(2 * PI * y) + x * x * y


def gx(x, y):
    xp = array_module(x)
    return PI * xp.cos(PI * x) * xp.sin(2 * PI * y) + 2 * x * y


def gy(x, y):
    xp = array_module(x)
    return 2 * PI * xp.sin(PI * x) * xp.cos(2 * PI * y) + x * x


def problem(n, kind, ops, backend="numpy"):
    grid = CartesianGrid((n, n), backend=backend)
    if kind == "dirichlet":
        bc = BoundaryConditions.uniform(2, Dirichlet(ex))
    else:
        bc = (BoundaryConditions().set("left", Neumann(lambda x, y: -gx(x, y))).set("right", Neumann(gx))
              .set("bottom", Neumann(lambda x, y: -gy(x, y))).set("top", Neumann(gy)))
    b = ops.div_f(FaceField(grid, [gx(*grid.face_meshgrid(0)), gy(*grid.face_meshgrid(1))]))
    return grid, CellField(grid, bc=bc), b


def error(grid, a, kind):
    x, y = grid.meshgrid()
    e = a.interior - ex(x, y)
    if kind == "neumann":
        e = e - e.mean()
    return float(abs(e).max())


@pytest.mark.parametrize("kind", ["dirichlet", "neumann"])
def test_iterations_independent_of_resolution(backend, kind):
    ops = FCCDFaceComplex()
    its = []
    for n in (32, 64, 128):
        grid, a, b = problem(n, kind, ops, backend)
        stats = defect_correction(a, b, ops=ops, tolerance=1e-10)
        assert stats.converged
        its.append(stats.i)
    assert max(its) <= 22 and max(its) - min(its) <= 4


def test_much_faster_than_plain_defect_correction_on_dirichlet_walls(backend):
    ops = FCCDFaceComplex()
    grid, a, b = problem(64, "dirichlet", ops, backend)
    gcr = poisson(a, b, ops=ops, tolerance=1e-10, method="dc")
    grid, a2, b = problem(64, "dirichlet", ops, backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = poisson(a2, b, ops=ops, tolerance=1e-10, method="mg", nitermax=400)
    assert gcr.converged and plain.converged
    assert gcr.nrelax < plain.i / 3  # multigrid cycles: ~34 against ~164
    assert error(grid, a, "dirichlet") == pytest.approx(error(grid, a2, "dirichlet"), rel=1e-3)


def test_line_search_variant_converges(backend):
    """memory = 1 is TwoPhaseFlow's minimal-residual line-search defect correction."""
    ops = FCCDFaceComplex()
    grid, a, b = problem(64, "dirichlet", ops, backend)
    hist = []
    stats = defect_correction(a, b, ops=ops, tolerance=1e-10, memory=1, history=hist)
    assert stats.converged and stats.i < 40
    assert all(h1 <= 2.0 * h0 for h0, h1 in zip(hist, hist[1:]))


def test_auto_method_keeps_basilisk_multigrid_for_fd2():
    ops = FD2FaceComplex()
    grid, a, b = problem(32, "dirichlet", ops)
    stats = poisson(a, b, ops=ops, tolerance=1e-8)
    assert stats.converged and stats.nrelax >= 1 and stats.i < 30


def test_helmholtz_with_lambda(backend):
    """div(grad a) - a = b with Dirichlet walls."""
    ops = FCCDFaceComplex()
    errs = []
    for n in (16, 32, 64):
        grid, a, b = problem(n, "dirichlet", ops, backend)
        x, y = grid.meshgrid()
        stats = poisson(a, b - ex(x, y), lam=-1.0, ops=ops, tolerance=1e-11)
        assert stats.converged
        errs.append(error(grid, a, "dirichlet"))
    assert math.log2(errs[1] / errs[2]) > 3.8


def test_fccd_taylor_green_uses_defect_correction_without_warnings(backend):
    from cases import taylor_green
    from basilisk_cupy.discretization import FCCDFaceComplex as F

    ns = taylor_green.setup(32, backend)
    ns.ops = F()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ns.start(events=[0.5])
        ns.run(0.5)
    assert ns.mgp.converged and ns.max_divergence() < 1e-4

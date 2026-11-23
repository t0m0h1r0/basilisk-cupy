"""Poisson--Helmholtz solver: convergence order and robustness (milestone M1)."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann, poisson
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.linear.multigrid import prolong_bilinear, restrict_cell

from .conftest import order

PI = math.pi


def _solve_dirichlet_sine(n, backend, ndim=2):
    grid = CartesianGrid((n,) * ndim, backend=backend)
    xp = grid.xp
    mesh = grid.meshgrid()
    exact = 1.0
    for c in mesh:
        exact = exact * xp.sin(PI * c)
    b = -ndim * PI**2 * exact
    a = CellField(grid, bc=BoundaryConditions.uniform(ndim, Dirichlet(0.0)), name="a")
    stats = poisson(a, b, tolerance=1e-10 * n**2)
    assert stats.converged
    return float(abs(a.interior - exact).max()), stats


def test_dirichlet_second_order_2d(backend):
    ns = [16, 32, 64, 128]
    errs = [_solve_dirichlet_sine(n, backend)[0] for n in ns]
    assert min(order(errs, ns)) > 1.9


def test_dirichlet_second_order_3d(backend):
    ns = [8, 16, 32]
    errs = [_solve_dirichlet_sine(n, backend, ndim=3)[0] for n in ns]
    assert min(order(errs, ns)) > 1.9


def test_multigrid_converges_in_few_cycles(backend):
    _, stats = _solve_dirichlet_sine(128, backend)
    assert stats.i <= 15


def test_periodic_zero_mean(backend):
    ns = [16, 32, 64]
    errs = []
    for n in ns:
        grid = CartesianGrid((n, n), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        exact = xp.sin(2 * PI * x) * xp.cos(4 * PI * y)
        b = -20 * PI**2 * exact
        a = CellField(grid, bc=BoundaryConditions.periodic(2))
        poisson(a, b, tolerance=1e-9 * n**2)
        sol = a.interior - a.interior.mean()
        errs.append(float(abs(sol - exact).max()))
    assert min(order(errs, ns)) > 1.9


def test_helmholtz_with_lambda(backend):
    n = 64
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    exact = xp.cos(PI * x) * xp.cos(PI * y)
    lam = -3.0
    b = (-2 * PI**2 + lam) * exact
    a = CellField(grid, bc=BoundaryConditions.uniform(2, Neumann(0.0)))
    stats = poisson(a, b, lam=lam, tolerance=1e-8)
    assert stats.converged
    assert float(abs(a.interior - exact).max()) < 5e-4


@pytest.mark.parametrize("ratio", [10.0, 1000.0])
def test_variable_coefficient_density_ratio(backend, ratio):
    """div(alpha grad a) = b with alpha = 1/rho and a smooth density jump of ``ratio``."""
    w = 0.05

    def fields(grid, x):
        xp = grid.xp
        H = 0.5 * (1 + xp.tanh((x - 0.5) / w))
        rho = 1 + (ratio - 1) * H
        dH = 0.5 / w / xp.cosh((x - 0.5) / w) ** 2
        alpha = 1 / rho
        dalpha = -(ratio - 1) * dH / rho**2
        return alpha, dalpha

    ns = [32, 64, 128, 256]
    errs = []
    for n in ns:
        grid = CartesianGrid((n, n), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        exact = xp.sin(PI * x) * xp.sin(PI * y)
        al, dal = fields(grid, x)
        # div(alpha grad a) = alpha lap a + alpha' a_x
        b = al * (-2 * PI**2 * exact) + dal * PI * xp.cos(PI * x) * xp.sin(PI * y)
        alpha = FaceField(grid, [fields(grid, grid.face_meshgrid(0)[0])[0],
                                 fields(grid, grid.face_meshgrid(1)[0])[0]])
        a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
        stats = poisson(a, b, alpha, tolerance=1e-9)
        assert stats.converged, stats
        errs.append(float(abs(a.interior - exact).max()))
    assert order(errs, ns)[-1] > 1.9


def test_transfer_operators_preserve_linear_fields():
    grid = CartesianGrid((8, 8))
    fine = CartesianGrid((16, 16))
    x, y = grid.meshgrid(with_ghosts=True)
    coarse = CellField(grid, data=2 * x - 3 * y + 1)
    xf, yf = fine.meshgrid()
    np.testing.assert_allclose(to_numpy(prolong_bilinear(coarse)), to_numpy(2 * xf - 3 * yf + 1), atol=1e-12)
    np.testing.assert_allclose(to_numpy(restrict_cell(2 * xf - 3 * yf + 1)), to_numpy(coarse.interior), atol=1e-12)


def test_non_power_of_two_grid(backend):
    grid = CartesianGrid((48, 48), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    exact = xp.sin(PI * x) * xp.sin(PI * y)
    a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
    stats = poisson(a, -2 * PI**2 * exact, tolerance=1e-8)
    assert stats.converged and stats.minlevel == 3

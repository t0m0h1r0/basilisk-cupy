import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann, Periodic
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FD2CellDerivative, FD2FaceComplex
from basilisk_cupy.grid import build_levels

from .conftest import order


def test_dirichlet_ghost_gives_face_value(backend):
    grid = CartesianGrid((8, 8), backend=backend)
    bc = BoundaryConditions.uniform(2, Neumann(0.0)).set("left", Dirichlet(3.0))
    f = CellField.from_interior(grid, grid.meshgrid()[0], bc)
    face_value = to_numpy(FD2FaceComplex().interp_f(f, 0))[0]
    np.testing.assert_allclose(face_value, 3.0)


def test_neumann_ghost_gives_face_gradient(backend):
    grid = CartesianGrid((8, 8), backend=backend)
    bc = BoundaryConditions.uniform(2, Neumann(0.0)).set("top", Neumann(2.5))
    f = CellField.from_interior(grid, grid.meshgrid()[1] ** 2, bc)
    grad = to_numpy(FD2FaceComplex().grad_f(f, 1))[:, -1]
    np.testing.assert_allclose(grad, 2.5)


def test_dirichlet_callable_value(backend):
    grid = CartesianGrid((8, 8), backend=backend)
    bc = BoundaryConditions().set("bottom", Dirichlet(lambda x, y: x))
    f = CellField.from_interior(grid, 0.0, bc)
    np.testing.assert_allclose(to_numpy(FD2FaceComplex().interp_f(f, 1))[:, 0], to_numpy(grid.centers(0)))


def test_periodic_ghosts_with_wide_stencil(backend):
    grid = CartesianGrid((6, 5), ghosts=2, backend=backend)
    f = CellField.from_interior(grid, grid.xp.arange(30.0).reshape(6, 5), BoundaryConditions.periodic(2))
    d = to_numpy(f.data)
    # padded index 2..7 is the interior along axis 0; ghosts are 0, 1 and 8, 9
    np.testing.assert_array_equal(d[0, 2:-2], d[6, 2:-2])
    np.testing.assert_array_equal(d[1, 2:-2], d[7, 2:-2])
    np.testing.assert_array_equal(d[8, 2:-2], d[2, 2:-2])
    np.testing.assert_array_equal(d[9, 2:-2], d[3, 2:-2])
    np.testing.assert_array_equal(d[2:-2, 1], d[2:-2, 6])


@pytest.mark.parametrize("axis", [0, 1])
def test_fd2_face_gradient_second_order(backend, axis):
    errs, ns = [], [16, 32, 64]
    for n in ns:
        grid = CartesianGrid((n, n), backend=backend)
        x, y = grid.meshgrid()
        f = CellField.from_interior(grid, grid.xp.sin(2 * math.pi * x) * grid.xp.cos(2 * math.pi * y),
                                    BoundaryConditions.periodic(2))
        xf, yf = grid.face_meshgrid(axis)
        exact = (2 * math.pi * grid.xp.cos(2 * math.pi * xf) * grid.xp.cos(2 * math.pi * yf) if axis == 0
                 else -2 * math.pi * grid.xp.sin(2 * math.pi * xf) * grid.xp.sin(2 * math.pi * yf))
        errs.append(float(abs(FD2FaceComplex().grad_f(f, axis) - exact).max()))
    assert min(order(errs, ns)) > 1.9


def test_fd2_divergence_of_face_field(backend):
    grid = CartesianGrid((32, 16), length=(2.0, 1.0), backend=backend)
    uf = FaceField(grid)
    uf[0] = grid.face_meshgrid(0)[0] ** 2
    uf[1] = 3.0 * grid.face_meshgrid(1)[1]
    x, _ = grid.meshgrid()
    np.testing.assert_allclose(to_numpy(FD2FaceComplex().div_f(uf)), to_numpy(2 * x + 3.0), atol=1e-12)


def test_cell_derivatives(backend):
    grid = CartesianGrid((64, 64), backend=backend)
    x, y = grid.meshgrid()
    xp = grid.xp
    f = CellField.from_interior(grid, xp.sin(2 * math.pi * x) * xp.sin(2 * math.pi * y), BoundaryConditions.periodic(2))
    ops = FD2CellDerivative()
    k = 2 * math.pi
    assert float(abs(ops.d1(f, 0) - k * xp.cos(k * x) * xp.sin(k * y)).max()) < 1e-2 * k
    assert float(abs(ops.d2(f, 1) + k * k * xp.sin(k * x) * xp.sin(k * y)).max()) < 1e-2 * k * k
    assert float(abs(ops.d11(f, 0, 1) - k * k * xp.cos(k * x) * xp.cos(k * y)).max()) < 2e-2 * k * k


def test_levels_stop_at_two_cells():
    levels = build_levels(CartesianGrid((64, 32)))
    assert [g.n for g in levels] == [(4, 2), (8, 4), (16, 8), (32, 16), (64, 32)]

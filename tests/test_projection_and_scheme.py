import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField, build_scheme, project
from basilisk_cupy.backend import cupy_available, to_numpy
from basilisk_cupy.discretization import FD2FaceComplex
from basilisk_cupy.scheme import available, register


@pytest.mark.parametrize("periodic", [True, False])
def test_projection_makes_face_velocity_divergence_free(backend, periodic):
    grid = CartesianGrid((64, 64), backend=backend)
    xp = grid.xp
    rng = np.random.default_rng(0)
    uf = FaceField(grid, [xp.asarray(rng.standard_normal(grid.face_shape(a))) for a in range(2)])
    if not periodic:  # walls: no flux through the boundary faces
        uf[0][0, :] = uf[0][-1, :] = 0.0
        uf[1][:, 0] = uf[1][:, -1] = 0.0
    else:
        uf[0][-1, :] = uf[0][0, :]
        uf[1][:, -1] = uf[1][:, 0]
    bc = BoundaryConditions.periodic(2) if periodic else BoundaryConditions()
    p = CellField(grid, bc=bc, name="p")
    dt = 0.01
    stats = project(uf, p, dt=dt, tolerance=1e-8)
    assert stats.converged
    div = FD2FaceComplex().div_f(uf)
    assert float(abs(div).max()) * dt < 1e-8 * 1.01
    if not periodic:
        assert float(abs(uf[0][0, :]).max()) == 0.0


def test_projection_with_density_ratio(backend):
    grid = CartesianGrid((64, 64), backend=backend)
    xp = grid.xp
    rho_face = [1 + 999.0 * (grid.face_meshgrid(a)[1] > 0.5) for a in range(2)]
    alpha = FaceField(grid, [1.0 / r for r in rho_face])
    rng = np.random.default_rng(1)
    uf = FaceField(grid, [xp.asarray(rng.standard_normal(grid.face_shape(a))) for a in range(2)])
    uf[0][0, :] = uf[0][-1, :] = 0.0
    uf[1][:, 0] = uf[1][:, -1] = 0.0
    p = CellField(grid, bc=BoundaryConditions(), name="p")
    stats = project(uf, p, alpha, dt=1.0, tolerance=1e-8)
    assert stats.converged
    assert float(abs(FD2FaceComplex().div_f(uf)).max()) < 1.01e-8


def test_scheme_from_config_and_registry():
    scheme = build_scheme({"grid": {"cells": [32, 16], "length": [2.0, 1.0]}, "operators": {"face": "fd2"}})
    assert scheme.grid.n == (32, 16) and scheme.face.name == "fd2"
    with pytest.raises(ValueError, match="unknown 'face' implementation"):
        build_scheme({"operators": {"face": "ccd"}})

    class Wide(FD2FaceComplex):
        name = "wide"
        stencil_ghosts = 3

    register("face", "wide", Wide)
    assert "wide" in available("face")
    assert build_scheme({"operators": {"face": "wide"}}).grid.ghosts == 3


@pytest.mark.skipif(not cupy_available(), reason="cupy not installed")
def test_numpy_and_cupy_agree():
    from basilisk_cupy import Dirichlet, poisson

    out = []
    for backend in ("numpy", "cupy"):
        grid = CartesianGrid((64, 64), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
        poisson(a, xp.sin(3 * x) * xp.cos(2 * y), tolerance=1e-10)
        out.append(to_numpy(a.interior))
    np.testing.assert_allclose(out[0], out[1], rtol=1e-9, atol=1e-12)

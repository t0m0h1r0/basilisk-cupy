"""Finite-volume transport fluxes (``FaceComplex.wall_free_flux``): zero on walls, same divergence.

FCCD's conservative flux ``F - h^2/24 d_nn F`` is not zero on an impermeable wall
(``-h^2/24 d_nn u_n``); a tracer advected with it leaks through the walls. ``wall_free_flux``
adds the curl of a boundary stream function (Coons-extended): zero on walls, no change of the
discrete divergence, an ``O(h^2)`` change inside.
"""

import math
import warnings

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField, Neumann
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic
from basilisk_cupy.discretization import FCCDFaceComplex, FD2FaceComplex
from basilisk_cupy.grid.mapping import TanhMapping
from basilisk_cupy.projection.approximate import project

from .conftest import order


def _projected(n, periodic, maps, backend):
    """A smooth face field (zero normal velocity on walls) projected with FCCD."""
    grid = CartesianGrid((n, n), mapping=maps, backend=backend)
    xp = grid.xp
    x0, y0 = grid.face_meshgrid(0)
    x1, y1 = grid.face_meshgrid(1)
    u = xp.cos(math.pi * y0) ** 2 * (1 + 0.5 * xp.sin(2 * math.pi * x0)) + 0.2 * xp.cos(3 * y0 + 2 * math.pi * x0)
    v = 0.3 * xp.sin(2 * math.pi * x1) * xp.cos(math.pi * y1) ** 2 + 0.1 * xp.sin(2 * math.pi * x1 - y1)
    if not periodic[0]:
        u[0], u[-1] = 0.0, 0.0
    if not periodic[1]:
        v[:, 0], v[:, -1] = 0.0, 0.0
    uf = FaceField(grid, [u, v])
    bc = BoundaryConditions()
    for a, side in ((0, "left"), (0, "right"), (1, "bottom"), (1, "top")):
        bc = bc.set(side, Periodic() if periodic[a] else Neumann(0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project(uf, CellField(grid, bc=bc), None, 1.0, ops=FCCDFaceComplex(), tolerance=1e-12)
    return grid, uf


def _sizes(grid):
    xp = grid.xp
    return [xp.full(grid.n[a], float(grid.h[a])) if grid.axis_uniform(a) else float(grid.h[a]) / grid.metric(a).J
            for a in range(2)]


def _volume_divergence(grid, f):
    sx, sy = _sizes(grid)
    return (f[0][1:] - f[0][:-1]) * sy[None, :] + (f[1][:, 1:] - f[1][:, :-1]) * sx[:, None]


CASES = [((True, False), None), ((False, False), None), ((False, False), (TanhMapping(2.0), TanhMapping(1.5)))]


@pytest.mark.parametrize("periodic,maps", CASES)
def test_zero_on_walls_and_same_divergence(backend, periodic, maps):
    grid, uf = _projected(32, periodic, maps, backend)
    ops = FCCDFaceComplex()
    hat = [ops.face_flux(uf[a], a, periodic[a]) for a in range(2)]
    phi = ops.wall_free_flux(uf, periodic)
    wall_hat = float(abs(to_numpy(hat[1][:, 0])).max())
    assert wall_hat > 1e-6  # the corrected flux is not zero on the wall
    assert float(abs(to_numpy(phi[1][:, 0])).max()) == 0.0 and float(abs(to_numpy(phi[1][:, -1])).max()) == 0.0
    if not periodic[0]:
        assert float(abs(to_numpy(phi[0][0])).max()) == 0.0 and float(abs(to_numpy(phi[0][-1])).max()) == 0.0
    diff = _volume_divergence(grid, phi) - _volume_divergence(grid, hat)
    assert float(abs(to_numpy(diff)).max()) < 1e-12 * float(grid.h[0])


def test_change_inside_is_second_order(backend):
    """Channel, faces off the walls: measured 5.6e-4 / 1.4e-4 / 3.5e-5 (N = 32 / 64 / 128)."""
    ns, change = [32, 64, 128], []
    for n in ns:
        grid, uf = _projected(n, (True, False), None, backend)
        ops = FCCDFaceComplex()
        phi = ops.wall_free_flux(uf, (True, False))
        hat = ops.face_flux(uf[1], 1, False)
        change.append(float(abs(to_numpy((phi[1] - hat)[:, 1:-1])).max()))
    assert min(order(change, ns)) > 1.8


def _tracer_mass_change(grid, flux, steps=200):
    """First-order upwind finite-volume transport (periodic in x, walls in y) of a tracer next to the wall."""
    xp = grid.xp
    x, y = grid.meshgrid()
    c = xp.exp(-((y - 0.05) / 0.05) ** 2) * (1 + 0.5 * xp.sin(2 * math.pi * x))
    sx, sy = _sizes(grid)
    vol = sx[:, None] * sy[None, :]
    qx, qy = flux[0] * sy[None, :], flux[1] * sx[:, None]
    m0 = float((c * vol).sum())
    dt = 0.2 * float(grid.h[0])
    for _ in range(steps):
        fx = xp.where(qx > 0, xp.concatenate([c[-1:], c], 0), xp.concatenate([c, c[:1]], 0)) * qx
        fy = xp.where(qy > 0, xp.concatenate([c[:, :1], c], 1), xp.concatenate([c, c[:, -1:]], 1)) * qy
        c = c - dt * ((fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1])) / vol
    return float((c * vol).sum()) / m0 - 1.0


def test_tracer_next_to_a_wall_is_conserved(backend):
    grid, uf = _projected(32, (True, False), None, backend)
    ops = FCCDFaceComplex()
    hat = FaceField(grid, [ops.face_flux(uf[a], a, (True, False)[a]) for a in range(2)])
    leak = _tracer_mass_change(grid, hat)
    kept = _tracer_mass_change(grid, ops.wall_free_flux(uf, (True, False)))
    assert abs(leak) > 1e-6 and abs(kept) < 1e-13  # measured 9.9e-5 and 2.2e-16


def test_fd2_returns_the_face_velocity(backend):
    grid, uf = _projected(16, (True, False), None, backend)
    phi = FD2FaceComplex().wall_free_flux(uf, (True, False))
    for a in range(2):
        np.testing.assert_array_equal(to_numpy(phi[a]), to_numpy(uf[a]))


def test_three_dimensions_refused():
    grid = CartesianGrid((8, 8, 8))
    with pytest.raises(NotImplementedError):
        FCCDFaceComplex().wall_free_flux(FaceField(grid), (False, False, False))

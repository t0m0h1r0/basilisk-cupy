"""Geometric VOF (interface/): geometry, fractions, advection against Basilisk ``src/test/reversed.c``."""

import math

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction, mycs, reconstruction
from basilisk_cupy.interface.geometry import line_alpha, line_area, rectangle_fraction
from cases import reversed_vof


def test_line_alpha_inverts_line_area():
    rng = np.random.default_rng(0)
    nx, ny = rng.uniform(-1, 1, 1000), rng.uniform(-1, 1, 1000)
    s = np.abs(nx) + np.abs(ny)
    nx, ny = nx / s, ny / s
    c = rng.uniform(0, 1, 1000)
    np.testing.assert_allclose(line_area(nx, ny, line_alpha(c, nx, ny)), c, atol=1e-12)


def test_rectangle_fraction_of_the_whole_cell_is_the_volume_fraction():
    rng = np.random.default_rng(1)
    nx, ny = rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100)
    s = np.abs(nx) + np.abs(ny)
    nx, ny = nx / s, ny / s
    c = rng.uniform(0, 1, 100)
    alpha = line_alpha(c, nx, ny)
    np.testing.assert_allclose(rectangle_fraction(nx, ny, alpha, -0.5, -0.5, 0.5, 0.5), c, atol=1e-12)
    # half of a vertical interface (nx = 1): the left half of the cell is full
    assert float(rectangle_fraction(np.array(1.0), np.array(0.0), np.array(0.0), -0.5, -0.5, 0.0, 0.5)) == 1.0


def test_mycs_on_a_straight_line(backend):
    """30-degree line: MYC is exact where the centred scheme applies and close elsewhere
    (measured: largest error 2.3e-3 over the cut cells)."""
    grid = CartesianGrid((16, 16), backend=backend)
    xp = grid.xp
    nx0, ny0 = math.cos(math.pi / 6), math.sin(math.pi / 6)
    c = fraction(grid, lambda x, y: 0.5 * nx0 + 0.45 * ny0 - (nx0 * x + ny0 * y))  # inside: n.x < const
    cp = xp.pad(c, 1, mode="edge")
    nx, ny = mycs(cp)
    cut = (c > 1e-3) & (c < 1 - 1e-3)
    s = nx0 + ny0
    inner = cut.copy()
    inner[0], inner[-1], inner[:, 0], inner[:, -1] = False, False, False, False
    err = xp.maximum(xp.abs(nx - nx0 / s), xp.abs(ny - ny0 / s))[inner]
    assert float(err.max()) < 5e-3 and float((err < 1e-12).mean()) > 0.5
    _, _, alpha = reconstruction(cp)
    assert bool(xp.all(alpha[~cut & ((c <= 0) | (c >= 1))] == 0.0))


@pytest.mark.parametrize("n", [32, 64, 128])
def test_fractions_reproduce_basilisk_volume_sums(backend, n):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    f = fraction(grid, reversed_vof.circle)
    total = float(f.sum()) * float(grid.h[0] * grid.h[1])
    assert total == pytest.approx(reversed_vof.REFERENCE[n][0], abs=5e-13)
    assert float(f.min()) == 0.0 and float(f.max()) == 1.0


def test_reversed_vortex_matches_basilisk(backend):
    """N = 32, 963 steps: sums, error norms (reversed.cref, constant grid) to the printed digits."""
    r = reversed_vof.run(32, backend)
    total, avg, rms, emax = reversed_vof.REFERENCE[32]
    assert r["sum0"] == pytest.approx(total, abs=5e-13) and r["sum"] == pytest.approx(total, abs=5e-13)
    assert f"{r['avg']:.6g}" == f"{avg:.6g}" and f"{r['rms']:.6g}" == f"{rms:.6g}" and r["emax"] == pytest.approx(emax)
    assert r["min"] > -1e-15 and r["max"] < 1 + 1e-15
    assert r["cdev"] < 1e-12  # the VOF concentration stays constant
    assert r["cfl"] <= 0.5 + 1e-6


@pytest.mark.slow
def test_reversed_vortex_finer_grids_match_basilisk(backend):
    for n in (64, 128):
        r = reversed_vof.run(n, backend)
        total, avg, rms, emax = reversed_vof.REFERENCE[n]
        assert r["sum"] == pytest.approx(total, abs=5e-13)
        assert f"{r['avg']:.6g}" == f"{avg:.6g}" and f"{r['rms']:.6g}" == f"{rms:.6g}" and f"{r['emax']:.6g}" == f"{emax:.6g}"


def test_numpy_and_cupy_agree():
    pytest.importorskip("cupy")
    a = reversed_vof.run(32, "numpy")
    b = reversed_vof.run(32, "cupy")
    assert abs(a["avg"] - b["avg"]) < 1e-12 and a["steps"] == b["steps"]
    assert to_numpy is not None

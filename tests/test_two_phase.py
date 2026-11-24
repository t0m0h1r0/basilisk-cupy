"""M4b: two-phase properties, height functions, interfacial force and reduced gravity (Basilisk
``two-phase.h``, ``heights.h``, ``curvature.h`` position(), ``iforce.h``, ``reduced.h``)."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA, heights, position
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

from .conftest import order


def test_position_of_a_flat_interface_is_exact(backend):
    grid = CartesianGrid((16, 16), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
    f = fraction(grid, lambda x, y: 0.3123 - y + 0 * x)
    pos = position(f, BoundaryConditions(), grid, (0.0, 1.0))
    found = pos != NODATA
    assert int(found.sum()) >= 16
    assert float(abs(to_numpy(pos[found]) - 0.3123).max()) < 1e-14


def test_position_is_second_order(backend):
    """eta = 0.01 cos(2 pi x) + 0.1 sin(pi x): 2.6e-3 / 6.7e-4 / 1.7e-4 / 4.2e-5 (N = 16 .. 128)."""
    errs, ns = [], [16, 32, 64, 128]
    for n in ns:
        grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
        xp = grid.xp
        eta = lambda x: 0.01 * xp.cos(2 * math.pi * x) + 0.1 * xp.sin(math.pi * x)  # noqa: E731
        f = fraction(grid, lambda x, y: eta(x) - y)
        pos = position(f, BoundaryConditions(), grid, (0.0, 1.0))
        x, _ = grid.meshgrid()
        found = pos != NODATA
        errs.append(float(abs(to_numpy((pos - eta(x))[found])).max()))
    assert min(order(errs, ns)) > 1.9


def test_gpu_column_propagation_is_bitwise_the_array_path():
    cp = pytest.importorskip("cupy")
    for n in (16, 64):
        grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
        f = fraction(grid, lambda x, y: 0.3 * np.cos(2 * np.pi * x) * np.sin(3 * x) - y)
        for a, b in zip(heights(f, BoundaryConditions()), heights(cp.asarray(f), BoundaryConditions())):
            np.testing.assert_array_equal(a, cp.asnumpy(b))


def test_properties(backend):
    grid = CartesianGrid((8, 8), backend=backend)
    xp = grid.xp
    ns = CenteredNS(grid)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.4 - y + 0 * x), BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=2.0, rho2=0.5, mu1=0.3, mu2=0.1)
    tp._properties(ns)
    fi = to_numpy(f.interior)
    np.testing.assert_allclose(to_numpy(ns.rho), np.clip(fi, 0, 1) * 1.5 + 0.5)
    ff = (fi[:, 1:] + fi[:, :-1]) / 2  # interior y-faces
    np.testing.assert_allclose(to_numpy(ns.alpha[1])[:, 1:-1], 1.0 / (ff * 1.5 + 0.5))
    np.testing.assert_allclose(to_numpy(ns.mu[1])[:, 1:-1], ff * 0.2 + 0.1)
    assert ns.cfl == 0.5 and xp is grid.xp


def test_flat_interface_stays_at_rest_under_reduced_gravity(backend):
    """The interfacial force [rho] G y grad f is balanced by the pressure jump exactly."""
    grid = CartesianGrid((16, 16), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
    xp = grid.xp
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-10)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.0313 + 0 * x), BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=0.1, mu1=0.01, mu2=0.01)
    tp.add_reduced_gravity(G=(0.0, 50.0))
    ns.DT = 0.002
    ns.start()
    for _ in range(20):
        ns.step()
    assert max(float(xp.max(xp.abs(c.interior))) for c in ns.u) < 1e-9


def _gravity_backend():
    try:
        import cupy  # noqa: F401, PLC0415

        return "cupy"
    except ImportError:
        return "numpy"


def test_gravity_wave_matches_basilisk():
    """Basilisk src/test/gravity.c, N = 16 (8 points per wavelength): 0.126721 against gravity.ref 0.126713."""
    from cases import gravity_wave

    r = gravity_wave.run(16, _gravity_backend())
    assert r["error"] == pytest.approx(gravity_wave.REFERENCE[8], rel=2e-4) and r["steps"] == 738


@pytest.mark.slow
def test_gravity_wave_convergence_matches_basilisk():
    """N = 32 / 64 / 128: 0.0317969 / 0.00809545 / 0.00455989 against 0.031758 / 0.00807759 / 0.00455815."""
    from cases import gravity_wave

    for n in (32, 64, 128):
        r = gravity_wave.run(n, _gravity_backend())
        assert r["error"] == pytest.approx(gravity_wave.REFERENCE[n // 2], rel=3e-3)

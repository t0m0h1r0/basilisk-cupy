"""M4c: height-function curvature and surface tension (Basilisk ``curvature.h``, ``parabola.h``, ``tension.h``).

The references in ``tests/data`` were printed by Basilisk itself (qcc built from the
comphy-lab/basilisk-C mirror, multigrid, ``%.17g``).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.constants import PI
from basilisk_cupy.interface.curvature import curvature
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

DATA = Path(__file__).parent / "data"

SHAPES = {  # name: (n, length, origin, level set); Basilisk's statistics (h, f, a, c) in the files
    "wave16": (16, 2.0, (0.0, -1.0), lambda x, y: y - 0.01 * np.cos(2.0 * PI * x)),
    "circle16": (16, 1.0, (-0.5, -0.5), lambda x, y: 0.15 ** 2 - (x - 0.02) ** 2 - (y - 0.03) ** 2),
    "small32": (32, 1.0, (-0.5, -0.5), lambda x, y: 0.05 ** 2 - (x - 0.013) ** 2 - (y + 0.021) ** 2),
    "ellipse32": (32, 1.0, (-0.5, -0.5), lambda x, y: 1.0 - ((x - 0.01) / 0.3) ** 2 - ((y + 0.02) / 0.12) ** 2),
    "tiny16": (16, 1.0, (-0.5, -0.5), lambda x, y: 0.06 ** 2 - (x - 0.013) ** 2 - (y + 0.021) ** 2),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_curvature_is_basilisks_bitwise(name):
    """All four methods occur: wave16 (HF 19), circle16 (HF 6, fit 6, average 6), small32 (fit 7,
    average 5), ellipse32 (HF 53, average 1), tiny16 (centroids 8)."""
    n, L, o, fn = SHAPES[name]
    lines = (DATA / f"curvature_basilisk_{name}.txt").read_text().splitlines()
    ref_stats = [int(v) for v in lines[0].split()[1:]]
    ref_c, ref_k = np.zeros((n, n)), np.zeros((n, n))
    for line in lines[1:]:
        i, j, c, k = line.split()
        ref_c[int(i), int(j)], ref_k[int(i), int(j)] = float(c), float(k)
    grid = CartesianGrid((n, n), length=(L, L), origin=o)
    c = fraction(grid, fn)
    stats = {}
    k = curvature(c, BoundaryConditions(), grid, stats=stats)
    np.testing.assert_array_equal(c, ref_c)
    np.testing.assert_array_equal(np.where(k == NODATA, 1e30, k), np.where(ref_k > 1e29, 1e30, ref_k))
    assert [stats[s] for s in "hfac"] == ref_stats


def test_curvature_of_a_circle_converges(backend):
    """Height-function curvature of a circle (R = 0.2) from accurate volume fractions (``fraction``
    on a 16 times finer grid, averaged): relative error over the cut cells at N = 32 / 64 / 128.
    ``fraction`` itself is only O(h^2) accurate, which leaves an O(1) curvature error (Basilisk's
    curvature.c uses exact fractions from Vofi)."""
    errs = []
    for n in (32, 64, 128):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
        xp = grid.xp
        s = 16
        fine = fraction(CartesianGrid((n * s, n * s), origin=(-0.5, -0.5), backend=backend),
                        lambda x, y: 0.04 - (x - 0.013) ** 2 - (y + 0.021) ** 2)
        c = xp.ascontiguousarray(fine.reshape(n, s, n, s).mean(axis=(1, 3)))
        k = curvature(c, BoundaryConditions(), grid)
        cut = (c > 0.0) & (c < 1.0)  # as Basilisk's curvature.c (interfacial full/empty cells excluded)
        errs.append(float(abs(to_numpy(k[cut]) * 0.2 - 1.0).max()))
    print(errs)
    assert errs[-1] < errs[0] / 3 and errs[-1] < 0.01


def test_first_step_of_the_capillary_wave_is_basilisks():
    """capwave.c, N = 16, after the first step (red/black): every field to round-off."""
    import cases.capillary_wave as cw

    ref = {"C": {}, "X": {}, "Y": {}}
    for line in (DATA / "capwave_step1_basilisk_redblack.txt").read_text().splitlines():
        w = line.split()
        if w[0] in ref:
            ref[w[0]][(int(w[1]), int(w[2]))] = [float(v) for v in w[3:]]
        elif w[0] == "mgp":
            it, resa = int(w[1]), float(w[3])
        elif w[0] == "dt":
            dt = float(w[1])
    n = 16
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-6)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.01 * np.cos(2.0 * PI * x)),
                                BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0, mu1=cw.MU, mu2=cw.MU, vof_first=True)
    tp.add_surface_tension(1.0)
    ns.start(events=cw.event_times()[1:])
    ns.step()
    assert ns.clock.tnext == pytest.approx(dt, rel=1e-15)  # the first step keeps init's dt (0.0022667)
    phi = curvature(f.interior, f.bc, grid, 1.0)
    for k, field in ((0, f.interior), (1, phi), (3, ns.u[0].interior), (4, ns.u[1].interior)):
        a = np.array([[ref["C"][(i, j)][k] for j in range(n)] for i in range(n)])
        assert float(np.abs(a - field).max()) < 1e-15
    for ax, key in ((0, "X"), (1, "Y")):
        for k, field in ((1, ns.a[ax]), (2, ns.uf[ax])):
            a = np.zeros(field.shape)
            for (i, j), v in ref[key].items():
                a[i, j] = v[k]
            assert float(np.abs(a - field).max()) < 1e-15
    assert ns.mgp.i == it and ns.mgp.resa == pytest.approx(resa, rel=1e-12)


def _gpu_or_numpy():
    try:
        import cupy  # noqa: F401, PLC0415

        return "cupy"
    except ImportError:
        return "numpy"


@pytest.mark.gpu
def test_capillary_wave_matches_basilisk():
    """N = 16 (8 points per wavelength), red/black: 0.138631; Basilisk -DGAUSS_SEIDEL 0.138638
    (round-off grows from ~step 50), its CPU reference (capwave.ref) 0.13865."""
    import cases.capillary_wave as cw

    r = cw.run(16, _gpu_or_numpy())
    assert r["error"] == pytest.approx(0.138638, rel=1e-4)


def test_laplace_pressure_of_a_static_drop(backend):
    """Drop R = 0.25, sigma = 1, equal densities: after 10 steps the pressure jump is sigma / R and the
    parasitic velocities are small (balanced force)."""
    n = 32
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-8)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.0625 - x * x - y * y), BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1)
    tp.add_surface_tension(1.0)
    ns.start()
    for _ in range(10):
        ns.step()
    x, y = grid.meshgrid()
    r = xp.sqrt(x * x + y * y)
    jump = float(xp.mean(ns.p.interior[r < 0.1]) - xp.mean(ns.p.interior[r > 0.4]))
    umax = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
    print(jump, umax)
    assert jump == pytest.approx(4.0, rel=0.02)
    assert math.isfinite(umax) and umax * 0.1 / 1.0 < 1e-3  # against the capillary velocity sigma / mu

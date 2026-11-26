"""VOF initial fractions from sub-cells and the CCD column curvature (docs/notes/66)."""

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.boundary import BoundaryConditions
from basilisk_cupy.fields import CellField
from basilisk_cupy.interface.curvature import curvature
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA
from basilisk_cupy.interface.vof_curvature import ccd_curvature
from cases.vof_curvature_ccd import circle_fractions

R = 0.25
C = (0.3 / 64, 0.17 / 64)


def _disc(x, y):
    return R * R - (x - C[0]) ** 2 - (y - C[1]) ** 2


def _rms(k):
    v = np.asarray(k)
    v = v[v != NODATA]
    return float(np.sqrt(np.mean((v - 1 / R) ** 2)) * R)


def test_fraction_refine():
    g = CartesianGrid((64, 64), origin=(-0.5, -0.5))
    exact = circle_fractions(64, C[0], C[1], R)
    assert np.array_equal(np.asarray(fraction(g, _disc, refine=1)), np.asarray(fraction(g, _disc)))
    errs = [float(np.abs(np.asarray(fraction(g, _disc, refine=k)) - exact).max()) for k in (1, 2, 4)]
    assert errs[0] == pytest.approx(1.5e-2, rel=0.05)
    assert errs[1] < errs[0] / 3.5 and errs[2] < errs[1] / 3.5  # O(1/k^2)


@pytest.mark.parametrize("n,basilisk,ccd", [(64, 1.96e-3, 3.03e-3), (128, 4.87e-4, 1.81e-4)])
def test_ccd_column_curvature(n, basilisk, ccd):
    """Exact fractions of a circle: tension.h's curvature is second order; CCD along the columns is far more
    accurate once the runs are long (N = 128), less so on coarse grids (45-degree run ends)."""
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    c = (0.3 / n, 0.17 / n)
    f = CellField.from_interior(g, circle_fractions(n, c[0], c[1], R), BoundaryConditions())
    assert _rms(curvature(f.interior, f.bc, g, 1.0)) == pytest.approx(basilisk, rel=0.2)
    st = {}
    k = ccd_curvature(f, 1.0, stats=st)
    assert st["ccd"] == st["interfacial"]
    assert _rms(k) == pytest.approx(ccd, rel=0.3)


def test_ccd_capillary_step():
    """CCD's curvature is stiffer at the grid scale: the capillary step shrinks by sqrt(5/12)."""
    import math

    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    g = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    dts = {}
    for method in ("basilisk", "ccd"):
        ns = TwoPhaseIMEX(g, lambda x, y: 0.0625 - x * x - y * y, rho1=1.0, rho2=0.001, sigma=1.0, interface="vof",
                          vof_curvature=method)
        ns.start()
        dts[method] = ns.dtmax
    assert dts["ccd"] == pytest.approx(dts["basilisk"] * math.sqrt(5.0 / 12.0), rel=1e-12)

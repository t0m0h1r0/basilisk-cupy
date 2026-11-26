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
    """The IMEX steppers take sqrt(5/12) of tension.h's capillary step for either curvature (SBDF2's explicit
    capillary term, docs/notes/70; CCD's stiffer grid-scale gain is covered by the same factor, not compounded)."""
    import math

    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    g = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    dts = {}
    for method in ("basilisk", "ccd"):
        ns = TwoPhaseIMEX(g, lambda x, y: 0.0625 - x * x - y * y, rho1=1.0, rho2=0.001, sigma=1.0, interface="vof",
                          vof_curvature=method)
        ns.start()
        dts[method] = ns.dtmax
    rhom = (1.0 + 0.001) / 2.0
    tension = math.sqrt(rhom * (1.0 / 32) ** 3 / math.pi)
    for method in ("basilisk", "ccd"):
        assert dts[method] == pytest.approx(tension * math.sqrt(5.0 / 12.0), rel=1e-12)


def test_cls_capillary_step_unchanged():
    """The CLS applies its own CCD factor once (a VOF-only factor must not compound with it)."""
    import math

    from basilisk_cupy.interface.cls_two_phase import CCD_CAPILLARY_DT_FACTOR
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    g = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    ns = TwoPhaseIMEX(g, 0.25 - np.hypot(x, y), rho1=1.0, rho2=0.001, sigma=1.0)
    ns.start()
    rhom = (1.0 + 0.001) / 2.0  # tension.h: the mean of 1/alpha over the extreme faces
    h = 1.0 / 32
    assert ns.dtmax == pytest.approx(math.sqrt(rhom * h ** 3 / (math.pi * 1.0)) * CCD_CAPILLARY_DT_FACTOR, rel=1e-12)


def test_phase_ccd_transfer_option():
    """PhaseCCDFaces(transfer="fd2"): fd2's two-point cell <-> face transfers, the phase-wise gradient kept."""
    from basilisk_cupy.discretization.ccd import FCCDFaceComplex
    from basilisk_cupy.discretization.fd2 import FD2FaceComplex
    from basilisk_cupy.discretization.phase_ccd import PhaseCCDFaces

    with pytest.raises(ValueError):
        PhaseCCDFaces(FCCDFaceComplex(), transfer="cubic")
    ops = PhaseCCDFaces(FCCDFaceComplex(), transfer="fd2")
    g = CartesianGrid((16, 16))
    x, y = g.meshgrid()
    c = CellField.from_interior(g, np.sin(2 * np.pi * np.asarray(x)) * np.cos(2 * np.pi * np.asarray(y)),
                                BoundaryConditions())
    c.fill_ghosts()
    ref = FD2FaceComplex()
    for ax in range(2):
        assert np.allclose(np.asarray(ops.interp_f(c, ax)), np.asarray(ref.interp_f(c, ax)))
    assert PhaseCCDFaces(FCCDFaceComplex()).interp_f.__self__ is not None  # FCCD's by default


def test_ccd_curvature_filter():
    """``filter_alpha`` (docs/notes/76): 0 leaves the CCD column curvature unchanged; a filtered run keeps a smooth
    circle's curvature accurate (the filter acts at the grid scale) and damps a grid-scale ruggedness of the
    interface (fractions of a circle with an O(h^2) zigzag radius)."""
    n = 128
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    c = (0.3 / n, 0.17 / n)
    f = CellField.from_interior(g, circle_fractions(n, c[0], c[1], R), BoundaryConditions())
    k0 = np.asarray(ccd_curvature(f, 1.0))
    assert np.array_equal(k0, np.asarray(ccd_curvature(f, 1.0, filter_alpha=0.0)))
    assert _rms(ccd_curvature(f, 1.0, filter_alpha=0.05, filter_order=2)) < 2e-3
    h = 1.0 / n
    rough = lambda x, y: (R + 2 * h * h * np.cos(np.round(np.arctan2(y - c[1], x - c[0]) * R / h) * np.pi)) ** 2 \
        - (x - c[0]) ** 2 - (y - c[1]) ** 2  # noqa: E731
    fr = CellField.from_interior(g, fraction(g, rough, refine=8), BoundaryConditions())
    assert _rms(ccd_curvature(fr, 1.0, filter_alpha=0.05, filter_order=2)) < 0.5 * _rms(ccd_curvature(fr, 1.0))

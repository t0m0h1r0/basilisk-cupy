"""Height-function curvature of the CLS field (crossings of psi = 1/2, CCD or 2nd-order along the interface)."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.cls import CLSInterface


def _crossing_kappa(q, k):
    out = []
    for ax in range(2):
        a, b = q, np.roll(q, -1, ax)
        ka, kb = k, np.roll(k, -1, ax)
        cut = (a - 0.5) * (b - 0.5) < 0
        th = (a - 0.5) / np.where(cut, a - b, 1)
        out.append(((1 - th) * ka + th * kb)[cut])
    return np.concatenate(out)


@pytest.mark.parametrize("kw,rms", [({}, 3.46e-4), ({"method": "heights"}, 1.70e-4),
                                    ({"method": "heights", "heights_method": "fd2"}, 3.29e-4),
                                    ({"method": "heights", "heights": "sum", "m": 15, "delta": 1e-3}, 1.15e-2)])
def test_circle_curvature_at_the_crossings(backend, kw, rms):
    """Exact profile of a circle R = 0.2, N = 128: rms of kappa R - 1 at the psi = 1/2 crossings (theta
    interpolated as the GFM jump). N = 256: 2.25e-4 (psi), 2.44e-5 (heights + CCD), 8.2e-5 (heights + fd2)."""
    n, R, cx, cy = 128, 0.2, 0.0123, -0.0217
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - xp.sqrt((x - cx) ** 2 + (y - cy) ** 2), BoundaryConditions.periodic(2))
    k = to_numpy(cls.curvature(**kw))
    e = _crossing_kappa(to_numpy(cls.psi.interior), k) * R - 1
    assert float(np.sqrt((e ** 2).mean())) == pytest.approx(rms, rel=0.03)


def test_capillary_wave_with_ccd_heights(backend):
    """GFM, corrected divergence, 6th-order face values, 8 / 16 / 32 / 64 points per wavelength:
    0.183 / 0.0496 / 0.0100 / 0.00478 with CCD heights (psi: 0.188 / 0.0639 / 0.0109 / 0.00209;
    2nd-order heights 0.187 / 0.0549 / 0.0120 / 0.00440)."""
    from basilisk_cupy.interface import cls as C
    from cases import capillary_wave as cw

    orig = C.CLSInterface.curvature
    C.CLSInterface.curvature = lambda self, **kw: orig(self, method="heights")
    try:
        r = cw.run(32, backend, interface="cls", pressure_jump=True, divergence="fccd", face_order=6)
    finally:
        C.CLSInterface.curvature = orig
    assert r["error"] == pytest.approx(0.0496, rel=2e-3)


@pytest.mark.parametrize("method,stable", [("psi", False), ("fd2", True)])
def test_static_drop_tension_bound_with_heights(backend, method, stable):
    """Density ratio 100, N = 64, tension.h's step (factor 1): the psi curvature grows (limit 0.95-1.0),
    2nd-order heights of the crossings stay stable (1.1e-11 after 800 steps; limit about 1.4)."""
    from basilisk_cupy.interface import cls as C
    from cases.capillary_dt_scan import growth

    orig = C.CLSInterface.curvature
    if method == "fd2":
        C.CLSInterface.curvature = lambda self, **kw: orig(self, method="heights", heights_method="fd2")
    try:
        r = growth(64, 0.01, 1.0, steps=700, backend=backend)
    finally:
        C.CLSInterface.curvature = orig
    assert r["stable"] == stable

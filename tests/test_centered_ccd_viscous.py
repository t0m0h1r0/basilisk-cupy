"""CenteredNS with an implicit CCD viscosity (``viscous=CCDViscosity()``) (docs/notes/31)."""

import pytest

from cases.ccd_viscous_figures import taylor_green


@pytest.mark.parametrize("ccd,fccd,errs", [
    (False, False, (1.37e-2, 3.35e-3)),   # 2nd order
    (True, False, (1.45e-2, 3.58e-3)),    # the projection and face interpolation keep it at 2nd order
    (True, True, (8.51e-4, 5.72e-5)),     # 4th order with FCCD face operators as well
])
def test_taylor_green_orders(ccd, fccd, errs):
    """kx = 2 pi, ky = 4 pi, nu = 0.01, Stokes, dt = 2e-4 to t = 0.04, N = 16 / 32 (N = 64: 8.34e-4 /
    8.92e-4 / 3.64e-6)."""
    for n, e in zip((16, 32), errs):
        assert taylor_green(n, ccd, fccd) == pytest.approx(e, rel=0.02)

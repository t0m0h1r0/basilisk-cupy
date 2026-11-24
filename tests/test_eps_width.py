"""Interface width eps / h (CLS): where wider profiles let the compact operators pay off (docs/notes/28)."""

import numpy as np
import pytest

from cases.eps_figures import exact_curvature, planar, translation


@pytest.mark.parametrize("c,value", [(1.5, 4.09e-4), (2.0, 1.34e-4), (3.0, 1.81e-5), (4.0, 4.60e-6)])
def test_acls_planar_imbalance(c, value):
    """ACLS (FCCD, 2008 form) on a planar profile, N = 64: max |div F| h falls about like (h/eps)^5."""
    assert planar(c) == pytest.approx(value, rel=0.03)


@pytest.mark.parametrize("c,value", [(1.5, 1.21e-3), (2.0, 1.20e-4), (3.0, 6.24e-6), (4.0, 9.71e-7)])
def test_exact_profile_curvature(c, value):
    """Curvature of psi on an exact circle, N = 128 (N = 64 / 256 at eps/h = 3: 2.32e-5 / 1.12e-5;
    at 1.5 the error grows like 1/h, 14)."""
    assert exact_curvature(c, 128) == pytest.approx(value, rel=0.03)


@pytest.mark.parametrize("c,l1,kappa", [(1.5, 1.12e-4, 0.0085), (2.0, 4.05e-5, 0.0030), (3.0, 1.92e-5, 0.0241)])
def test_translation(c, l1, kappa):
    """Translation with the corrected divergence and 6th-order face values, N = 64: psi L1 and curvature
    (N = 128: 1.10e-4 / 3.85e-5 / 9.96e-6 / 4.19e-6 and 0.0088 / 0.0018 / 0.0007 / 0.0022 for eps/h = 1.5 /
    2 / 3 / 4); at N = 64 the drop (R = 12.8 h) is too small for eps = 3 h."""
    e, k, *_ = translation(c, 64)
    assert e == pytest.approx(l1, rel=0.03) and k == pytest.approx(kappa, rel=0.05)


def test_capillary_wave_does_not_depend_on_eps(backend):
    """GFM capillary wave, 16 points per wavelength: 0.0639 / 0.0642 / 0.0645 / 0.0647 for eps/h = 1.5 / 2 /
    3 / 4: its error does not come from the interface width."""
    from cases import capillary_wave as cw

    r = cw.run(32, backend, interface="cls", pressure_jump=True, divergence="fccd", face_order=6, eps_factor=3.0)
    assert r["error"] == pytest.approx(0.06452, rel=2e-3)

"""FCCD projection with the paper's jump-corrected face gradient, smooth-pressure form (docs/notes/32)."""

import numpy as np
import pytest

from cases.fccd_projection_figures import capwave, drop


@pytest.mark.parametrize("fccd,smooth,jump,umax", [
    (False, False, 4.00267, 4.39e-5),  # fd2 faces, the pressure jumps (GFM, 16)
    (False, True, 4.00267, 4.38e-5),   # fd2 faces, smooth form: the same to round-off in the jump
    (True, False, 3.99987, 3.99e-3),   # FCCD faces across the jump: 90 times the parasitic velocity
    (True, True, 4.00267, 4.10e-5),    # FCCD faces on p~ = p - H j_ext
])
def test_static_drop(fccd, smooth, jump, umax):
    """Equal densities, R = 0.25, sigma = 1, mu = 0.1, N = 64, 10 steps (N = 32: 2.82e-4 / 2.82e-4 / 6.93e-3 /
    2.80e-4; FCCD smooth at N = 128: 7.43e-6, and 5.3e-7 after 400 steps at N = 64)."""
    ns, tp, x, y = drop(64, fccd, smooth)
    p = tp.full_pressure(ns)
    r = np.hypot(x, y)
    j = float(p[r < 0.1].mean() - p[r > 0.4].mean())
    u = max(float(np.abs(c.interior).max()) for c in ns.u)
    assert j == pytest.approx(jump, abs=2e-5) and u == pytest.approx(umax, rel=0.03)


def test_capillary_wave_fccd():
    """16 points per wavelength: Prosperetti RMS error 0.0419 (fd2 faces 0.0639), damping -10.8 %, period +1.8 %
    (fd2 faces: -12.1 %, +3.2 %); 32 points: 0.0082 (0.0109)."""
    dh, damping, period, err = capwave(32, 1.0, True)
    assert err == pytest.approx(0.0419, rel=0.01) and damping == pytest.approx(-0.1075, abs=2e-3)

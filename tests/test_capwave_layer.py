"""Capillary wave: normal-mode reference and the vorticity-layer resolution (docs/notes/30)."""

import math

import pytest

from cases import capillary_wave as cw
from cases.capwave_modes import growth, period_decay

K = 2 * math.pi


def test_inviscid_normal_mode():
    s = growth(K, 1.0, 1.0, 1e-12)
    assert abs(s.real) < 1e-4 and s.imag == pytest.approx(math.sqrt(K ** 3 / 2), rel=1e-6)


def test_normal_mode_matches_prosperetti():
    """At capwave.c's viscosity: period 3.4546, decay per period 0.6934 (Prosperetti's peaks: 3.457 / 0.692)."""
    p, d = period_decay(K, 1.0, 1.0, cw.MU)
    assert p == pytest.approx(3.4546, abs=1e-3) and d == pytest.approx(0.6934, abs=1e-3)


@pytest.mark.parametrize("n,f,err", [(32, 2.0, -0.0551), (64, 0.5, -0.0477)])
def test_damping_set_by_delta_over_h(n, f, err):
    """CLS (GFM), delta / h = 1.30 reached two ways: 16 points per wavelength with twice the viscosity and
    32 points with half of it give the same damping-rate error (-5.5 % / -4.8 %); at 16 points the error
    goes from -22 % to -2 % as the viscosity goes from x0.5 to x4."""
    from cases.capwave_layer import point

    r = point("cls", n, f)
    assert r["delta_h"] == pytest.approx(1.30, abs=0.01) and r["damping_err"] == pytest.approx(err, abs=2e-3)

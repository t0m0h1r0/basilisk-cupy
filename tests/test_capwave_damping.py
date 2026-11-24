"""Capillary wave: period and damping read from the amplitude peaks (docs/notes/29)."""

import pytest

from cases.capwave_damping import decay, run


@pytest.mark.parametrize("kind,period,dec", [("vof", 3.490, 0.6831), ("cls", 3.565, 0.7247)])
def test_viscous_damping_16_points(kind, period, dec):
    """16 points per wavelength (Prosperetti: period 3.457, decay per period 0.692): VOF is slightly
    over-damped, CLS (GFM) under-damped; 32 points: 3.470 / 0.6858 and 3.470 / 0.6992."""
    a, t, _ = run(kind, 32)
    p, d = decay(a, t)
    assert p == pytest.approx(period, abs=2e-3) and d == pytest.approx(dec, abs=5e-4)


@pytest.mark.parametrize("kind,dec", [("vof", 0.9712), ("cls", 0.9941)])
def test_inviscid_numerical_damping(kind, dec):
    """mu = 0, 16 points per wavelength: VOF's transport damps the wave by 2.9 % per period, CLS's by 0.6 %."""
    a, t, _ = run(kind, 32, mu=0.0)
    assert decay(a, t)[1] == pytest.approx(dec, abs=5e-4)

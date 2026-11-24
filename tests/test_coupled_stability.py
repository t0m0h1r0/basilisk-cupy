"""Linear model of the capillary loop and the viscous-only decay (docs/notes/33)."""

import math

import pytest

from cases.coupled_stability import limit, viscous_decay


@pytest.mark.parametrize("D", [0.0, 0.5, 1.0, 5.0, 100.0])
def test_model_limits_match_the_jury_conditions(D):
    """viscous_first: a <= 2(1+b) (the model we were sent); force_inside: a <= 4 + 2D;
    basilisk order: a <= min(2(1+b)/(2-b), (1+b)/(1-b)), which falls from 4 to 1 (omega dt 2 -> 1)."""
    b = 1 / (1 + D)
    assert limit("viscous_first", D) == pytest.approx(math.sqrt(2 * (1 + b)), abs=1e-9)
    assert limit("force_inside", D) == pytest.approx(min(math.sqrt(4 + 2 * D), 20.0), abs=1e-9)
    a = 2 * (1 + b) / (2 - b) if b == 1 else min(2 * (1 + b) / (2 - b), (1 + b) / (1 - b))
    assert limit("basilisk", D) == pytest.approx(math.sqrt(a), abs=1e-9)


def test_smoothing_the_transport_velocity_raises_the_limit():
    assert limit("basilisk", 10.0, 1.0) > 3 * limit("basilisk", 10.0, 0.0)


@pytest.mark.parametrize("visc,grows", [("fd2", False), ("split", False), ("layered", True)])
def test_viscous_only_decay_of_grid_scale_noise(visc, grows):
    """Projected white noise, frozen interface, no surface tension, N = 64, 40 steps: E/E0 = 1.18e-3 (fd2),
    1.21e-3 (CCD split), 7.8 (CCD layered: D1 D1 has no damping at the grid scale)."""
    e = viscous_decay(visc)
    assert (e[-1] > 1.0) == grows

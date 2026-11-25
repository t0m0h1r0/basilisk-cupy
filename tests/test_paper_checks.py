"""Checks behind note 55: two-fluid normal modes and the density-ratio projection (manufactured solution)."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface


def test_two_fluid_modes():
    from cases.capwave_modes import growth
    from cases.two_fluid_modes import inviscid, mode

    k = 2 * math.pi
    assert mode(k, 1.0, 1.0, 0.0182571749236, 1.0, 0.0182571749236) == pytest.approx(
        growth(k, 1.0, 1.0, 0.0182571749236), rel=1e-12)
    assert mode(10.0, 1.0, 1.0, 1e-4, 1e-4, 1e-8).real == pytest.approx(-2e-4 * 100, rel=0.02)  # weak damping
    # the paper's capillary wave (water-air, lambda = 10 mm) and Rayleigh-Taylor (water over oil)
    s = mode(2 * math.pi / 0.01, 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    assert s.real == pytest.approx(-0.802955, rel=1e-5) and s.imag == pytest.approx(134.342861, rel=1e-7)
    assert math.exp(s.real * 0.046742983863) == pytest.approx(0.963163, abs=1e-6)
    s = mode(4 * math.pi, 0.040, 950.0, 5e-3, 1000.0, 1e-3, 9.81)
    assert s.real == pytest.approx(1.754361, rel=1e-5) and abs(s.imag) < 1e-8
    assert inviscid(4 * math.pi, 0.0, 950.0, 1000.0, 9.81).real == pytest.approx(1.777899, rel=1e-6)


@pytest.mark.parametrize("variant,ratio,u_res", [
    ("fd2", 1.0, 2.381e-2), ("fd2", 1000.0, 5.403e-2), ("fccd-smooth", 1.0, 1.332e-4), ("fccd-smooth", 1000.0, 42.55),
])
def test_density_ratio_projection(variant, ratio, u_res):
    """Manufactured p_k = rho_k s + j0 H, N = 32: residual face velocity of the projection (note 55)."""
    from cases.split_ppe_check import solve

    assert solve(32, ratio, variant)["u_res"] == pytest.approx(u_res, rel=2e-3)

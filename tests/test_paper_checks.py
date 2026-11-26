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


def test_paper_capillary_wave_volume_tolerance():
    """The paper's capillary wave (water-air, 32^2, one period): the amplitude ratio at T (column-integral
    amplitude) converges with the projection bounded by the relative volume change (1e-6 / 1e-8 / 1e-10: 0.898968 /
    0.898932 / 0.898929); the absolute residual bound 1e-3 is too loose in these units (0.956)."""
    from cases.paper_benchmarks import run

    # the output times split the steps: 8 samples give 0.9046; the diffuse cell density of these measurements (the
    # default became "sharp" in docs/notes/69)
    r = run("capwave", samples=40, cell_density="diffuse")
    s = r["series"]
    assert s[-1]["amplitude"] / s[0]["amplitude"] == pytest.approx(0.898932, abs=2e-5)
    assert abs(s[-1]["mass"] / s[0]["mass"] - 1.0) < 1e-13


def test_drop_mode():
    """2-D viscous drop, mode 2: weak damping 2 n (n - 1) nu / R^2 and sqrt(6) in near vacuum; the paper's drop."""
    from cases.two_fluid_modes import drop_mode

    s = drop_mode(2, 1.0, 1.0, 1.0, 1e-3, 1e-4, 1e-7)
    assert s.real == pytest.approx(-0.004, rel=0.02) and s.imag == pytest.approx(math.sqrt(6.0), rel=1e-3)
    s = drop_mode(2, math.sqrt(5.5e-3 * 4.5e-3), 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    assert s.real == pytest.approx(-0.179428, rel=1e-4) and s.imag == pytest.approx(59.554813, rel=1e-6)

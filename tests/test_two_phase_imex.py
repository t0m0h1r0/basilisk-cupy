"""TwoPhaseIMEX: the paper's standard two-phase step as one combination (docs/notes/50)."""

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX
from cases.two_phase_imex_check import capwave, drop


@pytest.mark.parametrize("rho2,jump,u_end,u_max", [
    (1.0, 4.00977, 1.074e-4, 3.674e-4),
    (0.001, 4.01030, 1.171e-4, 6.758e-4),  # density ratio 1000 at tension.h's step x sqrt(5/12)
])
def test_static_drop(rho2, jump, u_end, u_max):
    """R = 0.25, sigma = 1, mu = 0.1, N = 32, t = 0.1 (60 / 81 steps): Laplace jump 4, mass to round-off."""
    r = drop(32, rho2, "imex", t_end=0.1)
    assert r["t"] == pytest.approx(0.1)
    assert r["jump"] == pytest.approx(jump, abs=1e-4)
    assert r["u_end"] == pytest.approx(u_end, rel=0.03) and r["u_max"] == pytest.approx(u_max, rel=0.03)
    assert abs(r["mass"]) < 1e-13


@pytest.mark.parametrize("transport,error,damping", [
    ("projected", 0.20320, -0.3601),  # the paper's stage 1: psi moved by u_f^n, first order in time
    ("ab2", 0.19070, -0.3011),        # midpoint velocity (1 + w/2) u_f^n - (w/2) u_f^{n-1}
])
def test_capillary_wave(transport, error, damping):
    """capwave.c, 8 points per wavelength (CenteredNS + CLS/GFM: RMS error 0.1866, damping -28.7 %)."""
    r = capwave(16, "imex", transport=transport)
    assert r["error"] == pytest.approx(error, rel=1e-3)
    assert r["damping"] == pytest.approx(damping, abs=1e-3)
    assert r["steps"] == 738


def test_options():
    grid = CartesianGrid((8, 8), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    phi = 0.25 - np.hypot(x, y)
    with pytest.raises(NotImplementedError):
        TwoPhaseIMEX(grid, phi, rho2=0.1, faces="fccd")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, faces="ccd")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, convection="weno")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, transport="rk2")
    ns = TwoPhaseIMEX(grid, phi, rho1=1.0, rho2=0.5, sigma=1.0, periodic=(0,))
    assert ns.uf_normal == {"bottom": 0.0, "top": 0.0}
    assert ns.cls.psi.bc.is_periodic(0) and not ns.cls.psi.bc.is_periodic(1)

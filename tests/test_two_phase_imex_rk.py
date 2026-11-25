"""IMEX Runge--Kutta tableaux and the two-phase IMEX-RK stepper (docs/notes/52)."""

import math

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.imex_rk import TABLEAUX, axis_limits
from basilisk_cupy.time.two_phase_imex_rk import TwoPhaseIMEXRK


@pytest.mark.parametrize("name,imag,real,esa,isa", [
    ("ars111", 0.0, 2.0, True, True), ("ars121", 0.0, 2.0, False, False), ("ars222", 0.0, 2.0, True, True),
    ("ars233", 1.732, 2.512, False, False), ("ars343", 2.828, 2.785, False, True), ("ars443", 1.569, 2.142, True, True),
])
def test_tableaux(name, imag, real, esa, isa):
    t = TABLEAUX[name]()
    i, r = axis_limits(t.AE, t.bE)
    assert i == pytest.approx(imag, abs=2e-3) and r == pytest.approx(real, abs=2e-3)
    assert t.explicit_stiffly_accurate == esa and t.implicit_stiffly_accurate == isa
    AE, AI = np.array(t.AE, float), np.array(t.AI, float)
    assert np.allclose(AE.sum(1), t.c) and np.allclose(AI.sum(1), t.c)  # consistent stage times
    assert sum(t.bE) == pytest.approx(1.0) and sum(t.implicit_weights) == pytest.approx(1.0)


@pytest.mark.parametrize("name,rate", [("ars111", 1.0), ("ars222", 2.0), ("ars343", 3.0)])
def test_temporal_order_stokes(name, rate):
    """Stokes, N = 64, kx != ky: rates 0.99-1.04 / 2.01-2.02 / 2.93-2.98 (each stage is projected, so the
    3rd-order tableaux are not held to the 2nd order of the incremental pressure correction)."""
    from cases.time_order import run

    ref = run(name, n=64, steps=640, stokes=True)
    e = [float(np.abs(run(name, n=64, steps=k, stokes=True) - ref).max()) for k in (20, 40)]
    assert math.log2(e[0] / e[1]) == pytest.approx(rate, abs=0.12)


@pytest.mark.parametrize("name,jump,umax", [
    ("ars111", 4.00972, 1.99e-4), ("ars222", 4.00888, 3.90e-4), ("ars343", 4.00967, 2.24e-4),
])
def test_static_drop(name, jump, umax):
    """R = 0.25, sigma = 1, mu = 0.1, N = 32, 40 steps; density ratio 1000 except ars111 (1)."""
    grid = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    rho2 = 1.0 if name == "ars111" else 0.001
    ns = TwoPhaseIMEXRK(grid, 0.25 - np.hypot(x, y), tableau=name, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, sigma=1.0,
                        tolerance=1e-8)
    ns.start()
    m0 = ns.mass()
    for _ in range(40):
        ns.step()
    p = ns.pressure()
    r = np.hypot(x, y)
    assert float(p[r < 0.1].mean() - p[r > 0.4].mean()) == pytest.approx(jump, abs=2e-5)
    assert ns.max_velocity() == pytest.approx(umax, rel=0.03)
    assert ns.mass() == pytest.approx(m0, rel=1e-13)


def test_options():
    grid = CartesianGrid((8, 8), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    phi = 0.25 - np.hypot(x, y)
    with pytest.raises(ValueError):
        TwoPhaseIMEXRK(grid, phi, tableau="rk4")
    with pytest.raises(NotImplementedError):
        TwoPhaseIMEXRK(grid, phi, rho2=0.1, faces="fccd")
    with pytest.raises(NotImplementedError):
        TwoPhaseIMEXRK(grid, phi, face_native=True)

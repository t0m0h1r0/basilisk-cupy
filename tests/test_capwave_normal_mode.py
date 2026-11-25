"""Capillary wave started on the exact normal mode (docs/notes/38)."""

import cmath
import math

import numpy as np
import pytest

from cases import capillary_wave as cw
from cases.capwave_layer_diag_figures import model_c
from cases.capwave_normal_mode import K, fit, mode, run


def test_mode_satisfies_the_interface_conditions():
    s, F = mode(K, 1.0, 1.0, cw.MU)
    x = np.array([0.1, 0.3])
    e = 1e-9
    up, lo = F(x, np.full(2, e)), F(x, np.full(2, -e))
    assert np.allclose(up[0], lo[0], atol=1e-7) and np.allclose(up[1], lo[1], atol=1e-7)  # u, v continuous
    # kinematic condition s eta = v(0): eta = 0.01 cos kx
    assert np.allclose(up[1] / s, 0.01 * np.cos(K * x), rtol=1e-6)
    # incompressible: u_x + v_y = 0 away from the interface
    y, d = 0.07, 1e-6
    ux = (F(np.array([0.2 + d]), np.array([y]))[0] - F(np.array([0.2 - d]), np.array([y]))[0]) / (2 * d)
    vy = (F(np.array([0.2]), np.array([y + d]))[1] - F(np.array([0.2]), np.array([y - d]))[1]) / (2 * d)
    assert abs(ux + vy)[0] < 1e-6 * abs(ux)[0]


def test_fit_recovers_a_damped_cosine():
    t = np.arange(200) * 3e-3
    s = complex(-1.2, 10.1)
    a = (0.01 * (0.8 - 0.3j) * np.exp(s * t)).real
    assert fit(t, a, complex(-1.0, 9.8)) == pytest.approx(s, rel=1e-8)


def test_coarse_wave_is_under_damped_and_the_budget_closes():
    r = run(16, 4.0, periods=0.6, budget=True)
    b = r["budget"]
    assert -0.25 < r["damping_err"] < -0.10  # measured -0.173 (delta / h = 0.92)
    assert b["projection"] == pytest.approx(b["force_work"] + b["pressure_work"] + b["quadratic"], rel=1e-9)
    assert b["total"] == pytest.approx(-r["s_h"].real, rel=0.05)


def test_layer_model_constant():
    assert model_c() == pytest.approx(0.376, abs=0.005)

"""Discrete dispersion relation of the capillary wave and the interface kinematics (docs/notes/41)."""

import math

import numpy as np
import pytest

from cases import capillary_wave as cw
from cases.capwave_jacobian import CapwaveMap, dispersion, kinematic_weights
from cases.capwave_kinematic_theory import growth
from cases.capwave_modes import growth as exact_growth
from cases.capwave_normal_mode import K, run

pytestmark = pytest.mark.xdist_group("capwave_theory")


def test_reduced_jacobian_matches_time_integration():
    d = dispersion(16, 4.0)
    r = run(16, 4.0, periods=1.5)
    assert d["residual"] < 1e-3  # the cos/sin family is invariant
    assert d["damping_err"] == pytest.approx(r["damping_err"], abs=0.01)
    assert d["frequency_err"] == pytest.approx(r["frequency_err"], abs=0.01)


def test_kinematic_weights_are_grid_scaled():
    for n in (16, 32):
        cm = CapwaveMap(n, 1.0, curvature="exact")
        yf, w = kinematic_weights(cm.tp)
        h = 2.0 / n
        assert w.sum() == pytest.approx(0.9815, abs=1e-3)
        assert np.sum(w * yf ** 2) / h ** 2 == pytest.approx(0.3307, abs=5e-3)  # 0.3337 at N = 16 (walls closer in h)
        assert abs(np.sum(w * yf)) < 1e-12


def test_theory_reduces_to_the_exact_mode_with_sharp_weights():
    nu = cw.MU
    s = exact_growth(K, 1.0, 1.0, nu)
    sh = growth(K, 1.0, 1.0, nu, (np.array([0.0]), np.array([1.0])), s)
    assert abs(sh - s) < 1e-10 * abs(s)


def test_sharp_kinematics_removes_the_deficit():
    """N = 16, x 4 (delta/h = 1.83 / 2 ... the same family as N = 32, x 1): CLS transport against the interface
    moved by the face velocity at the interface."""
    cls_ = dispersion(16, 4.0, curvature="exact")
    sharp = dispersion(16, 4.0, curvature="exact", kinematics="interface")
    assert cls_["damping_err"] < -0.10 and abs(sharp["damping_err"]) < 0.5 * abs(cls_["damping_err"])

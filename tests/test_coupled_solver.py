"""Outer iterations of CenteredNS(coupled_iterations=K): residuals, failure status, acceleration (docs/notes/36)."""

import warnings

import numpy as np
import pytest

from cases.coupled_residuals import drop, iterations, one_step
from cases.operator_commutator import split


@pytest.fixture(scope="module")
def ratio100():
    warnings.filterwarnings("ignore")
    return drop(16, 0.01, warm=10)


def test_residual_and_error_shrink_with_the_iterations(ratio100):
    rows, _ = iterations(ratio100, kmax=6, kref=30)
    err = [r["error"] for r in rows]
    res = [r["residual"] for r in rows]
    assert all(b < a for a, b in zip(err, err[1:])) and all(b < a for a, b in zip(res, res[1:]))
    assert all(r["div"][0] < 1e-12 for r in rows)  # the faces are divergence-free at every iteration


def test_failure_is_reported(ratio100):
    with pytest.raises(RuntimeError):
        one_step(ratio100, 2, coupled_tolerance=1e-14, coupled_on_failure="raise")
    with pytest.warns(RuntimeWarning):
        one_step(ratio100, 2, coupled_tolerance=1e-14, coupled_on_failure="warn")
    m, _ = one_step(ratio100, 60, coupled_tolerance=1e-10, coupled_on_failure="raise")
    assert m.coupled_info["converged"] and m.coupled_failures == ratio100.coupled_failures


def test_rtol_and_residual_criterion(ratio100):
    m, _ = one_step(ratio100, 40, coupled_rtol=1e-3, coupled_on_failure="raise")
    h = m.coupled_history
    assert h[-1] < 1e-3 * h[0] <= h[-2]
    m, _ = one_step(ratio100, 40, coupled_criterion="residual", coupled_tolerance=1e-6, coupled_on_failure="raise")
    assert m.coupled_residuals[-1] < 1e-6 <= m.coupled_residuals[-2]


def test_anderson_reaches_the_same_step_in_fewer_iterations(ratio100):
    plain, _ = one_step(ratio100, 40, coupled_tolerance=1e-10, coupled_on_failure="raise")
    acc, _ = one_step(ratio100, 40, coupled_tolerance=1e-10, coupled_on_failure="raise", coupled_anderson=2)
    assert acc.coupled_info["iterations"] < plain.coupled_info["iterations"]
    du = max(float(np.abs(a.interior - b.interior).max()) for a, b in zip(acc.u, plain.u))
    assert du < 1e-9


def test_lagging_pressure_residue_is_mostly_the_approximate_projection():
    """P (B - I) w = [P, B] w + (B - I) P w at N = 32: 4.21 + 6.91 -> 8.06 (ratio 100); the commutator
    vanishes for constant coefficients."""
    b = split(32, 0.01, 0.1)
    a = split(32, 1.0, 0.1)
    assert b["total"] == pytest.approx(8.062, rel=0.02)
    assert b["approx_projection"] > b["commutator"] and a["commutator"] < 1e-10

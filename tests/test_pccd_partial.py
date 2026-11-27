"""docs/notes/97: closure review, the partial-cell mixed cell, Floquet at several CFL numbers, fixed-time convergence."""

import json
import math

import numpy as np
import pytest

from cases.pccd_closure import closure, evaluate_blend
from cases.pccd_jump import A_, B_, exact
from cases.pccd_local import Split, SplitBlend, explicit_closure
from cases.pccd_twophase import averages


def _theta(k):
    return np.array(json.load(open("docs/figures/pccd_closure.json"))["runs"][k]["theta"])


def test_six_cell_closure_needs_six_cells():
    N = 32
    h = 1.0 / N
    a, b = 0.25 + 0.5 * h, 0.25 + 6.5 * h            # a heavy run of 5 pure cells (9 .. 13)
    e = np.zeros(N)
    with pytest.raises(ValueError, match="needs 6"):
        Split(N, 1e3, a, b, e, closure=closure(_theta(2)))
    Split(N, 1e3, a, b, e, closure=explicit_closure())   # the 5-cell closure works


def test_optimised_closures_regression():
    """notes 95/97: the extrapolated mixed cell with an optimised closure (blend_opt) peaks at lambda_max ~ 57."""
    for k in (1, 2):
        rows = evaluate_blend(closure(_theta(k)), ratio=1e6, zetas=(1e-2, 1e2))
        assert 50 < max(r[2] for r in rows) < 60 and max(r[1] for r in rows) < 31


def test_partial_cell_form_is_fifth_order_and_density_robust():
    e = []
    for N in (64, 128):
        ub = averages(N, A_, B_, 1e6, exact)
        R = SplitBlend(N, 1e6, A_, B_, ub, mixed_mode="partial")
        x = (np.arange(N * 41) + 0.5) / (N * 41)
        e.append(abs(R(x) - exact(x)).max())
    assert math.log2(e[0] / e[1]) > 4.8
    rows = evaluate_blend(explicit_closure(), ratio=1e6, zetas=(1e-6, 1.0, 1e2, 1e4), mixed_mode="partial")
    assert max(r[2] for r in rows) < 22 and max(r[1] for r in rows) < 23


def test_floquet_and_fixed_time_convergence():
    from cases.pccd_floquet import floquet_general, run_periodic

    r = floquet_general(32, 1e6, 1e-6, "blend_partial", "galerkin", 1, 16)
    assert r["spectral_radius"] < 1 + 1e-9 and r["sv_any_start"] < 2.2
    r = floquet_general(32, 1e6, 1e-6, "blend_partial", "remap", 1, 2)
    assert r["spectral_radius"] < 1 + 1e-9 and r["sv_max_powers"] < 4
    a = run_periodic(32, 1e6, "blend_partial", "galerkin", t_end=0.5, samples=())
    b = run_periodic(64, 1e6, "blend_partial", "galerkin", t_end=0.5, samples=())
    assert math.log2(a[0.5] / b[0.5]) > 3.5


def test_floquet_any_start_and_analytic_stiffness_bound():
    from cases.pccd_floquet import floquet_general
    from cases.pccd_gram import stiffness_analytic

    g = floquet_general(32, 1e6, 1e-6, "blend_partial", "galerkin", 1, 4)
    assert g["sv_any_start"] < 2.2 and g["sv_partial"] < 2.8 and g["spectral_radius"] < 1 + 1e-9
    s = stiffness_analytic(32, 1e4, 1e-4)
    assert s["lambda"] >= s["bound_mixed_datum"] > 1e8               # the user's lower bound, ~ (12/5) r^2

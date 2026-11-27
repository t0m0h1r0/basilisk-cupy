"""docs/notes/106: short phase runs (audit #116) and the 2-D D stiffness (audit #120 (d))."""

from cases.short_runs import continuity, evaluate
from cases.viscous_2d_twophase import check_single_phase, stiffness


def test_short_runs_are_defined_conservative_and_bounded():
    for nL in range(0, 6):
        o = evaluate(32, 1e3, nL, kind="blend")
        assert "undefined" not in o
        assert o["kink"]["mean_err"] < 1e-13 and o["smooth"]["mean_err"] < 1e-13
        assert o["lmax"] < 25 and o["kink"]["heavy"] < 1e-3
    assert "undefined" in evaluate(32, 1e3, 3, kind="base")             # docs/notes/97 as it is


def test_light_error_falls_with_the_run_length():
    e = [evaluate(32, 1e6, n, kind="blend")["kink"]["light"] for n in (1, 2, 4, 5)]
    assert e[0] > e[1] > e[2] > e[3]


def test_run_length_changes_are_continuous():
    o = continuity(32, 1e3, l0=3.3, l1=5.7, steps=25)
    assert all(r["rate"] is not None for r in o["rows"]) and o["max_rate"] < 10


def test_2d_viscous_matrix_and_d_stiffness():
    c = check_single_phase(16)
    assert c["rate_diff"] < 1e-12 and c["A_vs_K1d"] < 1e-12
    o = stiffness(32, 1e6, "nu", 1e-4, ntheta=3)
    assert o["bound_ok"] and o["D"] > 3 * o["D_1d"] and o["lambda_B_GA"] < 1.0
    o = stiffness(32, 1e3, "water-air", 0.5, ntheta=3)
    assert o["D"] < 35 and o["bound_ok"]

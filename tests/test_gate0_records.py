"""docs/notes/103 (Gate 0): the remeasured states, including the conditions that do not reach the target."""

import math

from cases.coupled_boost import run, slopes
from cases.viscous_2d import mixing
from cases.viscous_minenergy import residual_map
from cases.viscous_walls import startup


def test_coupled_boost_all_properties_structure_and_order():
    for kind in ("mu", "water-air", "nu"):
        errs = []
        for N in (32, 48, 64, 96):
            r = run(N, 1e3, kind, t_end=0.25, samples=())
            assert r["energy_max_increase"] == 0.0 and r["momentum_drift"] < 1e-11 and r["eps_G"] < 1e-12
            errs.append(r["errors"][0.25])
        # the order is set by method D (about 3, not 4): docs/notes/103 section 2
        assert slopes((32, 48, 64, 96), errs, nmin=48)[0] > 2.3


def test_method_d_is_not_fourth_order_but_c_is():
    a, b = residual_map(64, 1e3, "mu"), residual_map(128, 1e3, "mu")
    assert math.log2(a["C"]["max"] / b["C"]["max"]) > 3.7
    # D stays well above C (the gap widens with N): docs/notes/103 section 1, audit issue #114
    assert b["D"]["max"] > 50 * b["C"]["max"]


def test_consistent_force_is_fourth_order_in_the_startup_flow():
    a, b = startup(64, 1e3, "mu"), startup(128, 1e3, "mu")
    assert math.log2(a["corrected"] / b["corrected"]) > 3.5 and b["corrected"] < 1e-8
    assert math.log2(a["uncorrected"] / b["uncorrected"]) < 2.5


def test_transverse_longitudinal_leakage_is_fourth_order():
    a, b = mixing(32), mixing(64)
    for ra, rb in zip(a, b):
        if ra["slow_leak"] > 1e-12:
            assert math.log2(ra["slow_leak"] / rb["slow_leak"]) > 3.8
        else:
            assert rb["slow_leak"] < 1e-12

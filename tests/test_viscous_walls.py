"""docs/notes/101: method D's error source and the wall boundaries."""

import math

from cases.viscous_minenergy import single_phase_d
from cases.viscous_walls import balances, steady, wall_stokes


def test_true_single_phase_pccd_makes_d_as_accurate_as_c():
    p = single_phase_d(64)
    assert abs(p["D"] / p["C"] - 1) < 0.01 and p["gram_A_max"] < 1.54
    s = single_phase_d(64, kind_basis="split")
    assert s["D"] > 20 * s["C"]                       # the artificial interfaces of the phase split


def test_wall_balances_and_exact_steady_flows():
    b = balances()
    assert b["momentum_balance"] < 1e-12 and b["energy_balance"] < 1e-9 * b["dissipation"]
    for kind in ("mu", "water-air"):
        assert steady(64, 1e3, kind, "couette")["H4"] < 1e-12
        assert steady(64, 1e3, kind, "accel")["H"] < 1e-11      # acceleration force: stress linear in mass
    assert steady(64, 1.0, "water-air", "volume")["H"] < 1e-12  # viscosity jump only


def test_wall_stokes_is_fourth_order_for_c():
    a, b = wall_stokes(64, 1e3, "mu"), wall_stokes(128, 1e3, "mu")
    assert math.log2(max(a["C"]["rate_err"]) / max(b["C"]["rate_err"])) > 3.8
    assert math.log2(a["C"]["decay_err"] / b["C"]["decay_err"]) > 3.6

"""docs/notes/102: consistent volume force, the 2-D full viscous stress tensor, advection + viscosity coupled."""

import math

from cases.coupled_boost import run
from cases.viscous_2d import scan
from cases.viscous_walls import steady


def test_consistent_force_makes_volume_driven_poiseuille_exact():
    for kind in ("mu", "water-air", "nu"):
        r = steady(64, 1e3, kind, "volume", corrected=True)
        assert r["H"] < 1e-11 and r["d_norm"] > 0
    assert steady(64, 1e3, "mu", "accel", corrected=True)["d_norm"] == 0.0


def test_full_stress_tensor_is_fourth_order_in_2d():
    for method in ("C", "D"):
        a, b = scan(32, method), scan(64, method)
        for ra, rb in zip(a, b):
            assert math.log2(ra["div_free"] / rb["div_free"]) > 3.8
            assert math.log2(ra["curl_free"] / rb["curl_free"]) > 3.8


def test_coupled_advection_viscosity_is_stable_conservative_and_convergent():
    a = run(32, 1e3, "water-air", t_end=0.25, samples=())
    b = run(64, 1e3, "water-air", t_end=0.25, samples=())
    for r in (a, b):
        assert r["energy_max_increase"] == 0.0 and r["momentum_drift"] < 1e-11 and r["eps_G"] < 1e-11
    assert math.log2(a["errors"][0.25] / b["errors"][0.25]) > 3.3   # the best condition; all properties: test_gate0_records

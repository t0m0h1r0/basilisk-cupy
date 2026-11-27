"""docs/notes/114 (R0-P1): the feedback instability of the half-step source in the face value at high density
ratios. The mechanism's formulas, the one-step gain, and the interventions; the unstable and the non-conserving
variants are kept here as such."""

import math

from cases.r0p1_source_feedback import growth, kick, one_step_gain, one_sweep, single_phase, vortex


def test_uniform_g_kick_is_exact_in_one_sweep_and_sits_in_the_emptied_cells():
    for ratio in (10.0, 1e3):
        o = kick(ratio, (1.0, 0.0))
        assert o["difference_max"] < 1e-12 * max(1.0, o["kick_max"])
        assert abs(o["kick_max"] - o["half_step"] * (o["mass_ratio_max"] - 1.0)) < 0.02 * o["kick_max"]
    assert o["share_in_emptied_cells"] > 0.99 and o["kick_max"] > 100 * o["half_step"]
    two = kick(1e3, (1.0, 0.37))                                 # two sweeps: (K) as it is does not hold
    assert two["difference_max"] > 0.1 and abs(two["kick_max"] - o["kick_max"]) < 0.01


def test_random_g_face_formula_is_exact_and_the_acceleration_form_is_bounded():
    a, b = one_sweep(1e3), one_sweep(1e5)
    for o in (a, b):
        assert o["face"]["difference_max"] < 1e-11 * o["face"]["max"]
        assert o["accel"]["max_over_half_step"] < 4.0
    assert 80 < b["face"]["max_over_half_step"] / a["face"]["max_over_half_step"] < 125   # proportional to the ratio


def test_one_step_radius_is_proportional_to_the_mass_ratio():
    lo = one_step_gain(1e2, steps=(12,), iters=40)["steps"][0]
    hi = one_step_gain(1e3, steps=(12,), iters=40)["steps"][0]
    assert abs(lo["radius"] / lo["mass_ratio_max"] - hi["radius"] / hi["mass_ratio_max"]) < 0.01
    assert hi["radius"] > 10 and 0.09 < hi["radius"] / hi["mass_ratio_max"] < 0.12
    acc = one_step_gain(1e3, steps=(12,), mode="accel", iters=40)["steps"][0]
    assert acc["radius"] < 0.4


def test_growth_face_source_unstable_and_two_forms_stable():
    face = growth(1e3, "face", steps=80)
    assert face["unstable"] and 1.5 < face["rate"] < 2.4
    assert growth(1e3, "force", steps=40)["unstable"]            # the same impulse taken back with the old mass
    assert growth(1e3, "face+m", steps=40)["unstable"]           # (K) taken back: worse, not better
    assert not growth(100.0, "face", steps=80)["unstable"]
    for mode in ("accel", "trapezoid"):
        assert growth(1e6, mode, steps=80)["max"] < 1e-6


def test_all_forms_centre_the_single_phase_transport_in_time():
    for mode in ("face", "accel", "trapezoid"):
        e = [single_phase(n, mode)["max"] for n in (32, 64)]
        assert e[0] / e[1] > 5.0
    none = [single_phase(n, "none")["max"] for n in (32, 64)]
    assert none[0] / none[1] < 2.2
    assert single_phase(32, "trapezoid")["g_step_to_step"] > 0.3   # recorded: the pressure oscillates at 2 dt
    assert single_phase(32, "accel")["g_step_to_step"] < 0.1


def test_momentum_budget_of_a_drop_in_a_vortex():
    """Density ratio 1000: the default and the trapezoid form conserve the total momentum (pressure included); the
    acceleration form does not (recorded: it fails the 1e-9 requirement)."""
    for mode in ("none", "trapezoid"):
        o = vortex(mode, 32)
        assert o["failed"] is None and o["momentum_change"] < 1e-10 and o["umax"] < 3.0
    acc = vortex("accel", 32)
    assert acc["umax"] < 3.0 and 5e-4 < acc["momentum_change"] < 5e-3
    assert math.isfinite(acc["energy1"])

"""docs/notes/116: the two candidates for removing the 2 dt pressure oscillation of the "trapezoid" form of
docs/notes/114. Both are rejected; the failing conditions are recorded here."""

from cases.r0p1_source_feedback import growth, single_phase


def test_candidate_1_is_stable_but_its_pressure_oscillation_grows_with_the_grid():
    assert not growth(1e3, "trapezoid", steps=200, shape="band")["unstable"]
    osc = [single_phase(n, "trapezoid")["g_step_to_step"] for n in (32, 64)]
    ref = [single_phase(n, "none")["g_step_to_step"] for n in (32, 64)]
    assert osc[0] > 0.4 and osc[1] > osc[0] and ref[1] < ref[0] < 0.1


def test_candidate_2_damps_the_oscillation_but_is_unstable_at_density_ratio_1000():
    e = [single_phase(n, "trapezoid-p") for n in (32, 64)]
    assert e[0]["max"] / e[1]["max"] > 5.0
    assert e[1]["g_step_to_step"] < e[0]["g_step_to_step"] < 0.15
    for shape in ("drop", "band"):
        o = growth(1e3, "trapezoid-p", steps=120, shape=shape)
        assert o["unstable"] and o["rate"] > 1.3

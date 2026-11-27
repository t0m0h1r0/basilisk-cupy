"""docs/notes/113 (R0-P1 continued): the time order for unsteady / viscous single-phase flows. The first-order terms
E1 (time level of the carrying face velocity), E2 (no viscous source in the face value) and E3 (backward Euler), their
predicted shapes, and the minimal fixes. The configurations that stay first order are kept here as such."""

import math

import pytest

from cases.r0p1_unsteady_viscous import CONFIGS, NU, k2, run

T = (0.25,)


def _run(name, n, flow="tg", boost=True, nu=None, cfl=0.5):
    return run(n, cfl, flow=flow, nu=NU[flow] if nu is None else nu, boost=boost, t_ends=T, **CONFIGS[name])["errors"][-1]


@pytest.fixture(scope="module")
def tg64():
    names = ("cons/ab+visc+cn", "cons/ab", "cons/ab+cn", "cons/ab+visc", "cons/ab+visc+cn-naive", "cons/ab+lag+cn")
    return {name: _run(name, 64) for name in names}


def _coef(tg64, name):
    e, f = tg64[name], tg64["cons/ab+visc+cn"]
    return (e["c1"] - f["c1"]) / e["sum_dt2"], (e["c2"] - f["c2"]) / e["sum_dt2"]


def test_e2_and_e3_have_the_predicted_shapes_and_cancel_in_the_shift(tg64):
    lam = NU["tg"] * k2("tg")
    c1, c2 = _coef(tg64, "cons/ab")                              # E2 + E3: amplitude only, the shifts cancel
    assert abs(c1 - lam ** 2 / 2) < 0.15 * lam ** 2 / 2 and abs(c2) < 0.02 * lam
    c1, c2 = _coef(tg64, "cons/ab+cn")                           # E2 alone: a shift of the pattern, -lambda / 2
    assert abs(c2 + lam / 2) < 0.03 * lam and abs(c1) < 0.2 * lam ** 2
    c1, c2 = _coef(tg64, "cons/ab+visc")                         # E3 alone: + lambda / 2 and the amplitude
    assert abs(c2 - lam / 2) < 0.03 * lam and abs(c1 - lam ** 2 / 2) < 0.2 * lam ** 2


def test_one_fix_alone_is_worse_than_none_and_naive_crank_nicolson_is_first_order(tg64):
    base, fixed = tg64["cons/ab"]["max"], tg64["cons/ab+visc+cn"]["max"]
    assert tg64["cons/ab+cn"]["max"] > 1.5 * base and tg64["cons/ab+visc"]["max"] > 1.5 * base
    # recorded as it is: at N = 64 the unfixed E2 + E3 error (an amplitude error of the opposite sign to the space
    # error) leaves a SMALLER max error than the fixed configuration; the fixed one is better only from N = 128 on
    assert 1.2 * base < fixed < 2.0 * base
    lam = NU["tg"] * k2("tg")
    c1, c2 = _coef(tg64, "cons/ab+visc+cn-naive")                # the explicit half on u* + dt g: + lambda / 2
    assert abs(c2 - lam / 2) < 0.03 * lam
    c1, c2 = _coef(tg64, "cons/ab+lag+cn")                       # the lagged total acceleration as the source
    assert abs(c2) < 0.05 * lam and abs(c1) < 0.1 * lam ** 2


def test_without_a_boost_only_backward_euler_shows():
    """E1 and E2 are gradients for the pure decaying Taylor-Green flow: a test with it alone misses them."""
    lam = NU["tg"] * k2("tg")
    f = _run("cons+visc+cn", 64, boost=False)
    for name, want in (("cons", lam ** 2 / 2), ("cons+visc", lam ** 2 / 2), ("cons+cn", 0.0)):
        e = _run(name, 64, boost=False)
        assert abs((e["c1"] - f["c1"]) / e["sum_dt2"] - want) < 0.15 * lam ** 2 / 2 + 0.01
        if want == 0.0:
            assert abs(e["max"] - f["max"]) < 0.1 * f["max"]


def test_time_level_of_the_carrying_velocity_inviscid_boost():
    """E1: with the library's face velocity of time t^n the boosted inviscid flow is first order; the extrapolated
    face velocity (no projection) is as good as the exact one of t^{n+1/2}."""
    e = {name: [_run(name, n, flow="asym", nu=0.0)["max"] for n in (32, 64)] for name in ("cons", "cons/ab", "cons/exact")}
    assert e["cons"][0] / e["cons"][1] < 3.0 and e["cons"][1] > 1.5 * e["cons/ab"][1]
    assert e["cons/ab"][0] / e["cons/ab"][1] > 3.5
    assert e["cons/ab"][1] < 1.15 * e["cons/exact"][1]


def test_fixed_configuration_is_second_order_and_the_library_is_not():
    for flow in ("tg", "asym"):
        fix = [_run("cons/ab+visc+cn", n, flow=flow)["max"] for n in (32, 64)]
        lib = [_run("cons", n, flow=flow)["max"] for n in (32, 64)]
        assert fix[0] / fix[1] > 3.5
        assert lib[1] > fix[1]
    assert math.isfinite(lib[0])

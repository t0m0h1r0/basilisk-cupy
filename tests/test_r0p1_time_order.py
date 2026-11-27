"""docs/notes/112 (R0-P1): why the conserving momentum transport is first order in time, and the two fixes."""

from cases.r0p1_local_error import coupled, global_error, periodic_momentum_ghosts, split


def test_odd_part_matches_the_users_formula_and_not_the_commutator_alone():
    o = split(64, 0.2 / 64, "scalar")
    assert o["odd_minus_pred"] < 0.05 * o["odd"] and o["corr_odd_pred"] > 0.999
    assert o["odd_minus_commutator"] > 0.5 * o["odd"]          # the frozen dilation term is needed
    assert abs(o["odd_over_dt2"] - 5.23) < 0.05


def test_missing_half_step_source_is_an_even_second_order_local_error():
    a = split(64, 0.2 / 64, "momentum", source=False)
    assert abs(a["even_over_dt2"] - 7.0) < 0.2 and a["even_minus_pred"] < 0.08 * a["even"]
    b = split(64, 0.2 / 64, "momentum", source=True)
    assert b["even"] < 0.1 * a["even"] and abs(b["odd"] - a["odd"]) < 0.01 * a["odd"]


def test_oracle_transport_is_second_order_only_with_the_source_and_alternating_sweeps():
    e = {(src, alt): [global_error(n, 0.4, 0.25, "momentum", src, alt)["max"] for n in (32, 64)]
         for src, alt in ((False, True), (True, True), (True, False))}
    assert e[(True, True)][0] / e[(True, True)][1] > 3.3       # second order
    assert e[(False, True)][0] / e[(False, True)][1] < 2.4     # first order without the source
    assert e[(True, False)][0] / e[(True, False)][1] < 2.4     # first order with a fixed sweep order


def test_coupled_solver_needs_the_source_and_fresh_periodic_ghosts():
    old = [coupled(n, 0.5, "conserving-src", t_end=0.25, stale_ghosts=True)["max"] for n in (32, 64)]   # conserving.h
    fix = [coupled(n, 0.5, "conserving-src", t_end=0.25)["max"] for n in (32, 64)]   # the library (issue #146)
    nosrc = [coupled(n, 0.5, "conserving", t_end=0.25)["max"] for n in (32, 64)]
    with periodic_momentum_ghosts():                            # the test-side patch of note 112: the same numbers
        pat = [coupled(n, 0.5, "conserving-src", t_end=0.25, stale_ghosts=True)["max"] for n in (32, 64)]
    assert max(abs(a - b) for a, b in zip(fix, pat)) < 1e-9 * fix[0]
    assert fix[0] / fix[1] > 4.0 and fix[1] < 0.5 * old[1]      # better than second order; conserving.h's ghosts are not
    assert nosrc[0] / nosrc[1] < 2.4                            # the source is the dominant cause

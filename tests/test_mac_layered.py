"""docs/notes/108 (issue #132, T1): layered two-phase MAC; the failing conditions are recorded too."""

import numpy as np

from cases.mac_layered import geometry, leak_table, measure, projection_test


def test_leak_table_matches_the_users_values():
    t = leak_table(32)
    assert abs(t[1e6][0]["rayleigh"] - 19536) < 1 and abs(t[1e3][0]["rayleigh"] - 20.44) < 0.01
    assert abs(t[1.0][0]["rayleigh"] - 0.9236) < 1e-4
    assert t[1e6][0]["heavy_part"] > 0.9999 * t[1e6][0]["rayleigh"]     # the part no curl correction removes


def test_no_candidate_is_uniformly_equivalent_to_the_fixed_references():
    a, b = geometry(32, 0.75, 0.75)
    A, B, D0 = (measure(32, 1e6, a, b, k) for k in ("A", "B", "D0m"))
    assert A["GC_Mt"][1] > 1e5 and B["GF_Mn"][1] > 1e4          # upper bounds fail (leak into the heavy phase)
    assert D0["GF_Mn"][0] < 1e-5 and D0["L_Lref"][1] > 1e5      # the user's impossibility theorem: O(1 / r)
    assert abs(D0["GF_Mn"][0] - 1.49e-6) < 2e-8
    for o in (A, B, D0):
        assert o["null_L"] == 1 and o["const_F"] < 1e-12 and o["const_C"] < 1e-12
    assert measure(32, 1e6, a, b, "D1")["const_C"] > 1.0        # D1 does not reproduce a constant tangential velocity


def test_d0m_is_well_conditioned_without_a_reference():
    for r in (1.0, 1e3, 1e6):
        for c in (0.25, 0.75, 1e-4, 1 - 1e-4, 1 - 1e-9):
            o = measure(32, r, *geometry(32, c, c), "D0m")
            assert o["jacobi_GF"] < 3.5 and o["jacobi_GC"] < 4.0 and o["jacobi_L"] < 500
            assert o["GC_Mt"][0] > 0.99 and o["GC_Mt"][1] < 4.1 and o["const_C"] < 1e-9
    assert measure(32, 1e6, *geometry(32, 0.75, 0.75), "B")["jacobi_GF"] > 1e4
    # D0 (the light phase always restores the average) breaks down for a very thin light part
    assert measure(32, 1e3, *geometry(32, 1 - 1e-7, 1 - 1e-7), "D0")["jacobi_GC"] > 1e3


def test_projection_accuracy_separates_the_candidates():
    a, b = geometry(64, 0.75, 0.75)
    e = {k: projection_test(64, 1e6, a, b, k, 2) for k in ("M", "A", "D0m")}
    assert e["M"]["u_err"] > 1.0 and e["A"]["u_err"] > 0.1      # the reference-built MAC fails for c > 1/2
    assert e["D0m"]["u_err"] < 0.05 and e["D0m"]["v_err"] < 0.01
    lo = projection_test(64, 1e6, *geometry(64, 0.25, 0.25), "M", 2)
    assert lo["u_err"] < 0.01                                   # ... and is accurate for c < 1/2
    d3 = projection_test(64, 1e3, a, b, "D0m", 2)
    assert np.isclose(d3["u_err"], e["D0m"]["u_err"], rtol=0.02)  # independent of the density ratio


def test_blended_d0b_is_continuous_bounded_and_as_accurate():
    from cases.mac_layered import switch_continuity

    sc = switch_continuity(32, 3.0)
    assert sc["D0m"] > 1e5 and sc["D0b"] < 100                  # the switch of D0m jumps; the blend does not
    for r in (1.0, 3.0, 1e3, 1e6):
        for c in (0.25, 0.75, 1e-4, 1 - 1e-4, 1 - 1e-9):
            o = measure(32, r, *geometry(32, c, c), "D0b")
            assert o["jacobi_GF"] < 10 and o["jacobi_GC"] < 2.5 and o["const_C"] < 1e-9 and o["avg_err"] < 1e-12
    a, b = geometry(64, 0.75, 0.75)
    for src in ("normal", "tangential"):
        e = projection_test(64, 1e6, a, b, "D0b", 2, src)
        assert e["u_err"] < 0.05 and e["v_err"] < 0.01
        assert projection_test(64, 1e6, a, b, "M", 2, src)["u_err"] > 1.0

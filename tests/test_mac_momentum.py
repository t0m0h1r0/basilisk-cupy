"""docs/notes/109 (issue #134, T4): the momentum transform; the failing combinations are recorded too."""

from cases.mac_layered import geometry
from cases.mac_momentum import measure


def test_theorem_7_bound_and_dual_volumes_fail():
    a, b = geometry(32, 0.75, 0.75)
    for kind in ("A", "B", "D0m"):
        o = measure(32, 1e6, a, b, kind, ntheta=5)
        for cv in ("dual", "phase"):
            assert o[cv]["sig2d_max"] <= 1.0 + 1e-10 and o[cv]["empty"] == 0
    assert abs(measure(32, 1e6, a, b, "A", ntheta=5)["dual"]["sig2d_min"] - 0.807) < 5e-3    # the user's 0.810
    assert measure(32, 1e6, a, b, "D0m", ntheta=5)["dual"]["sig2d_min"] < 1e-3               # ~ r^{-1/2}
    assert measure(32, 1e6, *geometry(32, None, None), "B", ntheta=5)["dual"]["sig2d_min"] < 5e-3


def test_matching_reconstruction_and_control_volumes_are_uniform():
    for r in (1.0, 1e3, 1e6):
        for c in (0.25, 0.75, 1e-4, 1 - 1e-4, None):
            o = measure(32, r, *geometry(32, c, c), "D0m", ntheta=5)["phase"]
            assert o["sig2d_min"] > 0.6 and o["sig2d_divfree_min"] > 0.7
            assert 0.3 < o["G_n"][0] and o["G_n"][1] < 2.2
    # a mismatched pair fails: the low-order reconstruction with the phase-aware volumes
    assert measure(32, 1e6, *geometry(32, 0.75, 0.75), "A", ntheta=5)["phase"]["sig2d_min"] < 1e-2

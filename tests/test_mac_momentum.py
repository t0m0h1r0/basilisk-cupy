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


def test_smoothing_and_high_order_lose_the_robustness():
    # the user's independent values (1-D, density ratio 1e3, zeta_light = 1 and 2)
    r = 1e3
    o = measure(32, r, *geometry(32, r / (r + 1.0), r / (r + 1.0)), "D0b", ntheta=3, modes=("phase", "blend"))
    assert abs(o["phase"]["sig_n"][0] - 0.0841) < 1e-3
    o = measure(32, r, *geometry(32, r / (r + 2.0), r / (r + 2.0)), "D0b", ntheta=3, modes=("phase", "blend"))
    assert abs(o["blend"]["sig_n"][0] - 0.218) < 2e-3 and abs(o["blend"]["sig_t"][0] - 0.0204) < 5e-4
    # D0m with its own control volumes: the tangential transform is an isometry, but G_t / M grows like r^2
    for rr in (1e3, 1e6):
        o = measure(32, rr, *geometry(32, rr / (rr + 2.0), rr / (rr + 2.0)), "D0m", ntheta=3)["phase"]
        assert abs(o["sig_t"][0] - 1.0) < 1e-9 and o["sig_n"][0] > 0.7
        assert o["G_t"][1] > 0.2 * rr ** 2
    # the high-order prototype: ratio-independent but small
    o = measure(32, 1e6, *geometry(32, 0.75, 0.75), "D5m", ntheta=3)["phase"]
    assert 0.05 < o["sig2d_min"] < 0.3

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


# ---------------------------------------------------------------- audit #126: the number of rates against Sturm's count
def test_sturm_count_matches_a_brute_force_count_of_the_zeros():
    import math

    import numpy as np

    from cases.viscous_stress import dirichlet_count, mus

    def brute(mu_H, mu_L, ratio, LH, LL, lam, m=200001):
        kH, kL = math.sqrt(lam / mu_H), math.sqrt(lam / ratio / mu_L)
        x = np.linspace(0, LH, m)[1:]
        A, B = math.sin(kH * LH) / (mu_H * kH), math.cos(kH * LH)
        s = np.linspace(0, LL, m)[1:-1]
        u = np.concatenate([np.sin(kH * x) / (mu_H * kH), A * np.cos(kL * s) + B * np.sin(kL * s) / (mu_L * kL)])
        return int((u[:-1] * u[1:] < 0).sum())

    rng = np.random.default_rng(1)
    for kind, ratio in (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3)):
        mH, mL = mus(ratio, kind)
        for _ in range(6):
            a, lam = rng.uniform(0.05, 0.95), 10 ** rng.uniform(0, 4.5)
            assert brute(mH, mL, ratio, a, 1 - a, lam) == dirichlet_count(mH, mL, ratio, a, 1 - a, lam)


def test_periodic_scan_recovers_the_close_pair_it_used_to_drop():
    """Constant kinematic viscosity, density ratio 1e3, a = 0.2613, b = 0.7137: the 41st and 42nd rates are 0.010
    apart in sqrt(lambda), less than the scan step 0.037; the sign-change scan returned the 43rd rate as the 41st."""
    from cases.viscous_stress import mus, stokes_rate

    mH, mL = mus(1e3, "nu")
    r = stokes_rate(mH, mL, 1e3, 0.2613, 0.7137, which=43)
    assert abs(r[40] - 17408.519) < 0.01 and abs(r[41] - 17411.191) < 0.01 and abs(r[42] - 18958.40) < 0.01
    assert all(x <= y for x, y in zip(r[:-1], r[1:]))
    s = stokes_rate(1.0, 1.0, 1.0, 0.2613, 0.7137, which=6)   # the uniform fluid: double rates (2 pi m)^2, twice each
    import math
    assert all(abs(x - (2 * math.pi * (k // 2 + 1)) ** 2) < 1e-6 for k, x in enumerate(s))


def test_wall_rates_complete_at_the_conditions_of_note_103():
    """The three interface positions and four property sets of docs/notes/103, as many modes as ``startup`` takes
    and more: the count agrees (wall_rates raises otherwise), so no mode is missing from the reference solution."""
    from cases.viscous_stress import mus
    from cases.viscous_walls import wall_rates

    for kind, ratio in (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3)):
        mH, mL = mus(ratio, kind)
        for off in (0.0, 0.0071, 0.0193):
            lams, _ = wall_rates(mH, mL, ratio, 0.3613 + off, which=160)
            assert len(lams) == 160 and all(x < y for x, y in zip(lams[:-1], lams[1:]))

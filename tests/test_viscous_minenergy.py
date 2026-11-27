"""docs/notes/100: the minimum-viscous-energy reconstruction and the inner products of advection and viscosity."""

import math

import numpy as np

from cases.viscous_minenergy import MinEnergy, d_stiffness, stokes_test, structure, v4_test
from cases.viscous_stress import Setup, mus, stokes_rate


def test_exact_stokes_rates_are_the_lowest_ones():
    mH, mL = mus(1e3, "mu")
    r = stokes_rate(mH, mL, 1e3, 0.2613, 0.7137, which=2)
    assert abs(r[0] - 75.66675225) < 1e-6 and abs(r[1] - 192.42377814) < 1e-6
    s = stokes_rate(1.0, 1.0, 1.0, 0.2613, 0.7137, which=2)
    assert abs(s[0] / (2 * math.pi) ** 2 - 1) < 1e-10 and abs(s[1] / s[0] - 1) < 1e-12   # a double rate


def test_structure_and_compact_form():
    for kind in ("nu", "mu", "water-air"):
        st = structure(32, 1e3, kind)
        assert st["mean_err"] < 1e-13 and st["G1_m"] < 1e-13 and st["K1"] < 1e-12
        assert st["energy_identity"] < 1e-12 and st["G_minus_M_min_eig"] > -1e-14
    S = Setup(32, 1e6, (8 + 1 - 1e-4) / 32, 0.7137, *mus(1e6, "nu"))
    ME = MinEnergy(S)
    Gd = ME.gram_direct()
    assert np.abs(Gd - ME.G).max() < 1e-13 * np.abs(Gd).max()


def test_fourth_order_unsteady_stokes_with_jumps():
    for kind in ("mu", "water-air"):
        a, b = stokes_test(64, 1e3, kind), stokes_test(128, 1e3, kind)
        c = "C: Gnu, Knu"
        assert math.log2(max(a[c]["rate_err"]) / max(b[c]["rate_err"])) > 3.8
        assert math.log2(a[c]["decay_err"] / b[c]["decay_err"]) > 3.8


def test_method_d_keeps_fourth_order_rates_and_bounded_stiffness_for_water_air():
    a, b = v4_test(32, 1e3, "water-air"), v4_test(64, 1e3, "water-air")
    assert math.log2(max(a["D"]["rate_err"]) / max(b["D"]["rate_err"])) > 3.8
    assert a["PtGnuP_le_GA"] < 1e-12
    assert d_stiffness(32, 1e6, "water-air", 1e-4)["stiff_D"] < 11

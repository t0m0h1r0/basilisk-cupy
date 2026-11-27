"""docs/notes/99 (V1, V3 trial): the compact-stress viscous operator."""

import math

import numpy as np

from cases.viscous_stress import Setup, mus, steady_error, structure


def test_uniform_rows_are_the_fourth_order_compact_relation():
    S = Setup(16, 1.0, 0.2613, 0.7137, 1.0, 1.0)
    assert np.allclose(S.H4[5, 4:7] / S.h, [1 / 12, 5 / 6, 1 / 12])
    assert np.allclose(S.H3[5, 4:7] / S.h, [1 / 12, 5 / 6, 1 / 12])


def test_structure_constant_stress_momentum_energy_and_bounded_stiffness():
    for kind in ("nu", "mu", "water-air"):
        for r in (1.0, 1e3):
            s = structure(32, r, kind)
            assert s["const_stress_residual"] < 1e-12 and s["momentum"] < 1e-12 and s["K_min_eig_rel"] > -1e-14
            assert s["stiffness_h2_over_nu"] < 6.1                       # the user's bound is 32
    for c in (1e-8, 1e-4, 0.5):                                         # a vanishing heavy sliver at ratio 1e6
        a = (8 + 1 - c) / 32
        assert structure(32, 1e6, "nu", a=a)["stiffness_h2_over_nu"] < 6.1


def test_h4_is_fourth_order_single_phase_and_h3_restores_it_with_jumps():
    e = [steady_error(N, 1.0, "nu") for N in (64, 128)]
    assert math.log2(e[0] / e[1]) > 3.9
    e4 = [steady_error(N, 1e3, "mu") for N in (64, 128)]
    e3 = [steady_error(N, 1e3, "mu", H="H3") for N in (64, 128)]
    assert math.log2(e3[0] / e3[1]) > 3.6 and e3[1] < e4[1] / 100

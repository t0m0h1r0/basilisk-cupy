"""docs/notes/92 (P1-1): plain averages recovered from mass-weighted averages in the mixed cells."""

import math

from cases.pccd_density import test_case as case


def test_recovery_is_fifth_order_and_independent_of_the_density_ratio():
    for frac in (0.5, 1e-4):
        e = {}
        for N in (32, 64):
            h = 1.0 / N
            xa, xb = (N // 4 + 1 - frac) * h, (3 * N // 4 + 0.37) * h
            for ratio in (1.0, 1e6):
                r = case(N, xa, xb, ratio)
                e[(N, ratio)] = r["err"]
                assert r["cond_schur"] < 1.5
        assert math.log2(e[(32, 1e6)] / e[(64, 1e6)]) > 4.8
        assert abs(e[(64, 1e6)] / e[(64, 1.0)] - 1) < 0.05

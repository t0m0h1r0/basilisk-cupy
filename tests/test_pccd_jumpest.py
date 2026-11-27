"""docs/notes/94 (P1-3): jumps estimated from the data."""

import math

from cases.pccd_jump import reconstruct
from cases.pccd_jumpest import amplification


def test_estimated_jumps_keep_fifth_order_at_density_ratio_one_million():
    a, b = reconstruct(64, 1e6, True, estimate=True), reconstruct(256, 1e6, True, estimate=True)
    assert math.log2(a["err"] / b["err"]) / 2 > 4.8
    o = reconstruct(256, 1e6, True)                          # known jumps
    assert b["err"] < 1.2 * o["err"]


def test_amplification_stays_moderate():
    assert amplification(64, 1e6, 100.0) < 40


def test_joint_estimate_is_continuous_and_less_amplifying():
    from cases.pccd_jumpest import continuity

    assert amplification(64, 1e6, 100.0, method="joint") < 10
    a = continuity(64, 1.0, 1e-2, method="joint")["max_change"]
    b = continuity(64, 1.0, 5e-3, method="joint")["max_change"]
    assert b < 0.6 * a                     # shrinks with the spacing: continuous
    o = continuity(64, 1.0, 5e-3)["max_change"]
    assert o > 5                           # the one-sided stencil jumps

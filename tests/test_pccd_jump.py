"""docs/notes/93 (P1-2): jump-separated PCCD with known jumps."""

import math

from cases.pccd_jump import reconstruct


def test_separation_restores_high_order_and_the_plain_pccd_is_first_order():
    a, b = reconstruct(64, 1e6, True), reconstruct(256, 1e6, True)
    assert math.log2(a["err"] / b["err"]) / 2 > 4.5
    c, d = reconstruct(64, 1e6, False), reconstruct(256, 1e6, False)
    assert math.log2(c["err"] / d["err"]) / 2 < 1.5
    assert d["err_far"] > 1e-6          # without the separation the error spreads away from the interface

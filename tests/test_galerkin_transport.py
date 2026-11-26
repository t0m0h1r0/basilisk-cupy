"""The geometry-consistent Galerkin transport G^{n+1} u^{n+1} = C u^n in 1-D at density ratio 1000 (docs/notes/80)."""

import math

from cases.galerkin_transport import one_step, run


def test_structure_density_ratio_1000():
    """Free stream and total momentum to round-off, energy-weighted amplification at most 1, energy non-increasing."""
    for p in (3, 4):
        r = one_step(32, p)
        assert r["free_stream"] < 1e-12 and r["momentum"] < 1e-14
        assert r["amplification"] <= 1.0 + 1e-10 and r["dE"] < 1e-14 and r["min_eig_G"] > 0.0


def test_quartic_order():
    """Fourth order of the quartic reconstruction at t = 0.137 (not at the end of a period)."""
    e = [run(n, 4)[0] for n in (16, 32)]
    assert math.log2(e[0] / e[1]) > 4.0

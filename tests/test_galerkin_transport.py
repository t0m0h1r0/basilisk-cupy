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


# docs/notes/81: the piecewise reconstruction sharing the interface value a0


def test_piecewise_polynomial_exactness():
    """P0-2: per-phase polynomials (continuous at the interfaces) of degree <= p are reproduced to round-off."""
    from cases.galerkin_transport import exactness

    for p in (2, 3, 4):
        err, scale = exactness(16, p)
        assert err < 1e-12 * max(scale, 1.0)


def test_piecewise_schur_bound():
    """P0-3: the Schur denominator stays above 1/(p+1), independent of the density ratio."""
    from cases.galerkin_transport import sweep

    for p in (2, 3):
        res = [sweep(p, ratio, samples=101) for ratio in (1.0, 1e3, 1e6)]
        for r in res:
            assert r["schur"] > 1.0 / (p + 1)
            assert abs(r["amp"] - res[0]["amp"]) < 1e-8


def test_piecewise_structure_and_order():
    """Structure of the piecewise transport and fourth order for per-phase non-polynomial q."""
    for p in (3, 4):
        r = one_step(32, p, piecewise=True)
        assert r["free_stream"] < 1e-12 and r["momentum"] < 1e-14
        assert r["amplification"] <= 1.0 + 1e-10 and r["dE"] < 1e-14 and r["mass_eig"][0] > 1.0 - 1e-10
    e = [run(n, 4, kind="phases", piecewise=True)[0] for n in (32, 64)]
    assert math.log2(e[0] / e[1]) > 4.0


def test_piecewise_switching_jump():
    """P0-4 (records the open issue): the update jumps by O(1) when the interface crosses a cell boundary, but is
    continuous when it moves within a cell; the single polynomial is continuous both ways."""
    from cases.galerkin_transport import switching

    assert switching(32, 4, 1e-8, across=True, piecewise=True)[0] > 0.5
    assert switching(32, 4, 1e-8, across=False, piecewise=True)[0] < 1e-5
    assert switching(32, 4, 1e-8, across=True, piecewise=False)[0] < 1e-5

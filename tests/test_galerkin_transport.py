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


# docs/notes/82: the constrained weighted least-squares reconstruction (piecewise="ls", alpha 0.75, window +1)


def _ls_cond(s, alpha, extra, piecewise="ls", ratio=1e6, n=48):
    import numpy as np

    from cases.galerkin_transport import matrices

    G, _, _, new = matrices(n, 4, s - 1e-9, 1e-9, piecewise, ratio=ratio, alpha=alpha, extra=extra)
    Mh = np.diag(1 / np.sqrt(new.mass))
    ev = np.linalg.eigvalsh(Mh @ G @ Mh)
    return ev.max() / ev.min()


def test_ls_continuous_across_cell_boundary():
    """The update is continuous when the interface crosses a cell boundary (the piecewise basis jumps by O(1))."""
    import numpy as np

    from cases.galerkin_transport import A0, matrices

    n, h = 48, 1.0 / 48
    k = int(round(A0 / h)) + 1
    eps = 1e-8
    jump = {}
    for mode in ("ls", True):
        G, C, _, _ = matrices(n, 4, k * h - eps * h - A0, 2 * eps * h, mode, ratio=1e3, alpha=0.75, extra=1)
        jump[mode] = np.linalg.norm(np.linalg.solve(G, C) - np.eye(n), 2)
    assert jump["ls"] < 1e-3 and jump[True] > 0.5


def test_ls_exactness_and_structure():
    """Per-phase polynomials reproduced; free stream, momentum and energy-norm amplification <= 1."""
    import numpy as np

    from cases.galerkin_cls import step_stats
    from cases.galerkin_transport import exactness

    for s in (0.0, 0.3 / 32):
        err, scale = exactness(32, 4, 1e6, piecewise="ls", alpha=0.75, s=s, extra=1)
        assert err < 1e-10 * scale
    st = step_stats(48, 4, 0.0107, 0.37 / 48, 0.75, 1e3)
    assert np.abs(st["T"] @ np.ones(48) - 1).max() < 1e-12
    assert st["amp"] <= 1.0 + 1e-9


def test_ls_conditioning_near_cell_boundary():
    """At the heavy-left interface next to a cell boundary the square fits (piecewise, or ls without the wider
    window) are ill conditioned at density ratio 1e6; the over-determined mass-weighted fit is not."""
    import math

    from cases.galerkin_transport import B0

    h = 1.0 / 48
    s = (int(math.floor(B0 / h)) + 1 + 1e-4) * h - B0
    assert _ls_cond(s, 0.75, 1) < 10.0
    assert _ls_cond(s, 0.75, 0) > 1e3

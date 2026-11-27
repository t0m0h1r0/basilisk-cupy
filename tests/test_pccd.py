"""docs/notes/91 (P0): the primitive-function CCD (PCCD), constant density, 1-D periodic."""

import math

import numpy as np
from numpy.polynomial import polynomial as Pn

from cases.pccd import BUB, H_G0, H_G1, H_V0, H_V1, MEAN, _mono, accuracy, ccd_matrices, operators, symbols


def test_ccd_on_the_primitive_is_exact_to_quintics_and_the_reconstruction_to_quartics():
    rng = np.random.default_rng(0)
    for p in range(6):
        cu = rng.normal(size=p + 1)
        cF, du = Pn.polyint(cu), Pn.polyder(cu)
        u = lambda x: Pn.polyval(x, cu)  # noqa: E731
        ux = (lambda x: Pn.polyval(x, du)) if p > 0 else (lambda x: 0.0 * x)
        q = lambda i: Pn.polyval(i + 1, cF) - Pn.polyval(i, cF)  # noqa: E731
        i = 0.3
        r1 = u(i) + 7 / 16 * (u(i - 1) + u(i + 1)) - 1 / 16 * (ux(i + 1) - ux(i - 1)) - 15 / 16 * (q(i - 1) + q(i))
        r2 = ux(i) - 1 / 8 * (ux(i - 1) + ux(i + 1)) + 9 / 8 * (u(i + 1) - u(i - 1)) - 3 * (q(i) - q(i - 1))
        assert abs(r1) < 1e-12 and abs(r2) < 1e-12
        if p <= 4:
            c = u(0) * H_V0 + ux(0) * H_G0 + u(1) * H_V1 + ux(1) * H_G1
            c = c + (q(0) - c @ MEAN) * BUB / (BUB @ MEAN)
            s = np.linspace(0, 1, 11)
            assert abs(_mono(s) @ c - u(s)).max() < 1e-12


def test_reconstruction_is_c1_mean_preserving_and_fifth_order():
    a, b = accuracy(32), accuracy(64)
    for r in (a, b):
        assert r["mean"] < 1e-14 and r["c1_jump_value"] < 1e-13 and r["c1_jump_slope"] < 1e-13
    assert math.log2(a["rec"] / b["rec"]) > 4.8
    assert math.log2(a["v"] / b["v"]) > 5.8


def test_fourier_stability_dissipation_and_gram_range():
    """|T| <= 1 for all three transports; the user's dissipation values at theta = pi/2, C = 0.37."""
    worst = 0.0
    for C in (0.05, 0.37, 0.9):
        for t in np.linspace(1e-3, math.pi, 301):
            r = symbols(t, C)
            worst = max(worst, abs(r["remap"]), abs(r["galerkin"]), abs(r["corr"]))
            assert 1.0 - 1e-12 < r["gram"].real < 1.5322
    assert worst < 1 + 1e-12
    r = symbols(math.pi / 2, 0.37)
    d = {k: (1 - abs(r[k]) ** 2) / 0.37 for k in ("remap", "galerkin", "corr")}
    assert abs(d["remap"] / 8.073e-3 - 1) < 1e-3 and abs(d["galerkin"] / 6.658e-5 - 1) < 1e-3
    assert d["corr"] < 2e-9


def test_structure_of_the_transports():
    N = 32
    o = operators(N, 0.37)
    one = np.ones(N)
    for k in ("remap", "galerkin", "corr"):
        T = o[k]
        assert abs(T @ one - 1).max() < 1e-13                      # free stream
        assert abs(one @ T - one).max() < 1e-13                    # total mass (constant density: sum of averages)
    T, G = o["galerkin"], o["G"]
    ev = np.linalg.eigvalsh(T.T @ G @ T - G)
    assert ev.max() < 1e-13                                        # energy cannot grow
    A, _ = ccd_matrices(N)
    assert np.linalg.cond(A) < 25

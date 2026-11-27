"""docs/notes/83: the dissipation of the Galerkin transport at constant density (Bloch symbols)."""

import math

import numpy as np

from cases.galerkin_fourier import dissipation, jump_rate, symbol


def test_face_jump_formula():
    """|J|^2 = (1024/900) sin^10(theta/2) for the cell-wise quartic, and the small-CFL decay is |J|^2 / g."""
    for th in (math.pi / 4, math.pi / 2, math.pi):
        g, J2 = jump_rate(th)
        assert abs(J2 - 1024 / 900 * math.sin(th / 2) ** 10) < 1e-10 * J2
        assert abs(dissipation("poly", 1e-4, th)["d"] - g) < 1e-2 * g


def test_spline_dissipation_vanishes_with_dt():
    """Per unit time: the cell-wise quartic tends to a constant, the continuous B-spline to zero (proportional to
    the CFL number)."""
    p = [dissipation("poly", C, math.pi / 2)["d"] for C in (1e-3, 1e-2)]
    s = [dissipation("spline", C, math.pi / 2)["d"] for C in (1e-3, 1e-2)]
    assert abs(p[1] / p[0] - 1) < 0.05
    assert abs(s[1] / s[0] - 10) < 0.1
    assert s[0] < 1e-4 * p[0]


def test_symbol_matches_the_matrix():
    """The Bloch symbol is the eigenvalue of the assembled update (density ratio 1, cell-wise quartic)."""
    from cases.galerkin_transport import matrices

    n, C = 16, 0.37
    G, Cm, _, _ = matrices(n, 4, 0.0, C / n, False, ratio=1.0)
    T = np.linalg.solve(G, Cm)
    for m in (1, 4, 8):
        th = 2 * math.pi * m / n
        v = np.exp(1j * np.arange(n) * th)
        assert np.abs(T @ v - symbol("poly", C, th) * v).max() < 1e-12

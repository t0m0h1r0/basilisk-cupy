"""docs/notes/85: the shifted fit (fit="shift") against the constrained fit at a heavy sliver in a light cell."""

import numpy as np

from cases.galerkin_amp import four, place
from cases.galerkin_transport import exactness, matrices


def test_shift_removes_the_amplification_peak():
    """Density ratio 1e6, zeta = 100 (A0): ||T_G||_inf 272 -> 13; the piecewise-constant remap stays at 1."""
    c = four(64, 1e6, 100.0, "A0", 0.37, fit="constrained")
    s = four(64, 1e6, 100.0, "A0", 0.37, fit="shift")
    assert abs(c["PC"]["inf"] - 1) < 1e-12 and c["G"]["inf"] > 100
    assert s["G"]["inf"] < 20 and s["cond"] < 5


def test_shift_keeps_exactness_continuity_and_structure():
    import cases.galerkin_transport as gt

    R0 = gt.Reconstruction
    try:
        gt.Reconstruction = lambda *a, **k: R0(*a, **{**k, "fit": "shift"})
        err, scale = exactness(32, 4, 1e6, piecewise="ls", alpha=0.75, s=0.004, extra=1)
    finally:
        gt.Reconstruction = R0
    assert err < 1e-10 * scale
    n, h = 48, 1.0 / 48
    s0 = place(n, 1e3, 1.0, "A0")
    G, C, _, _ = matrices(n, 4, s0 - 1e-8 * h, 2e-8 * h, "ls", ratio=1e3, alpha=0.75, extra=1, fit="shift")
    T = np.linalg.solve(G, C)
    assert np.linalg.norm(T - np.eye(n), 2) < 1e-3
    assert np.abs(T @ np.ones(n) - 1).max() < 1e-11


def test_rows_integrate_the_actual_density_with_two_interfaces():
    """docs/notes/86: with both interfaces inside one window the fit's rows integrate the real density (the masses
    agree with the exact cell masses) and per-phase polynomials stay exact while the band is >= 7 cells."""
    import cases.galerkin_transport as gt

    B0 = gt.B0
    try:
        gt.B0 = gt.A0 + 7 / 48
        rec = gt.Reconstruction(48, 4, 0.0, "ls", 1e3, 0.75, 1, fit="shift")
        zz = gt.interfaces(0.0)[0]
        cz = int(np.floor(zz * 48))
        A, mass = rec._rows(zz, np.arange(cz - 5, cz + 7))
        assert np.allclose(mass, rec.mass[np.arange(cz - 5, cz + 7) % 48], rtol=1e-12)
        R0 = gt.Reconstruction
        gt.Reconstruction = lambda *a, **k: R0(*a, **{**k, "fit": "shift"})
        try:
            err, scale = gt.exactness(48, 4, 1e3, piecewise="ls", alpha=0.75, s=0.0, extra=1)
        finally:
            gt.Reconstruction = R0
        assert err < 1e-10 * scale
    finally:
        gt.B0 = B0

"""1-D theory of BCG-type face fluxes (docs/notes/74): periodic constant advection of cell averages, the face value r
of face i + 1/2 (u > 0, upwind cell i) and the update q_i -= C (r_{i+1/2} - r_{i-1/2}). The exact operator from
cell averages to the swept mean over a step is

    R_C(theta) = (1 - e^{-i C theta}) / (C (1 - e^{-i theta}))   (= 1 + (1 - C)/2 D + (1 - C)(1 - 2C)/12 D^2 + ...)

and the amplification G = 1 - C (1 - e^{-i theta}) R. Schemes compared:

* ``bcg``: R = 1 + (1 - C)/2 d1 (centred slope);
* ``ccd-parabola``: R = 1 + (1 - C)/2 d1 + (1 - C)(1 - 2C)/12 d2 with the CCD symbols d1, d2 (``cases/ccd_bcg.py``);
* ``compact(alpha, stencil)``: a compact flux ``alpha r_{f-1} + r_f + alpha r_{f+1} = sum_j a_j(C) qbar_{i+j}`` with
  a fixed left-hand side and right-hand coefficients matching the Taylor series of R_C to the order the stencil
  allows, for each C (the stencil ``j`` relative to the upwind cell).

For every scheme: the order at small theta, max over theta of |G| - 1 (stability), and |G - G_exact| at theta = pi/4
and pi/2, over C in (0, 1].

    python -m cases.compact_flux_theory
"""

from __future__ import annotations

import numpy as np

THETA = np.linspace(1e-3, np.pi, 400)
CS = np.linspace(0.05, 1.0, 20)


def exact_R(theta, C):
    return (1.0 - np.exp(-1j * C * theta)) / (C * (1.0 - np.exp(-1j * theta)))


def G_of(theta, C, R):
    return 1.0 - C * (1.0 - np.exp(-1j * theta)) * R


def ccd_symbols(theta):
    """CCD d1, d2 symbols (h = 1) evaluated at ``theta`` from the implemented periodic matrices."""
    from basilisk_cupy.discretization.ccd import ccd_matrices

    n = 512
    d1, d2 = ccd_matrices(n, 1.0, "periodic", "periodic")[:2]
    th = 2 * np.pi * np.arange(n // 2 + 1) / n
    v = np.exp(1j * np.outer(np.arange(n), th))
    s1, s2 = (d1 @ v)[0] / v[0], (d2 @ v)[0] / v[0]
    return np.interp(theta, th, s1.real) + 1j * np.interp(theta, th, s1.imag), np.interp(theta, th, s2.real)


def _series(C, order: int):
    """Taylor coefficients in D of R_C(D) = (1 - e^{-C D}) / (C (1 - e^{-D})), D = h d/dx (exact, from sympy)."""
    import sympy as sp

    D = sp.symbols("D")
    expr = (1 - sp.exp(-sp.Rational(C).limit_denominator(10 ** 6) * D)) / \
        (sp.Rational(C).limit_denominator(10 ** 6) * (1 - sp.exp(-D)))
    ser = sp.series(expr, D, 0, order + 1).removeO()
    return np.array([float(ser.coeff(D, k)) for k in range(order + 1)])


def compact_fit(C, alpha, stencil):
    """Right-hand coefficients a_j(C) of ``alpha r_{f-1} + r_f + alpha r_{f+1} = sum_j a_j qbar_{i+j}`` matching the
    Taylor series of ``R_C(D) (1 + alpha (e^D + e^-D))`` in D to the order the stencil allows (len(stencil) - 1): the
    classical compact design; the face shift is e^D per face, the cell shift e^{j D}."""
    import math

    K = len(stencil) - 1
    r = _series(float(C), K)
    lhs = np.zeros(K + 1)  # 1 + alpha (e^D + e^-D): 1 + 2 alpha at D^0, 2 alpha D^2/2!, ...
    for k in range(K + 1):
        lhs[k] = (1.0 if k == 0 else 0.0) + (alpha * 2.0 / math.factorial(k) if k % 2 == 0 else 0.0)
    target = np.array([sum(r[m] * lhs[k - m] for m in range(k + 1)) for k in range(K + 1)])
    A = np.array([[j ** k / math.factorial(k) for j in stencil] for k in range(K + 1)], dtype=float)
    return np.linalg.solve(A, target)


def compact_R(theta, C, alpha, stencil, a):
    return (np.exp(1j * np.outer(theta, stencil)) @ a) / (1.0 + 2.0 * alpha * np.cos(theta))


def evaluate(name, R_of):
    rows = []
    for C in CS:
        R = R_of(THETA, C)
        G = G_of(THETA, C, R)
        Ge = np.exp(-1j * C * THETA)
        err = np.abs(G - Ge)
        small = THETA < 0.2
        order = np.polyfit(np.log(THETA[small]), np.log(np.maximum(err[small], 1e-300)), 1)[0] - 1.0  # per step / C
        rows.append((C, float(np.abs(G).max() - 1.0), float(np.interp(np.pi / 4, THETA, err)),
                     float(np.interp(np.pi / 2, THETA, err)), order))
    worst = max(r[1] for r in rows)
    print(f"{name:34s} max|G|-1 {worst:+.1e} ({sum(r[1] > 1e-12 for r in rows)}/{len(rows)} C unstable)  "
          f"|G-Ge| at pi/4: max {max(r[2] for r in rows):.1e}  pi/2: max {max(r[3] for r in rows):.1e}  "
          f"order (small theta, C=0.5) {[r[4] for r in rows if abs(r[0] - 0.5) < 0.03][0]:.2f}", flush=True)


def main() -> None:
    s1, s2 = ccd_symbols(THETA)
    evaluate("bcg (centred slope)", lambda th, C: 1.0 + 0.5 * (1.0 - C) * 1j * np.sin(th))
    evaluate("ccd slope", lambda th, C: 1.0 + 0.5 * (1.0 - C) * s1)
    evaluate("ccd parabola", lambda th, C: 1.0 + 0.5 * (1.0 - C) * s1 + (1.0 - C) * (1.0 - 2.0 * C) / 12.0 * s2)
    for alpha in (0.0, 0.25, 1.0 / 3.0):
        for stencil in ((-1, 0, 1), (-2, -1, 0, 1), (-1, 0, 1, 2), (-2, -1, 0, 1, 2)):
            st = np.array(stencil)
            coeffs = {float(C): compact_fit(C, alpha, st) for C in CS}

            def R_of(th, C, alpha=alpha, st=st, coeffs=coeffs):
                return compact_R(th, C, alpha, st, coeffs[float(C)])

            evaluate(f"compact a={alpha:.3f} j={stencil}", R_of)


if __name__ == "__main__":
    main()

"""docs/notes/92 (P1-1): PCCD with a density jump and a smooth velocity (1-D periodic).

The stored velocity is the mass-weighted average  ubar_i = (1/m_i) int_{K_i} rho u.  PCCD needs the plain averages q_i.
In a pure cell q_i = ubar_i. In the mixed cells the q_i are unknowns fixed by

    (1/m_i) int_{K_i} rho (R_PCCD q) = ubar_i      (i mixed)

(the user's section 6.1): R_PCCD is linear in q, so this is a small linear system for the mixed cells (a Schur
complement of A_rho R), solved together with the CCD system.

    python -m cases.pccd_density
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import XG, WG, _mono, reconstruction_operator


def pieces(N, xa, xb):
    """Per cell: [(s0, s1, heavy)], heavy on (xa, xb) (0 < xa < xb < 1, periodic unit interval)."""
    h = 1.0 / N
    out = []
    for i in range(N):
        cuts = [0.0, 1.0] + [(x - i * h) / h for x in (xa, xb) if i * h < x < (i + 1) * h]
        cuts = sorted(cuts)
        out.append([(a, b, xa < (i + 0.5 * (a + b)) * h < xb) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 0])
    return out


def setup(N, xa, xb, ratio):
    """A_rho R (N, N): the mass-weighted mean of the reconstruction; masses m; the mixed cells."""
    h = 1.0 / N
    R = reconstruction_operator(N)
    P = pieces(N, xa, xb)
    AR = np.zeros((N, N))
    m = np.zeros(N)
    mixed = []
    for i, ps in enumerate(P):
        if len(ps) > 1:
            mixed.append(i)
        row = np.zeros(5)
        for a, b, hv in ps:
            r = 1.0 if hv else 1.0 / ratio
            s = a + (b - a) * XG
            row += r * (b - a) * (WG @ _mono(s))
            m[i] += r * (b - a)
        AR[i] = row @ R[i] / m[i]
    return AR, m * h, np.array(mixed, dtype=int), R, P


def recover(ubar, AR, mixed):
    """q: pure cells q = ubar, mixed cells from A_rho R q = ubar (rows of the mixed cells)."""
    N = ubar.size
    pure = np.setdiff1d(np.arange(N), mixed)
    q = ubar.copy()
    if mixed.size:
        S = AR[np.ix_(mixed, mixed)]
        q[mixed] = np.linalg.solve(S, ubar[mixed] - AR[np.ix_(mixed, pure)] @ ubar[pure])
        return q, float(np.linalg.cond(S))
    return q, 1.0


def test_case(N, xa, xb, ratio, k=1):
    h = 1.0 / N
    U = lambda x: np.sin(2 * np.pi * k * x)  # noqa: E731
    AR, m, mixed, R, P = setup(N, xa, xb, ratio)
    ubar = np.zeros(N)
    for i, ps in enumerate(P):
        num = den = 0.0
        for a, b, hv in ps:
            r = 1.0 if hv else 1.0 / ratio
            s = a + (b - a) * XG
            num += r * (b - a) * (WG @ U((i + s) * h))
            den += r * (b - a)
        ubar[i] = num / den
    q, cond = recover(ubar, AR, mixed)
    c = np.einsum("ipj,j->ip", R, q)
    s = np.linspace(0, 1, 21)
    err = float(np.abs(c @ _mono(s).T - U((np.arange(N)[:, None] + s[None, :]) * h)).max())
    qex = (np.cos(2 * np.pi * k * np.arange(N) * h) - np.cos(2 * np.pi * k * (np.arange(N) + 1) * h)) / (2 * np.pi * k * h)
    return {"err": err, "cond_schur": cond, "q_err": float(np.abs(q - qex).max()), "mixed": int(mixed.size)}


def main() -> None:
    res = []
    for N in (16, 32, 64, 128):
        h = 1.0 / N
        for ratio in (1.0, 1e3, 1e6):
            worst, cmax = 0.0, 0.0
            for frac in (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6, 0.999):
                # interfaces inside cells N/4 and 3N/4 + 1; heavy fraction frac of the first cut cell
                xa = (N // 4 + 1 - frac) * h
                xb = (3 * N // 4 + 0.37) * h
                r = test_case(N, xa, xb, ratio)
                worst, cmax = max(worst, r["err"]), max(cmax, r["cond_schur"])
            res.append({"N": N, "ratio": ratio, "err": worst, "cond": cmax})
            print(N, ratio, f"err {worst:.3e} cond {cmax:.3f}", flush=True)
    json.dump(res, open("docs/figures/pccd_density.json", "w"), indent=1)


if __name__ == "__main__":
    main()

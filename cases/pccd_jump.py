"""docs/notes/93 (P1-2): jump-separated PCCD with known jumps (the Oracle), 1-D periodic.

u = sin 2 pi x + [a < x < b] g(x), g = 3 (x - a)(b - x) cos 2 pi x: continuous, its derivatives jump at a and b.
Split u = w + J,  J = sum over the interfaces of  H(d) chi(d) T(d),  T the degree-4 Taylor polynomial of the jump
(outside minus inside, d = x - x_Gamma oriented to the + side) and chi a C-infinity cut-off (1 for |d| < L1, 0 beyond
L2) that keeps the two corrections apart. Then w is C^4 across both interfaces, and PCCD applies to w:

    u_h = R_PCCD q_w + J,     q_w from the mass-weighted averages (note 92):
    pure cells: q_w,i = ubar_i - mean(J);  mixed cells: (1/m_i) int rho (R q_w + J) = ubar_i.

Compared with PCCD applied to u itself (note 92's recovery, ignoring the jumps).

    python -m cases.pccd_jump
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, _mono, reconstruction_operator
from cases.pccd_density import pieces

A_, B_ = 0.2613, 0.7137
L1, L2 = 0.0, 0.2


def _chi(d):
    t = np.clip((np.abs(d) - L1) / (L2 - L1), 0.0, 1.0)
    f = lambda z: np.where(z > 0, np.exp(-1.0 / np.maximum(z, 1e-300)), 0.0)  # noqa: E731
    return 1.0 - f(t) / (f(t) + f(1.0 - t))


def exact(x, a=A_, b=B_):
    inside = (x > a) & (x < b)
    return np.sin(2 * np.pi * x) + np.where(inside, 3 * (x - a) * (b - x) * np.cos(2 * np.pi * x), 0.0)


def _g_derivs(x0, a, b, n=5):
    """g^(k)(x0), k = 0..n-1, of g(x) = 3 (x - a)(b - x) cos 2 pi x (Leibniz)."""
    p = np.poly1d([-3.0, 3 * (a + b), -3 * a * b])       # 3 (x - a)(b - x)
    out = []
    for k in range(n):
        s = 0.0
        for j in range(k + 1):
            pk = np.polyder(p, j)(x0) if j <= 2 else 0.0
            ck = (2 * np.pi) ** (k - j) * math.cos(2 * np.pi * x0 + (k - j) * math.pi / 2)
            s += math.comb(k, j) * pk * ck
        out.append(s)
    return out


def jump_terms(a=A_, b=B_, order=4):
    """[(x_Gamma, coefficients of T in d)]: at a the + side (x > a) is inside: jump = g; at b the + side is outside:
    jump = -g."""
    ga, gb = _g_derivs(a, a, b), _g_derivs(b, a, b)
    Ta = [ga[k] / math.factorial(k) for k in range(order + 1)]
    Tb = [-gb[k] / math.factorial(k) for k in range(order + 1)]
    return [(a, Ta), (b, Tb)]


def J(x, terms):
    out = np.zeros_like(x)
    for xg, T in terms:
        d = (x - xg + 0.5) % 1.0 - 0.5
        out += np.where(d > 0, _chi(d) * sum(c * d ** k for k, c in enumerate(T)), 0.0)
    return out


def estimate_terms(ubar, N, a, b, mixed, width=5):
    """docs/notes/94 (P1-3): the jumps from the data. On each side of each interface, the quartic whose averages over
    the ``width`` nearest pure cells (the mixed cell skipped) equal the data; the jump of its derivatives k = 1..4 at
    x_Gamma (the value's jump is zero: the velocity is continuous)."""
    h = 1.0 / N
    terms = []
    for xg in (a, b):
        m = int(math.floor(xg / h))
        sides = []
        for sgn in (1, -1):
            cells = [m + sgn * (k + 1) for k in range(width)]
            # averages of (x - xg)^p over the (unwrapped) cells, in units of h
            M = np.array([[((((c + 1) * h - xg) / h) ** (p + 1) - ((c * h - xg) / h) ** (p + 1)) / (p + 1)
                           for p in range(5)] for c in cells])
            coef = np.linalg.solve(M, ubar[np.array(cells) % N])     # P(x) = sum coef_p ((x - xg)/h)^p
            sides.append(coef)
        # the jump is (x > x_Gamma side) - (x < x_Gamma side) at both interfaces, as in jump_terms
        terms.append((xg, [0.0] + [(sides[0][k] - sides[1][k]) / h ** k for k in range(1, 5)]))
    return terms


def reconstruct(N, ratio, separate=True, a=A_, b=B_, order=4, estimate=False, data=None, width=5, ubar_in=None):
    """``estimate``: the jumps from the data (estimate_terms); ``data``: a function u(x) replacing ``exact`` (pure
    cells: plain averages; continuity tests use a random smooth field)."""
    h = 1.0 / N
    R = reconstruction_operator(N)
    P = pieces(N, a, b)
    terms = jump_terms(a, b, order) if (separate and not estimate) else []
    ubar = np.zeros(N)
    Jbar = np.zeros(N)          # plain average of J (pure cells) and its mass-weighted mean (all cells)
    JR = np.zeros(N)
    AR = np.zeros((N, N))
    mixed = []
    U = data or (lambda x: exact(x, a, b))
    if estimate and ubar_in is not None:
        ubar = np.asarray(ubar_in, dtype=float).copy()
        terms = estimate_terms(ubar, N, a, b, None, width)
    elif estimate:
        for i, ps in enumerate(P):
            num = den = 0.0
            for s0, s1, hv in ps:
                r = 1.0 if hv else 1.0 / ratio
                s = s0 + (s1 - s0) * XG
                w = (s1 - s0) * WG
                num += r * (w @ U((i + s) * h))
                den += r * w.sum()
            ubar[i] = num / den
        mixed0 = [i for i, ps in enumerate(P) if len(ps) > 1]
        terms = estimate_terms(ubar, N, a, b, mixed0, width)
    for i, ps in enumerate(P):
        num = den = 0.0
        row = np.zeros(5)
        jr = jp = 0.0
        for s0, s1, hv in ps:
            r = 1.0 if hv else 1.0 / ratio
            s = s0 + (s1 - s0) * XG
            w = (s1 - s0) * WG
            x = (i + s) * h
            num += r * (w @ U(x)) if ubar_in is None else 0.0
            den += r * w.sum()
            row += r * (w @ _mono(s))
            jv = J(x, terms) if terms else 0 * x
            jr += r * (w @ jv)
            jp += w @ jv
        ubar[i] = num / den if ubar_in is None else ubar_in[i]
        AR[i] = row @ R[i] / den
        JR[i] = jr / den
        Jbar[i] = jp
        if len(ps) > 1:
            mixed.append(i)
    mixed = np.array(mixed, dtype=int)
    pure = np.setdiff1d(np.arange(N), mixed)
    q = ubar - Jbar
    S = AR[np.ix_(mixed, mixed)]
    q[mixed] = np.linalg.solve(S, ubar[mixed] - JR[mixed] - AR[np.ix_(mixed, pure)] @ q[pure])
    c = np.einsum("ipj,j->ip", R, q)
    s = np.linspace(0.0, 1.0, 41)
    X = (np.arange(N)[:, None] + s[None, :]) * h
    Uh = c @ _mono(s).T + (J(X, terms) if terms else 0.0)
    err = np.abs(Uh - U(X))
    # errors near the interfaces (cells within 2 of a mixed cell) and away from them
    near = np.zeros(N, dtype=bool)
    for m in mixed:
        for o in range(-2, 3):
            near[(m + o) % N] = True
    return {"err": float(err.max()), "err_far": float(err[~near].max()), "cond": float(np.linalg.cond(S)),
            "values": Uh, "terms": terms}


def main() -> None:
    res = []
    for sep in (False, True):
        for ratio in (1.0, 1e3, 1e6):
            for N in (32, 64, 128, 256, 512):
                r = reconstruct(N, ratio, sep)
                r = {k: v for k, v in r.items() if k not in ("values", "terms")}
                res.append({"separate": sep, "ratio": ratio, "N": N, **r})
                print(sep, ratio, N, r, flush=True)
    json.dump(res, open("docs/figures/pccd_jump.json", "w"), indent=1)


if __name__ == "__main__":
    main()

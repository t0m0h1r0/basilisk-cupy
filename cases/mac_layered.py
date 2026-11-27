"""docs/notes/108 (issue #132, T1): layered two-phase MAC (density depends on y only), small matrices.

Normal direction (y): face values f_j at y_j = j h (face j below cell j), d_i = (f_{i+1} - f_i) / h, and a commuting
pair R_F f = f_i + int R_C d, with R_C a reconstruction of cell values that keeps the volume average.

Candidates (docs/notes/107, 3.5-3.6):
  A   piecewise-constant R_C (piecewise-linear normal velocity): the low-order MAC reconstruction;
  B   the single-phase MAC-PCCD (docs/notes/105), no interface treatment;
  D0  density-adapted, low order: in a cut cell the heavy part takes the value of the adjacent pure heavy cell (0 if
      none), the light part takes the constant that restores the cell's volume average;
  D0m as D0, but the phase with the larger mobility (length / rho) in the cut cell restores the average and the other
      phase takes its pure neighbour's value (D0 lets a very thin light part carry a large amplitude);
  D0b the blend of the two choices of D0m with a smooth weight of the heavy mobility share over [1/4, 3/4] (both are
      linear, keep the volume average and reproduce constants, so the blend does; continuous in the interface position);
  D1  density-adapted, low order: R_C d = d_i (1/rho) / mean_i(1/rho) (the variation sits in the light part).
All keep the volume average, so d/dy R_F = R_C D and the face values are reproduced (checked).

Reference inner products, fixed before looking at the results (the user's rule):
  normal      M_n = diag(int over the dual cell [y_j - h/2, y_j + h/2] of rho)       (series mobility h / int rho)
  tangential  M_t = diag(h / mean_i(1/rho))                                           (parallel mobility)
  M_pt = diag(rho(y_j) h) (the density at the face point) is reported as a second comparison only.
Reference-free numbers: the condition number of the Jacobi-scaled Gram and Schur matrices.

    python -m cases.mac_layered
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, _mono
from cases.pccd_density import pieces

RATIOS = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
KINDS = ("A", "B", "D0", "D0m", "D0b", "D1", "D5m", "D5b")
S2KINDS = ("A", "B", "D0m", "D5m", "D5b")


def _piece_values(N, P, ratio, kind):
    """Per cell: list of (s0, s1, heavy, row) with row (N,) the constant value of R_C on that piece per unit cell
    datum (kinds A, D0, D1)."""
    pure_heavy = [len(ps) == 1 and ps[0][2] for ps in P]
    pure_light = [len(ps) == 1 and not ps[0][2] for ps in P]
    out = []
    for i, ps in enumerate(P):
        e = np.zeros(N)
        e[i] = 1.0
        if len(ps) == 1 or kind == "A":
            out.append([(s0, s1, hv, e) for s0, s1, hv in ps])
            continue
        if kind == "D1":
            inv = sum((s1 - s0) * (1.0 if hv else ratio) for s0, s1, hv in ps)        # mean of 1 / rho
            out.append([(s0, s1, hv, e * ((1.0 if hv else ratio) / inv)) for s0, s1, hv in ps])
            continue
        # D0: the light phase closes the volume average. D0m: the phase with the larger mobility (length / rho)
        # closes it; the other phase takes the value of its adjacent pure cell (0 if none).
        mob_h = sum(s1 - s0 for s0, s1, hv in ps if hv)
        mob_l = sum(s1 - s0 for s0, s1, hv in ps if not hv) * ratio

        def closed_by(close_heavy):
            rows, used, llen = [], np.zeros(N), 0.0
            for s0, s1, hv in ps:
                if hv == close_heavy:
                    llen += s1 - s0
                    rows.append(None)
                    continue
                pure_same = pure_light if close_heavy else pure_heavy
                k = None
                if s0 == 0.0 and pure_same[(i - 1) % N]:
                    k = (i - 1) % N
                elif s1 == 1.0 and pure_same[(i + 1) % N]:
                    k = (i + 1) % N
                r = np.zeros(N)
                if k is not None:
                    r[k] = 1.0
                rows.append(r)
                used += (s1 - s0) * r
            closing = (e - used) / llen
            return [closing if r is None else r for r in rows]
        if kind == "D0b":
            # the weight of "heavy closes": a smooth step of the heavy mobility share over [1/4, 3/4]
            t = min(max((mob_h / (mob_h + mob_l) - 0.25) / 0.5, 0.0), 1.0)
            wh = t * t * (3 - 2 * t)
            rl, rh = closed_by(False), closed_by(True)
            rows = [(1 - wh) * x + wh * y for x, y in zip(rl, rh)]
        else:
            rows = closed_by(kind == "D0m" and mob_h > mob_l)
        out.append([(s0, s1, hv, r) for (s0, s1, hv), r in zip(ps, rows)])
    return out


def _sub(s0, s1):
    """The piece [s0, s1] cut at the dual-cell boundary s = 1/2 (control-volume integrals need it)."""
    return [(s0, 0.5), (0.5, s1)] if s0 < 0.5 < s1 else [(s0, s1)]


def build(N, ratio, a, b, kind):
    """Quadrature values of the bases: PhiC (points x N cells) for R_C, PhiF (points x N faces) for R_F, with the
    geometric weights w, the density rho, the phase flag (from the geometry, also valid at density ratio 1) and the
    reproduction errors. Every phase piece is also cut at s = 1/2, so that sums over the dual cells are exact."""
    h = 1.0 / N
    P = pieces(N, a, b)
    D = np.roll(np.eye(N), -1, axis=0) - np.eye(N)              # (D f)_i = f_{i+1} - f_i
    X, W, RHO, PC, PF, HV = [], [], [], [], [], []
    face_err = avg_err = 0.0
    if kind in ("D5m", "D5b"):
        return _build_high(N, ratio, a, b, kind, P, D, h)
    if kind == "B":
        from cases.mac_pccd import operators

        RF, RC, _, _, _ = operators(N)
        for i, ps in enumerate(P):
            for q0, q1, hv in ps:
                for s0, s1 in _sub(q0, q1):
                    s = s0 + (s1 - s0) * XG
                    X.append((i + s) * h), W.append((s1 - s0) * h * WG)
                    RHO.append(np.full(XG.size, 1.0 if hv else 1.0 / ratio)), HV.append(np.full(XG.size, hv))
                    PC.append(_mono(s) @ RC[i])
                    PF.append(np.stack([s ** p for p in range(6)], axis=-1) @ RF[i])
    else:
        vals = _piece_values(N, P, ratio, kind)
        for i, cell in enumerate(vals):
            cum = np.zeros(N)                                   # int_0^{s0} R_C (per unit cell data)
            e = np.zeros(N)
            e[i] = 1.0
            for q0, q1, hv, row in cell:
                for s0, s1 in _sub(q0, q1):
                    s = s0 + (s1 - s0) * XG
                    X.append((i + s) * h), W.append((s1 - s0) * h * WG)
                    RHO.append(np.full(XG.size, 1.0 if hv else 1.0 / ratio)), HV.append(np.full(XG.size, hv))
                    PC.append(np.tile(row, (XG.size, 1)))
                    PF.append(e[None, :] + (cum[None, :] + np.outer(s - s0, row)) @ D)
                    cum = cum + (s1 - s0) * row
            avg_err = max(avg_err, float(np.abs(cum - e).max()))
            e1 = np.zeros(N)
            e1[(i + 1) % N] = 1.0
            face_err = max(face_err, float(np.abs(e + cum @ D - e1).max()))
    X, W, RHO = np.concatenate(X), np.concatenate(W), np.concatenate(RHO)
    PC, PF = np.concatenate(PC), np.concatenate(PF)
    return {"X": X, "w": W, "rho": RHO, "heavy": np.concatenate(HV), "PhiC": PC, "PhiF": PF, "D": D, "h": h, "P": P,
            "avg_err": avg_err, "face_err": face_err}


def _build_high(N, ratio, a, b, kind, P, D, h):
    """docs/notes/108, section 4 (a first high-order prototype, not the work of #136): per phase the PCCD run on the
    volume averages of the pure cells (cases.pccd_local / cases.short_runs with density ratio 1, so the averages are
    volume averages), and in a cut cell the D0m / D0b structure with polynomials: every phase part is the extension
    of its own run's end-cell polynomial, and the phase that restores the volume average (larger mobility, or the
    D0b blend) gets a constant added."""
    from cases.short_runs import SplitShort

    RCp = np.zeros((N, 5, N))                                   # pure cells: s-coefficients per unit datum
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        RCp[:, :, j] = SplitShort(N, 1.0, a, b, e, mixed_mode="extend").coef
    pure = [len(ps) == 1 for ps in P]
    X, W, RHO, PC, PF, HV = [], [], [], [], [], []
    face_err = avg_err = 0.0
    pw = np.arange(5)

    def shifted(k_cell, off):
        """(5, N) s-coefficient map of P_k(s - off) expanded in s."""
        out = np.zeros((5, N))
        for q in range(5):
            for p_ in range(q + 1):
                out[p_] += RCp[k_cell, q] * math.comb(q, p_) * (-off) ** (q - p_)
        return out

    def prim(co, s):
        """int_0^s of sum co_p t^p: (len(s), N)."""
        return np.stack([s ** (p_ + 1) / (p_ + 1) for p_ in pw], axis=-1) @ co
    for i, ps in enumerate(P):
        e = np.zeros(N)
        e[i] = 1.0
        if pure[i]:
            cos = [RCp[i]]
        else:
            ext = []
            for s0, s1, hv in ps:
                k = None
                if s0 == 0.0 and pure[(i - 1) % N] and P[(i - 1) % N][0][2] == hv:
                    k, off = (i - 1) % N, -1.0
                elif s1 == 1.0 and pure[(i + 1) % N] and P[(i + 1) % N][0][2] == hv:
                    k, off = (i + 1) % N, 1.0
                ext.append(shifted(k, off) if k is not None else np.zeros((5, N)))
            tot = sum(prim(c, np.array([s1]))[0] - prim(c, np.array([s0]))[0] for c, (s0, s1, hv) in zip(ext, ps))
            mob_h = sum(s1 - s0 for s0, s1, hv in ps if hv)
            mob_l = sum(s1 - s0 for s0, s1, hv in ps if not hv) * ratio
            if kind == "D5b":
                t = min(max((mob_h / (mob_h + mob_l) - 0.25) / 0.5, 0.0), 1.0)
                wh = t * t * (3 - 2 * t)
            else:
                wh = 1.0 if mob_h > mob_l else 0.0
            len_h = sum(s1 - s0 for s0, s1, hv in ps if hv)
            cos = []
            for c, (s0, s1, hv) in zip(ext, ps):
                c = c.copy()
                wgt = wh if hv else 1.0 - wh                    # the share of the correction this phase takes
                c[0] += wgt * (e - tot) / (len_h if hv else 1.0 - len_h)
                cos.append(c)
        cum = np.zeros(N)
        for c, (q0, q1, hv) in zip(cos, ps):
            for s0, s1 in _sub(q0, q1):
                sq = s0 + (s1 - s0) * XG
                X.append((i + sq) * h), W.append((s1 - s0) * h * WG)
                RHO.append(np.full(XG.size, 1.0 if hv else 1.0 / ratio)), HV.append(np.full(XG.size, hv))
                PC.append(_mono(sq) @ c)
                base = prim(c, np.array([s0]))[0]
                PF.append(e[None, :] + (cum[None, :] + prim(c, sq) - base[None, :]) @ D)
                cum = cum + prim(c, np.array([s1]))[0] - base
        avg_err = max(avg_err, float(np.abs(cum - e).max()))
        e1 = np.zeros(N)
        e1[(i + 1) % N] = 1.0
        face_err = max(face_err, float(np.abs(e + cum @ D - e1).max()))
    return {"X": np.concatenate(X), "w": np.concatenate(W), "rho": np.concatenate(RHO), "heavy": np.concatenate(HV),
            "PhiC": np.concatenate(PC), "PhiF": np.concatenate(PF), "D": D, "h": h, "P": P, "avg_err": avg_err,
            "face_err": face_err}


def references(N, ratio, a, b):
    h = 1.0 / N

    def rho(y):
        return np.where(((y - a) % 1.0) < ((b - a) % 1.0), 1.0, 1.0 / ratio)

    def integ(fun, lo, hi):
        cuts = sorted({lo, hi, *[c + k for c in (a, b) for k in (-1.0, 0.0, 1.0) if lo < c + k < hi]})
        return sum((q - p) * (WG @ fun((p + (q - p) * XG) % 1.0)) for p, q in zip(cuts[:-1], cuts[1:]))
    Mn = np.array([integ(rho, (j - 0.5) * h, (j + 0.5) * h) for j in range(N)])
    Mt = np.array([h * h / integ(lambda y: 1.0 / rho(y), i * h, (i + 1) * h) for i in range(N)])
    Mpt = np.array([0.5 * (rho(np.array([(j * h - 1e-12) % 1.0]))[0] + rho(np.array([(j * h + 1e-12) % 1.0]))[0]) * h
                    for j in range(N)])
    return Mn, Mt, Mpt


def _geig(A, Mdiag):
    s = 1.0 / np.sqrt(Mdiag)
    return np.linalg.eigvalsh(0.5 * (A + A.T) * s[:, None] * s[None, :])


def _jacobi_cond(A, drop=0):
    s = 1.0 / np.sqrt(np.diag(A))
    ev = np.linalg.eigvalsh(0.5 * (A + A.T) * s[:, None] * s[None, :])
    return float(ev[-1] / ev[drop])


def measure(N, ratio, a, b, kind):
    """1-D (theta_x = 0): Gram of the normal velocity against M_n (and M_pt), Gram of the tangential cell
    reconstruction against M_t, the Schur complement against the series-mobility Laplacian, and reference-free
    Jacobi-scaled condition numbers."""
    R = build(N, ratio, a, b, kind)
    h = R["h"]
    wr = R["w"] * R["rho"]
    GF = R["PhiF"].T @ (wr[:, None] * R["PhiF"])
    GC = R["PhiC"].T @ (wr[:, None] * R["PhiC"])
    Mn, Mt, Mpt = references(N, ratio, a, b)
    B = R["D"] / h
    L = B @ np.linalg.solve(GF, B.T)
    Lref = B @ np.diag(1.0 / Mn) @ B.T
    Q = np.linalg.qr(np.column_stack([np.ones(N), np.eye(N)[:, :N - 1]]))[0][:, 1:]
    Lq, Lrq = Q.T @ L @ Q, Q.T @ Lref @ Q
    C = np.linalg.inv(np.linalg.cholesky(Lrq))
    evL = np.linalg.eigvalsh(C @ (0.5 * (Lq + Lq.T)) @ C.T)
    evF, evP, evC = _geig(GF, Mn), _geig(GF, Mpt), _geig(GC, Mt)
    evLfull = np.linalg.eigvalsh(0.5 * (L + L.T))
    one = np.ones(N)
    return {"N": N, "ratio": ratio, "kind": kind, "avg_err": R["avg_err"], "face_err": R["face_err"],
            "const_F": float(np.abs(R["PhiF"] @ one - 1.0).max()), "const_C": float(np.abs(R["PhiC"] @ one - 1.0).max()),
            "GF_Mn": [float(evF[0]), float(evF[-1])], "GF_Mpt": [float(evP[0]), float(evP[-1])],
            "GC_Mt": [float(evC[0]), float(evC[-1])], "L_Lref": [float(evL[0]), float(evL[-1])],
            "null_L": int((evLfull < 1e-10 * evLfull[-1]).sum()),
            "jacobi_GF": _jacobi_cond(GF), "jacobi_GC": _jacobi_cond(GC), "jacobi_L": _jacobi_cond(L, drop=1),
            "jacobi_Lref": _jacobi_cond(Lref, drop=1)}


def geometry(N, c_a, c_b=None):
    """Heavy arc a -> b: the cut cell at a holds the heavy fraction c_a on its right, the one at b the fraction c_b on
    its left. c = None puts the interface on a face (no cut cell)."""
    h = 1.0 / N
    ia, ib = N // 4, (3 * N) // 4
    a = ia * h if c_a is None else (ia + 1 - c_a) * h
    b = ib * h if c_b is None else (ib + c_b) * h
    return a, b


def leak_table(N=32):
    """The user's table (docs/notes/107, 3.5): single-phase MAC-PCCD, h = 1, heavy cells 0..N/2-1, the energy of the
    unit datum of a light face / (rho_L h), and its heavy-phase part (the lower bound that no curl correction can
    remove for x-uniform normal data)."""
    from cases.mac_pccd import M6, operators

    RF = operators(N)[0]
    out = {}
    for r in (1.0, 1e3, 1e6):
        heavy = np.arange(N) < N // 2
        rho = np.where(heavy, 1.0, 1.0 / r)
        rows = []
        for dist in range(1, 6):
            j = N // 2 + dist
            c = RF[:, :, j]
            e = np.einsum("ip,pq,iq->i", c, M6, c)
            rows.append({"dist": dist, "rayleigh": float((rho * e).sum() * r), "heavy_part": float((rho * e)[heavy].sum() * r)})
        out[r] = rows
    return out


# --------------------------------------------------------------------------------------------- projection accuracy
def exact_projection(ratio, a, b, m=1, source="normal"):
    """The density-weighted Helmholtz projection of w = (0, cos 2 pi y) e^{i k x}, k = 2 pi m, in the layered medium
    (heavy rho = 1 on [a, b), light 1 / ratio elsewhere, periodic): u = w - grad p / rho, div u = 0, i.e.
    (p' / rho)' - k^2 p / rho = w2'. Per layer p = rho A sin 2 pi y + alpha e^{k (y - y0)} + beta e^{-k (y - y1)},
    A = 2 pi / (4 pi^2 + k^2); p and p' / rho continuous at both interfaces (a 4 x 4 real system).
    source "tangential": w = (cos 2 pi y, 0) e^{i k x}; then p = i q with (q' / rho)' - k^2 q / rho = k cos 2 pi y and
    q = rho A2 cos 2 pi y + ..., A2 = -k / (4 pi^2 + k^2) (the returned p, dp are q, q').
    Returns p(y), dp(y), rho(y)."""
    k = 2 * math.pi * m
    A = 2 * math.pi / (4 * math.pi ** 2 + k * k) if source == "normal" else -k / (4 * math.pi ** 2 + k * k)
    trig = math.sin if source == "normal" else math.cos
    dtrig = (lambda z: math.cos(z)) if source == "normal" else (lambda z: -math.sin(z))
    rH, rL = 1.0, 1.0 / ratio
    LH, LL = (b - a) % 1.0, 1.0 - (b - a) % 1.0
    # heavy: t in [0, LH] from a; light: t in [0, LL] from b. Exponentials scaled to stay bounded.
    def hom(t, L):
        return np.array([math.exp(k * (t - L)), math.exp(-k * t)]), k * np.array([math.exp(k * (t - L)), -math.exp(-k * t)])
    M = np.zeros((4, 4))
    rhs = np.zeros(4)
    eH1, dH1 = hom(LH, LH)
    eL0, dL0 = hom(0.0, LL)
    eL1, dL1 = hom(LL, LL)
    eH0, dH0 = hom(0.0, LH)
    # at y = b: p_H = p_L, p_H'/rH = p_L'/rL
    M[0, :2], M[0, 2:], rhs[0] = eH1, -eL0, (rL - rH) * A * trig(2 * math.pi * b)
    M[1, :2], M[1, 2:] = dH1 / rH, -dL0 / rL
    # at y = a (the light layer's end): p_L = p_H, fluxes
    M[2, 2:], M[2, :2], rhs[2] = eL1, -eH0, (rH - rL) * A * trig(2 * math.pi * a)
    M[3, 2:], M[3, :2] = dL1 / rL, -dH0 / rH
    c = np.linalg.solve(M, rhs)

    def fields(y):
        y = np.asarray(y, dtype=float) % 1.0
        t = (y - a) % 1.0
        heavy = t < LH
        p, dp, rho = np.empty_like(y), np.empty_like(y), np.where(heavy, rH, rL)
        for idx in range(y.size):
            if heavy.flat[idx]:
                e, d = hom(t.flat[idx], LH)
                co, r = c[:2], rH
            else:
                e, d = hom(t.flat[idx] - LH, LL)
                co, r = c[2:], rL
            p.flat[idx] = r * A * trig(2 * math.pi * y.flat[idx]) + co @ e
            dp.flat[idx] = r * A * 2 * math.pi * dtrig(2 * math.pi * y.flat[idx]) + co @ d
        return p, dp, rho
    return fields, k


def projection_test(N, ratio, a, b, kind, m=1, source="normal"):
    """Discrete G-orthogonal projection of the exact face averages of w against the exact face averages of the
    projected field. kind "M": the lumped reference MAC (G_u = h M_t, G_v = h M_n). Max errors of the u faces and the
    v faces (the input has max |w| = 1)."""
    h = 1.0 / N
    fields, k = exact_projection(ratio, a, b, m, source)
    th = k * h
    phi = (np.exp(1j * th) - 1.0) / (1j * th)
    D = np.roll(np.eye(N), -1, axis=0) - np.eye(N)
    if kind == "M":
        Mn, Mt, _ = references(N, ratio, a, b)
        Gu, Gv = np.diag(h * Mt).astype(complex), np.diag(h * Mn).astype(complex)
    else:
        R = build(N, ratio, a, b, kind)
        wr = R["w"] * R["rho"]
        GF = R["PhiF"].T @ (wr[:, None] * R["PhiF"])
        GC = R["PhiC"].T @ (wr[:, None] * R["PhiC"])
        if kind in ("B", "D5m", "D5b"):
            from cases.mac_pccd import g_C, g_F
            gFx, gCx = g_F(th), g_C(th)
        else:
            gFx, gCx = (2 + math.cos(th)) / 3, 1.0
        Gu, Gv = (h * gFx * GC).astype(complex), (h * gCx * GF).astype(complex)
    G = np.block([[Gu, np.zeros((N, N))], [np.zeros((N, N)), Gv]])
    Bm = np.hstack([(np.exp(1j * th) - 1.0) / h * np.eye(N), D / h]).astype(complex)
    yf = np.arange(N) * h
    if source == "normal":
        x = np.concatenate([np.zeros(N), np.cos(2 * math.pi * yf) * phi])
    else:                                                      # exact y-cell averages of cos 2 pi y
        x = np.concatenate([(np.sin(2 * math.pi * (yf + h)) - np.sin(2 * math.pi * yf)) / (2 * math.pi * h), np.zeros(N)])
    GiBt = np.linalg.solve(G, Bm.conj().T)
    lam = np.linalg.solve(Bm @ GiBt, Bm @ x)
    xp = x - GiBt @ lam
    # exact: v faces (points in y), u faces (volume averages over the y-cells, Gauss per phase piece)
    p, dp, rho = fields(yf)
    # normal source: u = -i k p / rho, v = w2 - p' / rho. Tangential source (p = i q): u = w1 + k q / rho, v = -i q' / rho
    v_ex = (np.cos(2 * math.pi * yf) - dp / rho) * phi if source == "normal" else (-1j * dp / rho) * phi
    u_ex = np.zeros(N, complex)
    for i, ps in enumerate(pieces(N, a, b)):
        for s0, s1, hv in ps:
            for q in range(4):
                lo, hi = s0 + (s1 - s0) * q / 4, s0 + (s1 - s0) * (q + 1) / 4
                yq = (i + lo + (hi - lo) * XG) * h
                pq, _, rq = fields(yq)
                if source == "normal":
                    u_ex[i] += (hi - lo) * (WG @ (-1j * k * pq / rq))
                else:
                    u_ex[i] += (hi - lo) * (WG @ (np.cos(2 * math.pi * yq) + k * pq / rq))
    eu, ev = np.abs(xp[:N] - u_ex), np.abs(xp[N:] - v_ex)
    return {"N": N, "ratio": ratio, "kind": kind, "m": m, "u_err": float(eu.max()), "v_err": float(ev.max()),
            "u_rms": float(np.sqrt((eu ** 2).mean())), "v_rms": float(np.sqrt((ev ** 2).mean())),
            "u_max": float(np.abs(u_ex).max()), "div": float(np.abs(Bm @ xp).max() * h)}


def projection_scan():
    from concurrent.futures import ProcessPoolExecutor

    NS = (16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192)
    OFF = (0.0, 0.0071, 0.0193)
    jobs = [(N, r, 0.2613 + o, 0.7137 + o, k, m) for m in (1, 2) for k in ("M",) + KINDS
            for r in (1.0, 1e3, 1e6) for o in OFF for N in NS]
    tjobs = [(N, 1e6, 0.2613 + o, 0.7137 + o, k, 2, "tangential") for k in ("M",) + KINDS for o in OFF for N in NS]
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(_pt, jobs))
        tres = list(ex.map(_pt, tjobs))
    tan = []
    for k in ("M",) + KINDS:
        row = {"kind": k, "ratio": 1e6, "m": 2, "source": "tangential"}
        for key in ("u_err", "v_err", "u_rms", "v_rms"):
            worst = [max(o[key] for j, o in zip(tjobs, tres) if j[4] == k and j[0] == N) for N in NS]
            ln, le = np.log(np.array(NS, float)), np.log(np.array(worst))
            row[key] = worst
            row[key + "_order"] = [float(-np.polyfit(ln, le, 1)[0]), float(-np.polyfit(ln[8:], le[8:], 1)[0])]
        tan.append(row)
    out = []
    for m, k, r in [(m, k, r) for m in (1, 2) for k in ("M",) + KINDS for r in (1.0, 1e3, 1e6)]:
        if True:
            row = {"kind": k, "ratio": r, "m": m}
            for key in ("u_err", "v_err", "u_rms", "v_rms"):
                worst = [max(o[key] for o in res if o["kind"] == k and o["ratio"] == r and o["N"] == N and o["m"] == m) for N in NS]
                ln, le = np.log(np.array(NS, float)), np.log(np.array(worst))
                row[key] = worst
                row[key + "_order"] = [float(-np.polyfit(ln, le, 1)[0]), float(-np.polyfit(ln[8:], le[8:], 1)[0])]
            row["u_max"] = max(o["u_max"] for o in res if o["kind"] == k and o["ratio"] == r and o["m"] == m)
            out.append(row)
    return {"NS": NS, "rows": out, "tangential": tan}


def _pt(j):
    return projection_test(*j)


# --------------------------------------------------------------------------------------------- 2-D Schur complement
def schur_2d(N, ratio, a, b, kind, ntheta=17):
    """theta_x != 0: L = |b_x|^2 G_u^{-1} + D_y G_v^{-1} D_y^T / h^2 with G_u = h g_Fx G_C, G_v = h g_Cx G_F, against
    L_ref built from the fixed references (G_u = h M_t, G_v = h M_n). Over theta_x in (0, pi]: the generalised
    eigenvalues (min, max) and the Jacobi-scaled condition numbers of L and of L_ref."""
    R = build(N, ratio, a, b, kind)
    h = R["h"]
    wr = R["w"] * R["rho"]
    GF = R["PhiF"].T @ (wr[:, None] * R["PhiF"])
    GC = R["PhiC"].T @ (wr[:, None] * R["PhiC"])
    Mn, Mt, _ = references(N, ratio, a, b)
    Dy = R["D"] / h
    GFi, GCi = np.linalg.inv(GF), np.linalg.inv(GC)
    out = {"min": np.inf, "max": 0.0, "jacobi_L": 0.0, "jacobi_Lref": 0.0}
    for th in np.linspace(math.pi / (ntheta - 1), math.pi, ntheta - 1):
        if kind in ("B", "D5m", "D5b"):
            from cases.mac_pccd import g_C, g_F
            gFx, gCx = g_F(th), g_C(th)
        else:
            gFx, gCx = (2 + math.cos(th)) / 3, 1.0
        bx2 = (2 * math.sin(th / 2) / h) ** 2
        L = bx2 * GCi / (h * gFx) + Dy @ GFi @ Dy.T / (h * gCx)
        Lr = bx2 * np.diag(1.0 / (h * Mt)) + Dy @ np.diag(1.0 / (h * Mn)) @ Dy.T
        C = np.linalg.inv(np.linalg.cholesky(Lr))
        ev = np.linalg.eigvalsh(C @ (0.5 * (L + L.T)) @ C.T)
        out["min"], out["max"] = min(out["min"], float(ev[0])), max(out["max"], float(ev[-1]))
        out["jacobi_L"] = max(out["jacobi_L"], _jacobi_cond(L))
        out["jacobi_Lref"] = max(out["jacobi_Lref"], _jacobi_cond(Lr))
    return out


def schur_2d_scan(N=32):
    from concurrent.futures import ProcessPoolExecutor

    cs = [None, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-2, 1 - 1e-2, 1e-4, 1 - 1e-4]
    zetas = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    jobs = []
    for kind in S2KINDS:
        for r in RATIOS:
            for tag, v in [("c", c) for c in cs] + [("zeta", z) for z in zetas if r > 1] + [("zeta_light", z) for z in zetas]:
                c = v if tag == "c" else (v / (r + v) if tag == "zeta" else r / (r + v))
                jobs.append((kind, r, *geometry(N, c, c)))
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(_s2, [(N, j[1], j[2], j[3], j[0]) for j in jobs]))
    out = []
    for kind in S2KINDS:
        row = {"kind": kind}
        for key, fn in (("min", min), ("max", max), ("jacobi_L", max), ("jacobi_Lref", max)):
            vals = [fn(o[key] for (k, r, _, _), o in zip(jobs, res) if k == kind and r == rr) for rr in RATIOS]
            row[key] = vals
            row["slope_" + key] = slopes(vals)
        out.append(row)
    return out


def _s2(j):
    return schur_2d(*j)


def projection_slivers(N=64):
    """Worst projection errors over the interface positions of the Gram scan (fractions and both zeta scans),
    both sources, m = 2."""
    from concurrent.futures import ProcessPoolExecutor

    cs = [None, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-2, 1 - 1e-2, 1e-4, 1 - 1e-4]
    zetas = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    jobs = []
    for kind in ("M", "A", "D0m", "D0b", "D1", "D5m", "D5b"):
        for r in (1e3, 1e6):
            for tag, v in [("c", c) for c in cs] + [("zeta", z) for z in zetas] + [("zeta_light", z) for z in zetas]:
                c = v if tag == "c" else (v / (r + v) if tag == "zeta" else r / (r + v))
                for src in ("normal", "tangential"):
                    jobs.append((N, r, *geometry(N, c, c), kind, 2, src))
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(_pt, jobs))
    out = []
    for kind in ("M", "A", "D0m", "D0b", "D1", "D5m", "D5b"):
        for r in (1e3, 1e6):
            sel = [o for j, o in zip(jobs, res) if j[4] == kind and j[1] == r]
            out.append({"kind": kind, "ratio": r, "u_err": max(o["u_err"] for o in sel), "v_err": max(o["v_err"] for o in sel),
                        "u_rms": max(o["u_rms"] for o in sel), "v_rms": max(o["v_rms"] for o in sel)})
    return out


def switch_continuity(N=32, ratio=3.0, eps=1e-7):
    """The piece values of R_C in the cut cell as the heavy fraction c crosses the equal-mobility point
    (c = ratio (1 - c)): max |row(c + eps) - row(c)| / eps over c in [0.55, 0.95]. A jump gives ~ 1 / eps."""
    out = {}
    for kind in ("D0m", "D0b"):                                  # (the high-order kinds share these two rules)
        worst = 0.0
        for c in np.linspace(0.55, 0.95, 401):
            rows = []
            for cc in (c, c + eps):
                a, b = geometry(N, 0.5, cc)                      # move the cut at b only
                P = pieces(N, a, b)
                i = (3 * N) // 4
                rows.append(np.array([r for _, _, _, r in _piece_values(N, P, ratio, kind)[i]]))
            worst = max(worst, float(np.abs(rows[1] - rows[0]).max() / eps))
        out[kind] = worst
    return out


def slopes(vals):
    lr, lv = np.log10(np.array(RATIOS)), np.log10(np.maximum(np.array(vals, float), 1e-300))
    allr = float(np.polyfit(lr, lv, 1)[0])
    high = float(np.polyfit(lr[3:], lv[3:], 1)[0])
    loc = np.diff(lv) / np.diff(lr)
    return {"all": allr, "high": high, "local_max_abs": float(np.abs(loc).max())}


def scan(N=32):
    """All candidates over the density ratios and interface positions; the worst value over the positions per ratio
    and the slopes."""
    from concurrent.futures import ProcessPoolExecutor

    cs = [None, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-2, 1 - 1e-2, 1e-4, 1 - 1e-4]
    zetas = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    jobs = []
    for kind in KINDS:
        for r in RATIOS:
            pos = ([("c", c) for c in cs] + [("zeta", z) for z in zetas if r > 1]
                   + [("zeta_light", z) for z in zetas])
            for tag, v in pos:
                # zeta = r c / (1 - c) (thin heavy part); zeta_light = r (1 - c) / c (thin light part, mobility share)
                c = v if tag == "c" else (v / (r + v) if tag == "zeta" else r / (r + v))
                a, b = geometry(N, c, c)
                jobs.append((kind, r, tag, v, a, b))
    rows = []
    with ProcessPoolExecutor(20) as ex:
        fs = [(j, ex.submit(measure, N, j[1], j[4], j[5], j[0])) for j in jobs]
        for (kind, r, tag, v, a, b), f in fs:
            o = f.result()
            o.update({"pos": tag, "value": v})
            rows.append(o)
    summary = []
    for kind in KINDS:
        for key in ("GF_Mn", "GF_Mpt", "GC_Mt", "L_Lref"):
            mins = [min(o[key][0] for o in rows if o["kind"] == kind and o["ratio"] == r) for r in RATIOS]
            maxs = [max(o[key][1] for o in rows if o["kind"] == kind and o["ratio"] == r) for r in RATIOS]
            summary.append({"kind": kind, "key": key, "min": mins, "max": maxs, "slope_min": slopes(mins),
                            "slope_max": slopes(maxs), "worst_min": float(min(mins)), "worst_max": float(max(maxs))})
        for key in ("jacobi_GF", "jacobi_GC", "jacobi_L"):
            worst = [max(o[key] for o in rows if o["kind"] == kind and o["ratio"] == r) for r in RATIOS]
            summary.append({"kind": kind, "key": key, "max": worst, "slope_max": slopes(worst),
                            "worst_max": float(max(worst))})
    return rows, summary


def main() -> None:
    res = {"leak": leak_table(), "fixed": [measure(32, 1e6, *geometry(32, 0.75, 0.75), k) for k in KINDS]}
    for r, rows in res["leak"].items():
        print("leak", r, " ".join(f"{x['rayleigh']:.4g}({x['heavy_part']:.4g})" for x in rows))
    for o in res["fixed"]:
        print("c=0.75 r=1e6", o["kind"], "GF/Mn %.3g..%.3g  GC/Mt %.3g..%.3g  L/Lref %.3g..%.3g  null %d  jacobi GF %.3g L %.3g (Lref %.3g)"
              % (*o["GF_Mn"], *o["GC_Mt"], *o["L_Lref"], o["null_L"], o["jacobi_GF"], o["jacobi_L"], o["jacobi_Lref"]))
    for N in (32, 64):
        rows, summary = scan(N)
        res[f"scan_{N}"] = {"rows": rows, "summary": summary}
        print("N", N, "reproduction errors", max(o["avg_err"] for o in rows), max(o["face_err"] for o in rows),
              "null_L max", max(o["null_L"] for o in rows), "constants (R_F / R_C):",
              {k: ("%.1e" % max(o["const_F"] for o in rows if o["kind"] == k), "%.1e" % max(o["const_C"] for o in rows if o["kind"] == k))
               for k in KINDS})
        for sm in summary:
            mn = ("min %.2e [all ratios %.2e] (slope %+.2f/%+.2f) " % (sm["min"][-1], sm["worst_min"], sm["slope_min"]["all"], sm["slope_min"]["high"])) if "min" in sm else ""
            print(" ", sm["kind"], sm["key"], mn + "max %.2e [all ratios %.2e] (slope %+.2f/%+.2f)"
                  % (sm["max"][-1], sm["worst_max"], sm["slope_max"]["all"], sm["slope_max"]["high"]))
    res["schur_2d"] = schur_2d_scan(32)
    for row in res["schur_2d"]:
        print("schur 2-D", row["kind"], "min %.2e (%+.2f/%+.2f) max %.2e (%+.2f/%+.2f) jacobi L %.3g..%.3g (%+.2f/%+.2f), Lref %.3g..%.3g"
              % (row["min"][-1], row["slope_min"]["all"], row["slope_min"]["high"], row["max"][-1], row["slope_max"]["all"],
                 row["slope_max"]["high"], min(row["jacobi_L"]), max(row["jacobi_L"]), row["slope_jacobi_L"]["all"],
                 row["slope_jacobi_L"]["high"], min(row["jacobi_Lref"]), max(row["jacobi_Lref"])))
    res["projection"] = projection_scan()
    for row in res["projection"]["rows"]:
        print("projection m=%d" % row["m"], row["kind"], row["ratio"],
              "max: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f) | rms: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f)"
              % (row["u_err"][0], row["u_err"][-1], *row["u_err_order"], row["v_err"][0], row["v_err"][-1], *row["v_err_order"],
                 row["u_rms"][0], row["u_rms"][-1], *row["u_rms_order"], row["v_rms"][0], row["v_rms"][-1], *row["v_rms_order"]))
    for row in res["projection"]["tangential"]:
        print("projection tangential source m=2 1e6", row["kind"], "max: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f) | rms: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f)"
              % (row["u_err"][0], row["u_err"][-1], *row["u_err_order"], row["v_err"][0], row["v_err"][-1], *row["v_err_order"],
                 row["u_rms"][0], row["u_rms"][-1], *row["u_rms_order"], row["v_rms"][0], row["v_rms"][-1], *row["v_rms_order"]))
    res["projection_slivers"] = projection_slivers()
    for row in res["projection_slivers"]:
        print("projection, all positions and both sources", row["kind"], row["ratio"],
              "max u %.2e v %.2e | rms u %.2e v %.2e" % (row["u_err"], row["v_err"], row["u_rms"], row["v_rms"]))
    res["switch_continuity"] = switch_continuity()
    print("switch continuity (max rate)", res["switch_continuity"])
    json.dump(res, open("docs/figures/mac_layered.json", "w"), indent=1, default=float)


def figure() -> None:
    """docs/figures/mac_layered.png: worst generalised eigenvalues against the fixed references, the reference-free
    Jacobi-scaled condition numbers, and the projection error against N (density ratio 1e6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"A": "#257de0", "B": "#f47230", "D0": "#00b894", "D1": "#f2a600", "M": "#666666", "D5m": "#ee7fac"}
    res = json.load(open("docs/figures/mac_layered.json"))
    sm = res["scan_32"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=150)
    col["D0m"] = col["D0"]
    for k in ("A", "B", "D0m", "D5m"):
        g = next(x for x in sm if x["kind"] == k and x["key"] == "GF_Mn")
        axes[0].loglog(RATIOS, g["max"], "-o", color=col[k], lw=2, ms=4, label=f"{k} max")
        axes[0].loglog(RATIOS, g["min"], "--s", color=col[k], lw=1.5, ms=4, label=f"{k} min")
        for key, ls in (("jacobi_GF", "-o"), ("jacobi_L", "--s")):
            j = next(x for x in sm if x["kind"] == k and x["key"] == key)
            axes[1].loglog(RATIOS, j["max"], ls, color=col[k], lw=2 if ls == "-o" else 1.5, ms=4,
                           label=f"{k} {'Gram' if key == 'jacobi_GF' else 'Schur'}")
    axes[0].set_title("normal Gram vs dual-cell mass M_n (worst position, N = 32)", fontsize=9)
    axes[1].set_title("reference-free: Jacobi-scaled condition number", fontsize=9)
    for a in axes[:2]:
        a.set_xlabel("density ratio", fontsize=8)
    pr = res["projection"]
    for k in ("M", "A", "B", "D0m", "D1", "D5m"):
        row = next(x for x in pr["rows"] if x["kind"] == k and x["ratio"] == 1e6 and x["m"] == 2)
        axes[2].loglog(pr["NS"], row["u_rms"], "-o", color=col[k], lw=2, ms=4, label=f"{k} (u, rms)")
    axes[2].set_title("projection error, ratio 1e6, m = 2 (worst of 3 offsets)", fontsize=9)
    axes[2].set_xlabel("N", fontsize=8)
    from matplotlib.ticker import NullFormatter
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xticks([16, 32, 64, 128, 192])
    axes[2].set_xticklabels(["16", "32", "64", "128", "192"])
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
        a.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig("docs/figures/mac_layered.png")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()

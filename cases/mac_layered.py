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
KINDS = ("A", "B", "D0", "D0m", "D1")


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
        close_heavy = kind == "D0m" and mob_h > mob_l
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
        light = (e - used) / llen
        out.append([(s0, s1, hv, light if r is None else r) for (s0, s1, hv), r in zip(ps, rows)])
    return out


def build(N, ratio, a, b, kind):
    """Quadrature values of the bases: PhiC (points x N cells) for R_C, PhiF (points x N faces) for R_F, with the
    geometric weights w, the density rho, and the reproduction errors."""
    h = 1.0 / N
    P = pieces(N, a, b)
    D = np.roll(np.eye(N), -1, axis=0) - np.eye(N)              # (D f)_i = f_{i+1} - f_i
    X, W, RHO, PC, PF = [], [], [], [], []
    face_err = avg_err = 0.0
    if kind == "B":
        from cases.mac_pccd import operators

        RF, RC, _, _, _ = operators(N)
        for i, ps in enumerate(P):
            for s0, s1, hv in ps:
                s = s0 + (s1 - s0) * XG
                X.append((i + s) * h), W.append((s1 - s0) * h * WG)
                RHO.append(np.full(XG.size, 1.0 if hv else 1.0 / ratio))
                PC.append(_mono(s) @ RC[i])
                PF.append(np.stack([s ** p for p in range(6)], axis=-1) @ RF[i])
    else:
        vals = _piece_values(N, P, ratio, kind)
        for i, cell in enumerate(vals):
            cum = np.zeros(N)                                   # int_0^{s0} R_C (per unit cell data)
            e = np.zeros(N)
            e[i] = 1.0
            for s0, s1, hv, row in cell:
                s = s0 + (s1 - s0) * XG
                X.append((i + s) * h), W.append((s1 - s0) * h * WG)
                RHO.append(np.full(XG.size, 1.0 if hv else 1.0 / ratio))
                PC.append(np.tile(row, (XG.size, 1)))
                PF.append(e[None, :] + (cum[None, :] + np.outer(s - s0, row)) @ D)
                cum = cum + (s1 - s0) * row
            avg_err = max(avg_err, float(np.abs(cum - e).max()))
            e1 = np.zeros(N)
            e1[(i + 1) % N] = 1.0
            face_err = max(face_err, float(np.abs(e + cum @ D - e1).max()))
    X, W, RHO = np.concatenate(X), np.concatenate(W), np.concatenate(RHO)
    PC, PF = np.concatenate(PC), np.concatenate(PF)
    return {"X": X, "w": W, "rho": RHO, "PhiC": PC, "PhiF": PF, "D": D, "h": h, "P": P,
            "avg_err": avg_err, "face_err": face_err}


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
def exact_projection(ratio, a, b, m=1):
    """The density-weighted Helmholtz projection of w = (0, cos 2 pi y) e^{i k x}, k = 2 pi m, in the layered medium
    (heavy rho = 1 on [a, b), light 1 / ratio elsewhere, periodic): u = w - grad p / rho, div u = 0, i.e.
    (p' / rho)' - k^2 p / rho = w2'. Per layer p = rho A sin 2 pi y + alpha e^{k (y - y0)} + beta e^{-k (y - y1)},
    A = 2 pi / (4 pi^2 + k^2); p and p' / rho continuous at both interfaces (a 4 x 4 real system).
    Returns p(y), dp(y), rho(y)."""
    k = 2 * math.pi * m
    A = 2 * math.pi / (4 * math.pi ** 2 + k * k)
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
    M[0, :2], M[0, 2:], rhs[0] = eH1, -eL0, (rL - rH) * A * math.sin(2 * math.pi * b)
    M[1, :2], M[1, 2:] = dH1 / rH, -dL0 / rL
    # at y = a (the light layer's end): p_L = p_H, fluxes
    M[2, 2:], M[2, :2], rhs[2] = eL1, -eH0, (rH - rL) * A * math.sin(2 * math.pi * a)
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
            p.flat[idx] = r * A * math.sin(2 * math.pi * y.flat[idx]) + co @ e
            dp.flat[idx] = r * A * 2 * math.pi * math.cos(2 * math.pi * y.flat[idx]) + co @ d
        return p, dp, rho
    return fields, k


def projection_test(N, ratio, a, b, kind, m=1):
    """Discrete G-orthogonal projection of the exact face averages of w against the exact face averages of the
    projected field. kind "M": the lumped reference MAC (G_u = h M_t, G_v = h M_n). Max errors of the u faces and the
    v faces (the input has max |w| = 1)."""
    h = 1.0 / N
    fields, k = exact_projection(ratio, a, b, m)
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
        if kind == "B":
            from cases.mac_pccd import g_C, g_F
            gFx, gCx = g_F(th), g_C(th)
        else:
            gFx, gCx = (2 + math.cos(th)) / 3, 1.0
        Gu, Gv = (h * gFx * GC).astype(complex), (h * gCx * GF).astype(complex)
    G = np.block([[Gu, np.zeros((N, N))], [np.zeros((N, N)), Gv]])
    Bm = np.hstack([(np.exp(1j * th) - 1.0) / h * np.eye(N), D / h]).astype(complex)
    yf = np.arange(N) * h
    x = np.concatenate([np.zeros(N), np.cos(2 * math.pi * yf) * phi])
    GiBt = np.linalg.solve(G, Bm.conj().T)
    lam = np.linalg.solve(Bm @ GiBt, Bm @ x)
    xp = x - GiBt @ lam
    # exact: v faces (points in y), u faces (volume averages over the y-cells, Gauss per phase piece)
    p, dp, rho = fields(yf)
    v_ex = (np.cos(2 * math.pi * yf) - dp / rho) * phi
    u_ex = np.zeros(N, complex)
    for i, ps in enumerate(pieces(N, a, b)):
        for s0, s1, hv in ps:
            for q in range(4):
                lo, hi = s0 + (s1 - s0) * q / 4, s0 + (s1 - s0) * (q + 1) / 4
                yq = (i + lo + (hi - lo) * XG) * h
                pq, _, rq = fields(yq)
                u_ex[i] += (hi - lo) * (WG @ (-1j * k * pq / rq))
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
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(_pt, jobs))
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
    return {"NS": NS, "rows": out}


def _pt(j):
    return projection_test(*j)


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
    res["projection"] = projection_scan()
    for row in res["projection"]["rows"]:
        print("projection m=%d" % row["m"], row["kind"], row["ratio"],
              "max: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f) | rms: u %.2e -> %.2e (%.2f / %.2f) v %.2e -> %.2e (%.2f / %.2f)"
              % (row["u_err"][0], row["u_err"][-1], *row["u_err_order"], row["v_err"][0], row["v_err"][-1], *row["v_err_order"],
                 row["u_rms"][0], row["u_rms"][-1], *row["u_rms_order"], row["v_rms"][0], row["v_rms"][-1], *row["v_rms_order"]))
    json.dump(res, open("docs/figures/mac_layered.json", "w"), indent=1, default=float)


def figure() -> None:
    """docs/figures/mac_layered.png: worst generalised eigenvalues against the fixed references, the reference-free
    Jacobi-scaled condition numbers, and the projection error against N (density ratio 1e6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"A": "#257de0", "B": "#f47230", "D0": "#00b894", "D1": "#f2a600", "M": "#666666"}
    res = json.load(open("docs/figures/mac_layered.json"))
    sm = res["scan_32"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=150)
    col["D0m"] = col["D0"]
    for k in ("A", "B", "D0m"):
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
    for k in ("M", "A", "B", "D0m", "D1"):
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

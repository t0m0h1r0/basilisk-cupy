"""docs/notes/109 (issue #134, T4): the transform from the MAC face velocities to the momenta of control volumes,
layered two-phase (density depends on y), with the reconstructions of cases.mac_layered.

    (P f)_i = int_{V_i} rho R_1 f dV,   M = diag(int_{V_i} rho dV),   Q = M^{-1/2} P G^{-1/2}

Theorem 7 (docs/notes/107): if the control volumes of each velocity component partition the domain, 0 <= sigma(Q)
<= 1. Two sets of control volumes:
  dual    the usual MAC dual cells: for the normal velocity [y_j - h/2, y_j + h/2], for the tangential velocity the
          y-cell (times the dual x-interval);
  phase   phase-aware, following the ownership of the D0m reconstruction: in a cut cell the phase with the larger
          mobility (length / rho) keeps the usual assignment (nearest face; own cell), the other phase goes to the
          face it touches (normal) and to its adjacent pure cell (tangential). Pure cells are unchanged.
Both partition the domain, so the theorem applies to both.

    python -m cases.mac_momentum
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.mac_layered import RATIOS, build, geometry, slopes


def _mobility(ps, ratio):
    mob_h = sum(s1 - s0 for s0, s1, hv in ps if hv)
    mob_l = sum(s1 - s0 for s0, s1, hv in ps if not hv) * ratio
    return mob_h, mob_l


def _blend_weight(ps, ratio):
    """The weight of "the heavy phase restores the average" of D0b / D5b (cases.mac_layered)."""
    mob_h, mob_l = _mobility(ps, ratio)
    t = min(max((mob_h / (mob_h + mob_l) - 0.25) / 0.5, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _assign(N, R, ratio, mode):
    """Control volumes as weighted assignments: a list of (index_normal, index_tangential, weight per point).
    "dual": the MAC dual cells. "phase": the hard D0m rule. "blend": the two hard rules ("light restores" /
    "heavy restores") weighted like the D0b reconstruction (a weighted control volume, sum of weights = 1)."""
    h = R["h"]
    X, P = R["X"], R["P"]
    cell = np.floor(X / h + 1e-12).astype(int) % N
    s = X / h - np.floor(X / h + 1e-12)
    heavy_pt = R["heavy"]                                        # from the geometry (valid at density ratio 1)
    dual_n = np.where(s < 0.5, cell, (cell + 1) % N)
    if mode == "dual":
        return [(dual_n, cell.copy(), np.ones(X.size))]
    pure = [len(ps) == 1 for ps in P]

    def rule(stay):
        """stay[i]: the phase (True = heavy) that keeps the usual assignment in cut cell i."""
        pn, pt = dual_n.copy(), cell.copy()
        for i, ps in enumerate(P):
            if len(ps) == 1:
                continue
            for s0, s1, hv in ps:
                if hv == stay[i]:
                    continue
                sel = (cell == i) & (heavy_pt == hv) & (s >= s0 - 1e-14) & (s <= s1 + 1e-14)
                if s0 == 0.0:
                    pn[sel] = i
                elif s1 == 1.0:
                    pn[sel] = (i + 1) % N
                k = None
                if s0 == 0.0 and pure[(i - 1) % N] and P[(i - 1) % N][0][2] == hv:
                    k = (i - 1) % N
                elif s1 == 1.0 and pure[(i + 1) % N] and P[(i + 1) % N][0][2] == hv:
                    k = (i + 1) % N
                if k is not None:
                    pt[sel] = k
        return pn, pt
    if mode == "phase":
        stay = [(_mobility(ps, ratio)[0] > _mobility(ps, ratio)[1]) if len(ps) > 1 else True for ps in P]
        return [(*rule(stay), np.ones(X.size))]
    wh = np.array([_blend_weight(ps, ratio) if len(ps) > 1 else 0.0 for ps in P])[cell]
    return [(*rule([False] * N), 1.0 - wh), (*rule([True] * N), wh)]


def _x_symbols(kind, th):
    """(p_Fx, g_Fx, p_Cx, g_Cx) for h = 1: the dual-x-interval integral and the Gram symbol of the face
    reconstruction (tangential component), the cell integral (= 1) and the Gram symbol of the cell one (normal)."""
    if kind in ("B", "D5m", "D5b"):
        from cases.mac_pccd import face_coeffs, g_C, g_F

        c = face_coeffs(th)
        p = np.arange(6)
        half = 0.5 ** (p + 1) / (p + 1)
        pF = (c * half).sum() + np.exp(-1j * th) * (c * (1.0 / (p + 1) - half)).sum()
        return float(abs(pF)), g_F(th), 1.0, (g_C(th) if th != 0 else 1.0)
    return (3 + math.cos(th)) / 4, (2 + math.cos(th)) / 3, 1.0, 1.0


def _sig(P, M, G):
    L = np.linalg.cholesky(0.5 * (G + G.T))
    return np.linalg.svd((P / np.sqrt(M)[:, None]) @ np.linalg.inv(L).T, compute_uv=False)


def measure(N, ratio, a, b, kind, ntheta=None, modes=("dual", "phase")):
    """1-D (y) singular values and Gram comparisons, and the 2-D values over all the discrete x-wavenumbers
    theta_x = 2 pi k / N, k = 0 .. N/2 (ntheta: a coarser set), full space and divergence-free subspace."""
    R = build(N, ratio, a, b, kind)
    h = R["h"]
    wr = R["w"] * R["rho"]
    GF = R["PhiF"].T @ (wr[:, None] * R["PhiF"])
    GC = R["PhiC"].T @ (wr[:, None] * R["PhiC"])
    out = {"N": N, "ratio": ratio, "kind": kind}
    D = R["D"] / h
    thetas = (2 * math.pi * np.arange(N // 2 + 1) / N) if ntheta is None else np.linspace(0.0, math.pi, ntheta)
    for name in modes:
        Mn, Mt = np.zeros(N), np.zeros(N)
        Pn, Pt = np.zeros((N, N)), np.zeros((N, N))
        for idx_n, idx_t, wgt in _assign(N, R, ratio, name):
            ww = wr * wgt
            Mn += np.bincount(idx_n, weights=ww, minlength=N)
            Mt += np.bincount(idx_t, weights=ww, minlength=N)
            np.add.at(Pn, idx_n, ww[:, None] * R["PhiF"])
            np.add.at(Pt, idx_t, ww[:, None] * R["PhiC"])
        sn, st = _sig(Pn, Mn, GF), _sig(Pt, Mt, GC)
        evn = np.linalg.eigvalsh(GF / np.sqrt(np.outer(Mn, Mn)))
        evt = np.linalg.eigvalsh(GC / np.sqrt(np.outer(Mt, Mt)))
        o = {"sig_n": [float(sn[-1]), float(sn[0])], "sig_t": [float(st[-1]), float(st[0])],
             "G_n": [float(evn[0]), float(evn[-1])], "G_t": [float(evt[0]), float(evt[-1])],
             "empty": int((Mn <= 0).sum() + (Mt <= 0).sum()),
             # the reference derived from the momenta, M_hat = P^T M^-1 P: (G, M_hat) lies in [1, 1 / sigma_min^2]
             "G_Mhat_n_max": float(1.0 / sn[-1] ** 2), "G_Mhat_t_max": float(1.0 / st[-1] ** 2)}
        full, divfree, smax = np.inf, np.inf, 0.0
        for th in thetas:
            pF, gF, pC, gC = _x_symbols(kind, th)
            fu, fv = pF / math.sqrt(gF), pC / math.sqrt(gC)
            full = min(full, fu * o["sig_t"][0], fv * o["sig_n"][0])
            smax = max(smax, fu * o["sig_t"][1], fv * o["sig_n"][1])
            # block system: u (tangential, R_C in y), v (normal, R_F in y); the common factor h cancels in Q
            G = np.block([[gF * GC, np.zeros((N, N))], [np.zeros((N, N)), gC * GF]]).astype(complex)
            P2 = np.block([[pF * Pt, np.zeros((N, N))], [np.zeros((N, N)), pC * Pn]]).astype(complex)
            M2 = np.concatenate([Mt, Mn])
            Bm = np.hstack([(np.exp(1j * th) - 1.0) / h * np.eye(N), D]).astype(complex)
            _, sv, Vh = np.linalg.svd(Bm)
            rank = int((sv > 1e-10 * sv[0]).sum())               # N, or N - 1 at theta_x = 0
            Z = Vh[rank:].conj().T                               # null space of the discrete divergence
            Gz = Z.conj().T @ G @ Z
            Lz = np.linalg.cholesky(0.5 * (Gz + Gz.conj().T))
            Qz = (P2 / np.sqrt(M2)[:, None]) @ Z @ np.linalg.inv(Lz).conj().T
            divfree = min(divfree, float(np.linalg.svd(Qz, compute_uv=False)[-1]))
        o.update({"sig2d_min": float(full), "sig2d_max": float(smax), "sig2d_divfree_min": float(divfree)})
        out[name] = o
    return out


COMBOS = (("A", "dual"), ("A", "phase"), ("B", "dual"), ("D0m", "dual"), ("D0m", "phase"), ("D0b", "phase"),
          ("D0b", "blend"), ("D5m", "dual"), ("D5m", "phase"), ("D5b", "phase"), ("D5b", "blend"))


def positions(r):
    """Fractions, the two sliver scans, and the neighbourhood of the mobility switch (zeta_light = r (1 - c) / c
    near 1, the user's request)."""
    cs = [None, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-2, 1 - 1e-2, 1e-4, 1 - 1e-4]
    zetas = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    near = (0.25, 0.5, 0.8, 0.95, 1.05, 1.25, 2.0, 4.0)
    out = [c for c in cs] + ([z / (r + z) for z in zetas] if r > 1 else []) + [r / (r + z) for z in zetas + near]
    return out


def scan(N=32):
    from concurrent.futures import ProcessPoolExecutor

    kinds = sorted({k for k, _ in COMBOS})
    jobs = [(N, r, *geometry(N, c, c), k, None, tuple(m for kk, m in COMBOS if kk == k))
            for k in kinds for r in RATIOS for c in positions(r)]
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(_m, jobs))
    summary = []
    for kind, name in COMBOS:
        row = {"kind": kind, "cv": name}
        sel = lambda r: [o[name] for o in res if o["kind"] == kind and o["ratio"] == r]  # noqa: E731
        for key, fn, pick in (("sig_n", min, ("sig_n", 0)), ("sig_t", min, ("sig_t", 0)), ("sig2d_min", min, None),
                              ("sig2d_divfree_min", min, None), ("sig2d_max", max, None),
                              ("G_n_min", min, ("G_n", 0)), ("G_n_max", max, ("G_n", 1)),
                              ("G_t_min", min, ("G_t", 0)), ("G_t_max", max, ("G_t", 1))):
            vals = [fn((x[pick[0]][pick[1]] if pick else x[key]) for x in sel(r)) for r in RATIOS]
            row[key] = vals
            row["slope_" + key] = slopes(vals)
        row["empty"] = max(x["empty"] for r in RATIOS for x in sel(r))
        summary.append(row)
    return res, summary


def _m(j):
    return measure(*j)


def main() -> None:
    out = {}
    for N in (32, 64):
        res, summary = scan(N)
        out[f"scan_{N}"] = {"rows": res, "summary": summary}
        print("N", N, "(all discrete theta_x; worst over the positions; density ratio 1 / 1e3 / 1e6)")
        for s in summary:
            f3 = lambda v: " / ".join("%.3g" % v[i] for i in (0, 3, 6))  # noqa: E731
            print("  %-4s %-5s sig_n %s | sig_t %s | 2-D %s | div-free %s | max %.3f | G_n max %s | G_t max %s (slope %+.2f)"
                  % (s["kind"], s["cv"], f3(s["sig_n"]), f3(s["sig_t"]), f3(s["sig2d_min"]), f3(s["sig2d_divfree_min"]),
                     max(s["sig2d_max"]), f3(s["G_n_max"]), f3(s["G_t_max"]), s["slope_G_t_max"]["high"]))
    json.dump(out, open("docs/figures/mac_momentum.json", "w"), indent=1, default=float)


def figure() -> None:
    """docs/figures/mac_momentum.png: the smallest normalised singular values (normal and tangential, 1-D; worst
    position) and the tangential Gram against the control-volume masses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"A": "#257de0", "B": "#f47230", "D0m": "#00b894", "D0b": "#f2a600", "D5m": "#ee7fac", "D5b": "#6a4ce2"}
    ls = {"dual": ":s", "phase": "-o", "blend": "--^"}
    sm = json.load(open("docs/figures/mac_momentum.json"))["scan_32"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=150)
    for s in sm:
        if (s["kind"], s["cv"]) in (("A", "phase"), ("D5m", "dual")):
            continue
        lab = f"{s['kind']}, {s['cv']}"
        axes[0].loglog(RATIOS, s["sig_n"], ls[s["cv"]], color=col[s["kind"]], lw=1.8, ms=4, label=lab)
        axes[1].loglog(RATIOS, s["sig_t"], ls[s["cv"]], color=col[s["kind"]], lw=1.8, ms=4, label=lab)
        axes[2].loglog(RATIOS, s["G_t_max"], ls[s["cv"]], color=col[s["kind"]], lw=1.8, ms=4, label=lab)
    axes[0].set_title("min sigma(Q), normal velocity", fontsize=9)
    axes[1].set_title("min sigma(Q), tangential velocity", fontsize=9)
    axes[2].set_title("tangential Gram / control-volume mass (max)", fontsize=9)
    for a in axes:
        a.set_xlabel("density ratio", fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig("docs/figures/mac_momentum.png")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()

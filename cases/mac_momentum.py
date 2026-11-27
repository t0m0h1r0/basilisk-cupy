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


def _assign(N, R, ratio, a, b):
    """Index of the control volume of every quadrature point: (dual_n, dual_t, phase_n, phase_t)."""
    h = R["h"]
    X, P = R["X"], R["P"]
    cell = np.floor(X / h + 1e-12).astype(int) % N
    s = X / h - np.floor(X / h + 1e-12)
    heavy_pt = R["rho"] > 0.999
    dual_n = np.where(s < 0.5, cell, (cell + 1) % N)
    dual_t = cell.copy()

    pure = [len(ps) == 1 for ps in P]
    phase_n = dual_n.copy()
    phase_t = cell.copy()
    for i, ps in enumerate(P):
        if len(ps) == 1:
            continue
        mob_h = sum(s1 - s0 for s0, s1, hv in ps if hv)
        mob_l = sum(s1 - s0 for s0, s1, hv in ps if not hv) * ratio
        stay_heavy = mob_h > mob_l                               # the phase that restores the average in D0m
        for s0, s1, hv in ps:
            if hv == stay_heavy:
                continue                                         # nearest face, own cell
            sel = (cell == i) & (heavy_pt == hv) & (s >= s0 - 1e-14) & (s <= s1 + 1e-14)
            # normal: the other phase goes to the face it touches
            if s0 == 0.0:
                phase_n[sel] = i
            elif s1 == 1.0:
                phase_n[sel] = (i + 1) % N
            # tangential: ... and to its adjacent pure cell
            k = None
            if s0 == 0.0 and pure[(i - 1) % N] and P[(i - 1) % N][0][2] == hv:
                k = (i - 1) % N
            elif s1 == 1.0 and pure[(i + 1) % N] and P[(i + 1) % N][0][2] == hv:
                k = (i + 1) % N
            if k is not None:
                phase_t[sel] = k
    return dual_n, dual_t, phase_n, phase_t


def _x_symbols(kind, th):
    """(p_Fx, g_Fx, p_Cx, g_Cx) for h = 1: the dual-x-interval integral and the Gram symbol of the face
    reconstruction (tangential component), the cell integral (= 1) and the Gram symbol of the cell one (normal)."""
    if kind == "B":
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


def measure(N, ratio, a, b, kind, ntheta=17):
    R = build(N, ratio, a, b, kind)
    h = R["h"]
    wr = R["w"] * R["rho"]
    GF = R["PhiF"].T @ (wr[:, None] * R["PhiF"])
    GC = R["PhiC"].T @ (wr[:, None] * R["PhiC"])
    dn, dt, pn, pt = _assign(N, R, ratio, a, b)
    out = {"N": N, "ratio": ratio, "kind": kind}
    mats = {}
    for name, idx_n, idx_t in (("dual", dn, dt), ("phase", pn, pt)):
        Mn = np.bincount(idx_n, weights=wr, minlength=N)
        Mt = np.bincount(idx_t, weights=wr, minlength=N)
        Pn = np.zeros((N, N))
        Pt = np.zeros((N, N))
        np.add.at(Pn, idx_n, wr[:, None] * R["PhiF"])
        np.add.at(Pt, idx_t, wr[:, None] * R["PhiC"])
        sn, st = _sig(Pn, Mn, GF), _sig(Pt, Mt, GC)
        evn = np.linalg.eigvalsh(GF / np.sqrt(np.outer(Mn, Mn)))
        evt = np.linalg.eigvalsh(GC / np.sqrt(np.outer(Mt, Mt)))
        out[name] = {"sig_n": [float(sn[-1]), float(sn[0])], "sig_t": [float(st[-1]), float(st[0])],
                     "G_n": [float(evn[0]), float(evn[-1])], "G_t": [float(evt[0]), float(evt[-1])],
                     "empty": int((Mn <= 0).sum() + (Mt <= 0).sum())}
        mats[name] = (Pn, Mn, Pt, Mt)
    # 2-D: all theta_x; the full space and the divergence-free subspace (dual and phase control volumes)
    D = R["D"] / h
    for name in ("dual", "phase"):
        Pn, Mn, Pt, Mt = mats[name]
        full, divfree, smax = np.inf, np.inf, 0.0
        for th in np.linspace(0.0, math.pi, ntheta):
            pF, gF, pC, gC = _x_symbols(kind, th)
            fu, fv = pF / math.sqrt(gF), pC / math.sqrt(gC)
            full = min(full, fu * out[name]["sig_t"][0], fv * out[name]["sig_n"][0])
            smax = max(smax, fu * out[name]["sig_t"][1], fv * out[name]["sig_n"][1])
            if th == 0.0:
                continue
            # block system: u (tangential, R_C in y), v (normal, R_F in y); the common factor h cancels in Q
            G = np.block([[gF * GC, np.zeros((N, N))], [np.zeros((N, N)), gC * GF]]).astype(complex)
            P2 = np.block([[pF * Pt, np.zeros((N, N))], [np.zeros((N, N)), pC * Pn]]).astype(complex)
            M2 = np.concatenate([Mt, Mn])
            Bm = np.hstack([(np.exp(1j * th) - 1.0) / h * np.eye(N), D]).astype(complex)
            _, sv, Vh = np.linalg.svd(Bm)
            Z = Vh[N:].conj().T                                  # null space of the discrete divergence (N columns)
            Gz = Z.conj().T @ G @ Z
            Lz = np.linalg.cholesky(0.5 * (Gz + Gz.conj().T))
            Qz = (P2 / np.sqrt(M2)[:, None]) @ Z @ np.linalg.inv(Lz).conj().T
            divfree = min(divfree, float(np.linalg.svd(Qz, compute_uv=False)[-1]))
        out[name].update({"sig2d_min": float(full), "sig2d_max": float(smax), "sig2d_divfree_min": float(divfree)})
    return out


def scan(N=32):
    from concurrent.futures import ProcessPoolExecutor

    cs = [None, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-2, 1 - 1e-2, 1e-4, 1 - 1e-4]
    zetas = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    jobs = []
    for kind in ("A", "B", "D0m"):
        for r in RATIOS:
            for tag, v in [("c", c) for c in cs] + [("zeta", z) for z in zetas if r > 1] + [("zeta_light", z) for z in zetas]:
                c = v if tag == "c" else (v / (r + v) if tag == "zeta" else r / (r + v))
                jobs.append((N, r, *geometry(N, c, c), kind))
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(_m, jobs))
    summary = []
    for kind in ("A", "B", "D0m"):
        for name in ("dual", "phase"):
            row = {"kind": kind, "cv": name}
            sel = lambda r: [o[name] for o in res if o["kind"] == kind and o["ratio"] == r]  # noqa: E731
            for key, fn, pick in (("sig2d_min", min, None), ("sig2d_divfree_min", min, None), ("sig2d_max", max, None),
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
    fixed = {k: {r: measure(32, r, *geometry(32, 0.75, 0.75), k) for r in (1.0, 1e3, 1e6)} for k in ("A", "B", "D0m")}
    for k, d in fixed.items():
        print("c = 0.75", k, "dual: 2-D min", " / ".join("%.3g" % d[r]["dual"]["sig2d_min"] for r in d),
              "div-free", " / ".join("%.3g" % d[r]["dual"]["sig2d_divfree_min"] for r in d),
              "| phase: 2-D min", " / ".join("%.3g" % d[r]["phase"]["sig2d_min"] for r in d),
              "div-free", " / ".join("%.3g" % d[r]["phase"]["sig2d_divfree_min"] for r in d))
    out = {"fixed": {k: {str(r): v for r, v in d.items()} for k, d in fixed.items()}}
    for N in (32, 64):
        res, summary = scan(N)
        out[f"scan_{N}"] = {"rows": res, "summary": summary}
        print("N", N)
        for s in summary:
            print("  %-3s %-5s sigma min %.2e (%+.2f/%+.2f) div-free %.2e (%+.2f/%+.2f) max %.3f | G_n %.2e..%.2e (%+.2f, %+.2f) G_t %.2e..%.2e (%+.2f, %+.2f) empty %d"
                  % (s["kind"], s["cv"], min(s["sig2d_min"]), s["slope_sig2d_min"]["all"], s["slope_sig2d_min"]["high"],
                     min(s["sig2d_divfree_min"]), s["slope_sig2d_divfree_min"]["all"], s["slope_sig2d_divfree_min"]["high"],
                     max(s["sig2d_max"]), min(s["G_n_min"]), max(s["G_n_max"]), s["slope_G_n_min"]["high"], s["slope_G_n_max"]["high"],
                     min(s["G_t_min"]), max(s["G_t_max"]), s["slope_G_t_min"]["high"], s["slope_G_t_max"]["high"], s["empty"]))
    json.dump(out, open("docs/figures/mac_momentum.json", "w"), indent=1, default=float)


def figure() -> None:
    """docs/figures/mac_momentum.png: the smallest normalised singular value of the momentum transform (worst
    position, all theta_x) and the normal Gram against the control-volume masses, for the dual and the phase-aware
    control volumes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"A": "#257de0", "B": "#f47230", "D0m": "#00b894"}
    sm = json.load(open("docs/figures/mac_momentum.json"))["scan_32"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=150)
    for s in sm:
        ls = "-o" if s["cv"] == "phase" else "--s"
        lab = f"{s['kind']}, {'phase-aware' if s['cv'] == 'phase' else 'dual'}"
        axes[0].loglog(RATIOS, s["sig2d_min"], ls, color=col[s["kind"]], lw=2, ms=4, label=lab)
        axes[1].loglog(RATIOS, s["sig2d_divfree_min"], ls, color=col[s["kind"]], lw=2, ms=4, label=lab)
        axes[2].loglog(RATIOS, s["G_n_max"], ls, color=col[s["kind"]], lw=2, ms=4, label=lab + " max")
        axes[2].loglog(RATIOS, s["G_n_min"], ls, color=col[s["kind"]], lw=1, ms=3, alpha=0.6)
    axes[0].set_title("momentum transform: min sigma(Q), full space", fontsize=9)
    axes[1].set_title("min sigma(Q), divergence-free subspace", fontsize=9)
    axes[2].set_title("normal Gram vs control-volume mass (max bold, min thin)", fontsize=9)
    for a in axes:
        a.set_xlabel("density ratio", fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
        a.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig("docs/figures/mac_momentum.png")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()

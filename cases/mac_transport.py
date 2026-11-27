"""docs/notes/110 (the user's priority 4-5, docs/notes/107 sections 5.5-5.6): transport with moving phase-aware
control volumes, 1-D, exact uniform translation of a layered interface.

A scalar velocity component is carried by the exact map y = x + Delta (density rho^{n+1}(y) = rho^n(y - Delta)).
Two components: "t" (tangential: cell data, reconstruction R_C) and "n" (normal: face data, R_F), with the low-order
reconstructions of cases.mac_layered (A, D0m, D0b). Control volumes chi (cases.mac_momentum: dual, phase, blend).
With pi the exact transport measure,

    m_ij = int rho^n(x) chi_i^n(x) chi_j^{n+1}(x + Delta) dx,    p_ij = the same with u^n(x) = R^n f^n,

so sum_j m_ij = m_i^n, sum_i m_ij = m_j^{n+1} (the control volumes partition the domain) and p^* = sum_i p_ij is the
exact momentum of the transported reconstruction in the new control volumes, including the transfer caused by a change
of ownership at the mobility switch. Two updates:

    Galerkin   G^{n+1} f = C f^n          (energy non-increasing, theorem 4; local momentum not imposed)
    Remap      P^{n+1} f = p^*            (local momentum exact; energy bounded only by 1 / sigma_min^2)

    python -m cases.mac_transport
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.mac_layered import _piece_values
from cases.mac_momentum import _assign
from cases.pccd import WG, XG
from cases.pccd_density import pieces


class Geometry:
    """Bases and control-volume weights at arbitrary points for the interface positions (a, b)."""

    def __init__(self, N, ratio, a, b, kind, cv):
        self.N, self.ratio, self.a, self.b, self.kind, self.cv = N, ratio, a % 1.0, b % 1.0, kind, cv
        self.h = 1.0 / N
        self.P = pieces(N, self.a, self.b)
        self.vals = _piece_values(N, self.P, ratio, kind)
        self.D = np.roll(np.eye(N), -1, axis=0) - np.eye(N)

    def heavy(self, Y):
        return ((Y - self.a) % 1.0) < ((self.b - self.a) % 1.0)

    def bases(self, Y):
        """(PhiC, PhiF) at the points Y (in [0, 1))."""
        N, h = self.N, self.h
        cell = np.floor(Y / h + 1e-13).astype(int) % N
        s = Y / h - np.floor(Y / h + 1e-13)
        PC, PF = np.zeros((Y.size, N)), np.zeros((Y.size, N))
        for i in np.unique(cell):
            k = np.where(cell == i)[0]
            cum = np.zeros(N)
            e = np.zeros(N)
            e[i] = 1.0
            for s0, s1, hv, row in self.vals[i]:
                if len(self.vals[i]) > 1:
                    sel = k[(s[k] >= s0 - 1e-13) & (s[k] <= s1 + 1e-13) & (self.heavy(Y[k]) == hv)]
                else:
                    sel = k
                PC[sel] = row
                PF[sel] = e[None, :] + (cum[None, :] + np.outer(s[sel] - s0, row)) @ self.D
                cum = cum + (s1 - s0) * row
        return PC, PF

    def weights(self, Y):
        R = {"X": Y, "P": self.P, "heavy": self.heavy(Y), "h": self.h}
        return _assign(self.N, R, self.ratio, self.cv)

    def cuts(self, i):
        """Breakpoints (in s) of cell i: phase pieces and the dual-cell boundary."""
        return {0.0, 0.5, 1.0, *[s1 for _, s1, _ in self.P[i]]}


def step_operators(g0: Geometry, g1: Geometry, delta):
    """All matrices of one step: old geometry g0, new geometry g1 = g0 shifted by delta."""
    N, h = g0.N, g0.h
    d = (delta / h) % 1.0
    Y, W = [], []
    for i in range(N):
        cuts = sorted(g1.cuts(i) | {d, (d + 0.5) % 1.0} |
                      {(c + d) % 1.0 for j in ((i - int(math.floor(delta / h)) - 1) % N, (i - int(math.floor(delta / h))) % N)
                       for c in g0.cuts(j)})
        cuts = [c for c in cuts if 0.0 <= c <= 1.0]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo < 1e-15:
                continue
            Y.append((i + lo + (hi - lo) * XG) * h), W.append((hi - lo) * h * WG)
    Y, W = np.concatenate(Y), np.concatenate(W)
    Xo = (Y - delta) % 1.0
    rho = np.where(g1.heavy(Y), 1.0, 1.0 / g0.ratio)
    wr = rho * W
    out = {}
    PCn, PFn = g1.bases(Y)
    PCo, PFo = g0.bases(Xo)
    w1, w0 = g1.weights(Y), g0.weights(Xo)
    for comp, Pn, Po, col in (("t", PCn, PCo, 1), ("n", PFn, PFo, 0)):
        G1 = Pn.T @ (wr[:, None] * Pn)
        G0 = Po.T @ (wr[:, None] * Po)
        C = Pn.T @ (wr[:, None] * Po)
        M1, M0 = np.zeros(N), np.zeros(N)
        P1, Pt = np.zeros((N, N)), np.zeros((N, N))
        for a1 in w1:
            idx, wt = a1[col], a1[2]
            M1 += np.bincount(idx, weights=wr * wt, minlength=N)
            np.add.at(P1, idx, (wr * wt)[:, None] * Pn)
            np.add.at(Pt, idx, (wr * wt)[:, None] * Po)
        mij = np.zeros((N, N))
        for a0 in w0:
            M0 += np.bincount(a0[col], weights=wr * a0[2], minlength=N)
            for a1 in w1:
                np.add.at(mij, (a0[col], a1[col]), wr * a0[2] * a1[2])
        out[comp] = {"G0": G0, "G1": G1, "C": C, "P1": P1, "Pt": Pt, "M0": M0, "M1": M1, "mij": mij}
    return out


def _amp(T, G1, G0):
    """Largest generalised eigenvalue of (T^T G1 T, G0): the worst one-step energy amplification."""
    L = np.linalg.cholesky(0.5 * (G0 + G0.T))
    Li = np.linalg.inv(L)
    return float(np.linalg.eigvalsh(Li @ (T.T @ G1 @ T) @ Li.T)[-1])


def run(N, ratio, kind, cv, a, b, deltas):
    """Translate the interface by the sequence deltas. Per step and per component: the mass balances of m_ij, the
    free stream, the worst energy amplification of both updates, the local-momentum defect of Galerkin, the largest
    change of a control-volume mass (transfer by re-assignment), and the smallest singular value of Q at the new
    step. Also the multi-step products applied to a smooth field."""
    one = np.ones(N)
    g0 = Geometry(N, ratio, a, b, kind, cv)
    x = np.arange(N) / N
    f = {m: {c: np.sin(2 * math.pi * x) + 0.3 for c in ("t", "n")} for m in ("gal", "rem")}
    E0 = None
    prod = {m: {c: np.eye(N) for c in ("t", "n")} for m in ("gal", "rem")}
    Gstart = {}
    rec = {c: {"mass_row": 0.0, "mass_col": 0.0, "free_gal": 0.0, "free_rem": 0.0, "amp_gal": 0.0, "amp_rem": 0.0,
               "defect_gal": 0.0, "mass_jump": 1.0, "sig_min": 1.0, "E_gal": [], "E_rem": [], "mom_gal": 0.0,
               "mom_rem": 0.0, "prod_gal": 0.0, "prod_rem": 0.0} for c in ("t", "n")}
    pos = 0.0
    for dl in deltas:
        g1 = Geometry(N, ratio, a + pos + dl, b + pos + dl, kind, cv)
        ops = step_operators(g0, g1, dl)
        for c in ("t", "n"):
            o, r = ops[c], rec[c]
            r["mass_row"] = max(r["mass_row"], float(np.abs(o["mij"].sum(1) - o["M0"]).max() / o["M0"].max()))
            r["mass_col"] = max(r["mass_col"], float(np.abs(o["mij"].sum(0) - o["M1"]).max() / o["M1"].max()))
            Tg = np.linalg.solve(o["G1"], o["C"])
            Tr = np.linalg.solve(o["P1"], o["Pt"])
            r["free_gal"] = max(r["free_gal"], float(np.abs(Tg @ one - 1).max()))
            r["free_rem"] = max(r["free_rem"], float(np.abs(Tr @ one - 1).max()))
            r["amp_gal"] = max(r["amp_gal"], _amp(Tg, o["G1"], o["G0"]))
            r["amp_rem"] = max(r["amp_rem"], _amp(Tr, o["G1"], o["G0"]))
            # local momentum defect of Galerkin, normalised: || M1^-1/2 (P1 Tg - Pt) G0^-1/2 ||
            L = np.linalg.cholesky(0.5 * (o["G0"] + o["G0"].T))
            Dm = (o["P1"] @ Tg - o["Pt"]) / np.sqrt(o["M1"])[:, None] @ np.linalg.inv(L).T
            r["defect_gal"] = max(r["defect_gal"], float(np.linalg.svd(Dm, compute_uv=False)[0]))
            ratio_m = o["M1"] / np.maximum(o["M0"], 1e-300)
            r["mass_jump"] = max(r["mass_jump"], float(max(ratio_m.max(), 1.0 / ratio_m.min())))
            L1 = np.linalg.cholesky(0.5 * (o["G1"] + o["G1"].T))
            sv = np.linalg.svd((o["P1"] / np.sqrt(o["M1"])[:, None]) @ np.linalg.inv(L1).T, compute_uv=False)
            r["sig_min"] = min(r["sig_min"], float(sv[-1]))
            if E0 is None or c not in E0:
                E0 = E0 or {}
                E0[c] = float(f["gal"][c] @ o["G0"] @ f["gal"][c])
                r["mom0"] = float(one @ o["G0"] @ f["gal"][c])
            Gstart.setdefault(c, o["G0"])
            for m, T in (("gal", Tg), ("rem", Tr)):
                prod[m][c] = T @ prod[m][c]
                r["prod_" + m] = max(r.get("prod_" + m, 0.0), _amp(prod[m][c], o["G1"], Gstart[c]))
                f[m][c] = T @ f[m][c]
                r["E_" + m].append(float(f[m][c] @ o["G1"] @ f[m][c]) / E0[c])
                r["mom_" + m] = max(r["mom_" + m], abs(float(one @ o["G1"] @ f[m][c]) - r["mom0"]) / abs(r["mom0"]))
        g0 = g1
        pos += dl
    for c in ("t", "n"):
        r = rec[c]
        r["E_gal_max"], r["E_rem_max"] = max(r["E_gal"]), max(r["E_rem"])
        r["E_gal_end"], r["E_rem_end"] = r["E_gal"][-1], r["E_rem"][-1]
        del r["E_gal"], r["E_rem"]
    return {"N": N, "ratio": ratio, "kind": kind, "cv": cv, "steps": len(deltas), **{c: rec[c] for c in ("t", "n")}}


def scenarios(N, ratio):
    """(name, a, b, deltas): uniform translation at two CFL numbers, a fine crossing of the mobility switch at the
    interface b (heavy fraction c of its cut cell passing c* = r / (r + 1)), and a back-and-forth across it."""
    h = 1.0 / N
    a0, b0 = (N // 4 + 0.37) * h, (3 * N // 4 + 0.21) * h
    cstar = ratio / (ratio + 1.0)
    eps = (1.0 - cstar) * h                                    # the light thickness at the switch
    bsw = (3 * N // 4 + cstar) * h
    out = [("translate CFL 0.37", a0, b0, [0.37 * h] * 60), ("translate CFL 0.1", a0, b0, [0.1 * h] * 100),
           ("cross the switch", a0, bsw - 4.5 * 0.2 * eps, [0.2 * eps] * 10),
           ("back and forth", a0, bsw - 0.3 * eps, [0.6 * eps, -0.6 * eps] * 20),
           ("translate 3 periods", a0, b0, [0.37 * h] * int(round(3 / (0.37 * h))))]
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    N = 32
    combos = (("A", "dual"), ("D0m", "dual"), ("D0m", "phase"), ("D0b", "phase"), ("D0b", "blend"))
    jobs = [(N, r, k, cv, a, b, dl, name) for r in (1.0, 1e3, 1e6) for (k, cv) in combos
            for (name, a, b, dl) in scenarios(N, r)]
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(_r, jobs))
    for j, o in zip(jobs, res):
        o["scenario"] = j[7]
        for c in ("t", "n"):
            r = o[c]
            print("%-4s %-5s r=%-7g %-19s %s: mass %.0e/%.0e free %.0e/%.0e | amp gal %.4f rem %.3g | prod gal %.4f "
                  "rem %.3g | defect %.2e | sig %.3g | m-jump %.3g | E(sin) gal %.4f rem %.4f (max %.4g) | mom %.0e/%.0e"
                  % (o["kind"], o["cv"], o["ratio"], o["scenario"], c, r["mass_row"], r["mass_col"], r["free_gal"],
                     r["free_rem"], r["amp_gal"], r["amp_rem"], r["prod_gal"], r["prod_rem"], r["defect_gal"],
                     r["sig_min"], r["mass_jump"], r["E_gal_end"], r["E_rem_end"], r["E_rem_max"], r["mom_gal"],
                     r["mom_rem"]), flush=True)
    json.dump(res, open("docs/figures/mac_transport.json", "w"), indent=1, default=float)


def _r(j):
    return run(*j[:7])


def figure() -> None:
    """docs/figures/mac_transport.png: worst multi-step energy amplification of the momentum Remap, and the
    local-momentum defect of Galerkin, per combination, against the density ratio (max over the scenarios)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open("docs/figures/mac_transport.json"))
    col = {("A", "dual"): "#257de0", ("D0m", "dual"): "#f47230", ("D0m", "phase"): "#00b894", ("D0b", "phase"): "#f2a600",
           ("D0b", "blend"): "#ee7fac"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), dpi=150)
    R = (1.0, 1e3, 1e6)
    for (k, cv), c in col.items():
        for comp, ls in (("t", "--s"), ("n", "-o")):
            sel = lambda r: [o[comp] for o in res if o["kind"] == k and o["cv"] == cv and o["ratio"] == r]  # noqa: E731
            lab = f"{k}, {cv}, {'tangential' if comp == 't' else 'normal'}"
            axes[0].loglog(R, [max(x["prod_rem"] for x in sel(r)) for r in R], ls, color=c, lw=1.8, ms=4, label=lab)
            axes[1].loglog(R, [max(max(x["defect_gal"], 1e-16) for x in sel(r)) for r in R], ls, color=c, lw=1.8, ms=4)
            axes[2].loglog(R, [max(x["E_rem_max"] for x in sel(r)) for r in R], ls, color=c, lw=1.8, ms=4)
    axes[0].set_title("momentum Remap: worst multi-step energy amplification", fontsize=9)
    axes[1].set_title("Galerkin: local-momentum defect (normalised)", fontsize=9)
    axes[2].set_title("momentum Remap: max energy of a smooth field / initial", fontsize=9)
    for a in axes:
        a.set_xlabel("density ratio", fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
    axes[0].legend(fontsize=5.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig("docs/figures/mac_transport.png")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()

"""docs/notes/85 (P0-12a, P0-12e): where does the max-norm amplification come from?

Four transports of the cell averages, compared at the same displacement s -> s + ds (note-82 reconstruction,
piecewise="ls", alpha 0.75, window +1):

* T_PC   = M1^-1 J1* U J0      piecewise-constant geometric remap (masses moved between cells; ||.||_inf = 1)
* T_geo  = M1^-1 J1* U R0      the high-order reconstruction integrated geometrically (the matrix Cg)
* T_G    = G1^-1 C             the Galerkin transport
* T_corr = T_G [I + (I - T^dagger T)/2]   the adjoint-corrected one (note 84)

The interface is placed by the phase-mass ratio zeta = r c / (1 - c) of a light cell holding a heavy sliver of
volume fraction c = zeta / (r + zeta) (the user's sampling): A0 (heavy on the right; with U > 0 the sliver leaves the
light cell) and B0 (heavy on the left; the sliver enters).

    python -m cases.galerkin_amp [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cases.galerkin_cls import corrected
from cases.galerkin_transport import A0, B0, _pieces, interfaces, matrices, rho_at


def pc_remap(n, s, ds, ratio, mass_new):
    """T_PC: (T)_ij = mass of rho^n over K_j that lands in K_i, / m_i^{n+1}."""
    h = 1.0 / n
    cuts = [z + m for z in interfaces(s) for m in (-1, 0, 1)]
    T = np.zeros((n, n))
    for j in range(n):
        for i_shift in (0, 1, 2):  # ds <= h: K_j moves into cells j and j + 1
            i = (j + i_shift) % n
            lo = max(j * h, (j + i_shift) * h - ds)
            hi = min((j + 1) * h, (j + i_shift + 1) * h - ds)
            if hi <= lo:
                continue
            xg, wg = np.polynomial.legendre.leggauss(4)
            m = 0.0
            for a, b in _pieces(lo, hi, cuts):
                x = 0.5 * (b - a) * xg + 0.5 * (a + b)
                m += 0.5 * (b - a) * float(np.sum(wg * rho_at(x, s, ratio)))
            T[i, j] += m
    return T / mass_new[:, None]


def place(n, ratio, zeta, which):
    """The displacement putting the interface so that a light cell holds a heavy sliver of mass ratio zeta."""
    h = 1.0 / n
    c = zeta / (ratio + zeta)
    if which == "A0":  # heavy right: light cell k - 1, interface at theta = 1 - c
        k = int(math.floor(A0 / h)) + 1
        return k * h - c * h - A0
    k = int(math.floor(B0 / h)) + 1  # heavy left: light cell k, interface at theta = c
    return k * h + c * h - B0


def four(n, ratio, zeta, which, cfl, alpha=0.75, extra=1, fit="constrained"):
    h = 1.0 / n
    s0 = place(n, ratio, zeta, which)
    ds = cfl * h
    G, C, old, new, Cg = matrices(n, 4, s0, ds, "ls", ratio=ratio, remap=True, alpha=alpha, extra=extra, fit=fit)
    G0 = matrices(n, 4, s0 - ds, ds, "ls", ratio=ratio, alpha=alpha, extra=extra, fit=fit)[0]
    TG = np.linalg.solve(G, C)
    ops = {"PC": pc_remap(n, s0, ds, ratio, new.mass), "geo": Cg / new.mass[:, None], "G": TG,
           "corr": corrected(TG, G0, C, 0.5)}
    Mh = np.diag(1 / np.sqrt(new.mass))
    ev = np.linalg.eigvalsh(Mh @ G @ Mh)
    out = {"cond": float(ev.max() / ev.min())}
    for k, T in ops.items():
        rows = np.abs(T).sum(1)
        i = int(np.argmax(rows))
        out[k] = {"inf": float(rows.max()), "row": i, "free_stream": float(np.abs(T @ np.ones(n) - 1).max())}
    return out


def _job(kw):
    return kw, four(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    jobs = []
    for ratio in (1e2, 1e3, 1e6):
        for which in ("A0", "B0"):
            for cfl in (0.05, 0.37):
                for e in np.arange(-3.0, 3.01, 0.5):
                    for fit in ("constrained", "shift"):
                        jobs.append({"n": 64, "ratio": ratio, "zeta": float(10 ** e), "which": which, "cfl": cfl,
                                     "fit": fit})
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for kw, r in ex.map(_job, jobs):
            res.append({"args": kw, "result": r})
    with open("docs/figures/galerkin_amp.json", "w") as f:
        json.dump(res, f, indent=1)
    for fit in ("constrained", "shift"):
        for ratio in (1e2, 1e3, 1e6):
            for which in ("A0", "B0"):
                for cfl in (0.05, 0.37):
                    rows = [r for r in res if r["args"]["ratio"] == ratio and r["args"]["which"] == which
                            and r["args"]["cfl"] == cfl and r["args"]["fit"] == fit]
                    print(f"-- {fit} ratio {ratio:g} {which} CFL {cfl}: max over zeta of ||T||_inf PC geo G corr, cond")
                    x = {k: max(r["result"][k]["inf"] for r in rows) for k in ("PC", "geo", "G", "corr")}
                    zarg = max(rows, key=lambda r: r["result"]["G"]["inf"])["args"]["zeta"]
                    print("   " + "  ".join(f"{k} {v:.3g}" for k, v in x.items())
                          + f"  (G peak at zeta {zarg:.3g}), cond max {max(r['result']['cond'] for r in rows):.3g}")


if __name__ == "__main__":
    main()

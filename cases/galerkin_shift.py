"""docs/notes/85 (P0-12b, P0-12d, P0-13 and the checks of fit="shift"): the note-82 reconstruction with its cell
constraint met by a constant shift of an unconstrained weighted fit ("shift") against the constrained fit.

* P0-12b: the worst one-step energy-loss mode u0 (smallest energy-norm singular value), its lost function
  r = U R0 u0 - R1 T u0, integrated cell by cell and phase by phase (sum = 1 - sigma_min^2 for |u0|_G0 = 1).
* P0-13: the oracle enrichment: add psi = f - R1 fbar (zero cell averages) of the k worst lost functions to the new
  space and recompute the worst loss of the projection (k = 0, 1, 2, 4).
* P0-12d: the largest eigenvalue of K v = lambda G v, K_ij = int mu phi_i' phi_j' with mu = rho (a uniform kinematic
  viscosity, so that the physical diffusion rate is the same in both phases; broken over the cells).

    python -m cases.galerkin_shift [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cases.galerkin_amp import place
from cases.galerkin_cls import p0_1, run10, sweep10
from cases.galerkin_transport import _pieces, interfaces, matrices, rho_at

KW = {"alpha": 0.75, "extra": 1}


def _quadrature(n, s, ds, ratio, order=10):
    """Points and rho^{n+1}-weights on the pieces of every new cell (cut at the cell boundaries, the old ones moved
    by ds, and the interfaces), with the cell and the phase of every point."""
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(order)
    cuts = ([k * h for k in range(-1, n + 2)] + [k * h + ds for k in range(-2, n + 2)]
            + [z + m for z in interfaces(s + ds) for m in (-1, 0, 1)])
    pts = []
    for i in range(n):
        for a, b in _pieces(i * h, (i + 1) * h, cuts):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, s + ds, ratio)
            heavy = bool(rho_at(np.array([0.5 * (a + b)]), s + ds, ratio)[0] == 1.0)
            pts.append((i, heavy, x, w))
    return pts


def _values(rec, x, n):
    idx, v = rec.basis(x)
    out = np.zeros((x.size, n))
    out[:, idx] = v
    return out


def lost_function(n, ratio, zeta, which, cfl, fit, modes=(0, 1, 2, 4)):
    h = 1.0 / n
    s = place(n, ratio, zeta, which)
    ds = cfl * h
    G, C, old, new = matrices(n, 4, s, ds, "ls", ratio=ratio, fit=fit, **KW)
    T = np.linalg.solve(G, C)
    pts = _quadrature(n, s, ds, ratio)
    F = np.vstack([np.sqrt(w)[:, None] * _values(old, x - ds, n) for _, _, x, w in pts])   # U R0
    P1 = np.vstack([np.sqrt(w)[:, None] * _values(new, x, n) for _, _, x, w in pts])       # R1
    cell = np.concatenate([np.full(x.size, i) for i, _, x, _ in pts])
    heavy = np.concatenate([np.full(x.size, hv) for _, hv, x, _ in pts])
    sw = np.concatenate([np.sqrt(w) for _, _, _, w in pts])
    # the worst mode: largest loss ||(I - P) F u||^2 / ||F u||^2
    QF, RF = np.linalg.qr(F)
    Q1, _ = np.linalg.qr(P1)
    Rres = QF - Q1 @ (Q1.T @ QF)
    U_, S_, Vt = np.linalg.svd(Rres, full_matrices=False)
    worst = float(S_[0] ** 2)
    y = Vt[0]
    u0 = np.linalg.solve(RF, y)  # F u0 = QF y, ||F u0|| = 1
    r = F @ u0 - P1 @ (T @ u0)
    # per cell and phase (r is already weighted by sqrt(rho w))
    per = {}
    for i in range(n):
        for hv in (False, True):
            m = (cell == i) & (heavy == hv)
            e = float(np.sum(r[m] ** 2))
            if e > 1e-3 * worst:
                per[f"{i}{'H' if hv else 'L'}"] = e
    dist = {k: min(min(abs((int(k[:-1]) + 0.5) * h - z), 1 - abs((int(k[:-1]) + 0.5) * h - z))
                   for z in interfaces(s + ds)) / h for k in per}
    # oracle enrichment
    oracle = {}
    for kmax in modes:
        basis = P1.copy()
        Rk = Rres
        for _ in range(kmax):
            Uk, Sk, Vk = np.linalg.svd(Rk, full_matrices=False)
            f = QF @ Vk[0]                       # the worst lost function (weighted)
            fw = f / np.where(sw > 0, sw, 1)     # unweighted values
            fbar = np.array([np.sum((sw ** 2)[cell == i] * fw[cell == i]) / np.sum((sw ** 2)[cell == i])
                             for i in range(n)])
            psi = f - P1 @ fbar                  # zero cell averages, weighted
            basis = np.hstack([basis, psi[:, None]])
            Qb, _ = np.linalg.qr(basis)
            Rk = QF - Qb @ (Qb.T @ QF)
        oracle[kmax] = float(np.linalg.svd(Rk, compute_uv=False)[0] ** 2)
    return {"worst_loss": worst, "check_sum": float(np.sum(r ** 2)), "per_cell_phase": per,
            "distance": dist, "oracle": oracle}


def stiffness(n, ratio, zeta, which, fit, c0=None):
    """lambda_max h^2 of K v = lambda G v (mu = rho: nu = 1 in both phases; the cell-wise derivative)."""
    h = 1.0 / n
    s = place(n, ratio, zeta, which)
    G = matrices(n, 4, s - 1e-9, 1e-9, "ls", ratio=ratio, fit=fit, c0=c0, **KW)[0]
    rec = matrices(n, 4, s, 1e-9, "ls", ratio=ratio, fit=fit, c0=c0, **KW)[2]
    K = np.zeros((n, n))
    xg, wg = np.polynomial.legendre.leggauss(8)
    cuts = [k * h for k in range(-1, n + 2)] + [z + m for z in interfaces(s) for m in (-1, 0, 1)]
    d = 1e-7 * h
    for i in range(n):
        for a, b in _pieces(i * h, (i + 1) * h, cuts):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, s, ratio)
            dv = (_values(rec, x + d, n) - _values(rec, x - d, n)) / (2 * d)
            K += (dv * w[:, None]).T @ dv
    ev = np.linalg.eigvals(np.linalg.solve(G, K)).real
    return {"lambda_max_h2": float(ev.max() * h * h)}


def _job(args):
    name, fn, kw = args
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    jobs = []
    for fit in ("constrained", "shift"):
        jobs.append(("P0-1", p0_1, {"alpha": 0.75, "ratio": 1e3, "fit": fit}))
        for n in (16, 32, 64, 128, 256):
            jobs.append(("P0-4", run10, {"n": n, "p": 4, "alpha": 0.75, "ratio": 1e3, "beta": 0.0, "fit": fit}))
        for r in (1e3, 1e6):
            jobs.append(("long", run10, {"n": 64, "p": 4, "alpha": 0.75, "ratio": r, "beta": 0.0, "t_end": 0.5,
                                         "samples": (0.25,), "transient": True, "fit": fit}))
        for beta in (0.0, 0.5):
            for r in (1.0, 1e3):
                for c in (0.001, 0.01, 0.37):
                    jobs.append(("loss", sweep10, {"alpha": 0.75, "ratio": r, "beta": beta, "cfl": c, "samples": 6,
                                                   "fit": fit}))
        for r, z, w in ((1.0, 1.0, "A0"), (1e3, 10.0, "A0"), (1e6, 100.0, "A0"), (1e6, 31.6, "B0")):
            for c in (0.01, 0.37):
                jobs.append(("lost", lost_function, {"n": 48, "ratio": r, "zeta": z, "which": w, "cfl": c,
                                                     "fit": fit}))
            for c0 in (None, "mean", "mass"):
                jobs.append(("stiff", stiffness, {"n": 48, "ratio": r, "zeta": z, "which": w, "fit": fit, "c0": c0}))
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, r in ex.map(_job, jobs):
            res.append({"test": name, "args": kw, "result": r})
            print(json.dumps({"test": name, "args": kw, "result": r}, default=str)[:400], flush=True)
    with open("docs/figures/galerkin_shift.json", "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()

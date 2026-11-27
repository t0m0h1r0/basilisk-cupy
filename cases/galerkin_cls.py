"""docs/notes/82: the constrained weighted least-squares reconstruction (piecewise="ls") of the Galerkin transport
(cases/galerkin_transport.py), measured in the user's order P0-1 .. P0-7, over the density ratio and the mass-weight
exponent alpha (W_jj = geo_j (m_j / m_ref)^alpha).

    python -m cases.galerkin_cls [--jobs N] [--only P0-1,P0-3,...]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cases.galerkin_transport import A0, Reconstruction, exact_averages, interfaces, matrices

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RATIOS = (1.0, 10.0, 100.0, 1e3, 1e6)


def _sqrtm(A):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1 / np.sqrt(w)) @ V.T


def step_stats(n, p, s, ds, alpha, ratio, mode="ls", extra=1):
    """One step s -> s + ds: T, ||T||_inf, ||T||_2, the energy-norm amplification and the smallest energy-norm
    singular value (worst-case energy loss 1 - smin^2), cond(M^-1/2 G M^-1/2) of the new G, sigma_min of the fits."""
    G, C, old, new = matrices(n, p, s, ds, mode, ratio=ratio, alpha=alpha, extra=extra)
    G0 = matrices(n, p, s - ds, ds, mode, ratio=ratio, alpha=alpha, extra=extra)[0]
    T = np.linalg.solve(G, C)
    G1h, _ = _sqrtm(G)
    _, G0ih = _sqrtm(G0)
    sv = np.linalg.svd(G1h @ T @ G0ih, compute_uv=False)
    Mh = np.diag(1 / np.sqrt(new.mass))
    ev = np.linalg.eigvalsh(Mh @ G @ Mh)
    return {"T": T, "inf": float(np.abs(T).sum(1).max()), "two": float(np.linalg.norm(T, 2)),
            "amp": float(sv[0]), "smin": float(sv[-1]), "cond": float(ev.max() / ev.min()),
            "sigma": float(min(old.sigma_min, new.sigma_min)), "G": G, "G0": G0, "old": old, "new": new}


def positions(n, samples):
    """Displacements putting the interface A0 at `samples` places across one cell (theta in [0, 1))."""
    h = 1.0 / n
    k = int(math.floor(A0 / h)) + 1
    return [k * h + t * h - A0 for t in np.linspace(0.0, 1.0, samples, endpoint=False)]


# -- P0-1: the jump across a cell boundary ------------------------------------------------------------------------
def p0_1(alpha, ratio, n=48, p=4, mode="ls", kind="kink", extra=1):
    h = 1.0 / n
    k = int(round(A0 / h)) + 1
    out = {}
    for across in (True, False):
        rows = []
        for eps in (1e-2, 1e-4, 1e-6, 1e-8):
            start = (k * h if across else (k + 0.5) * h) - eps * h
            s0, ds = start - A0, 2 * eps * h
            st = step_stats(n, p, s0, ds, alpha, ratio, mode, extra)
            u0 = exact_averages(n, s0, st["old"].mass, kind, ratio)
            u1 = st["T"] @ u0
            ex = exact_averages(n, s0 + ds, st["new"].mass, kind, ratio)
            dE = (u1 @ st["G"] @ u1) / (u0 @ st["G0"] @ u0) - 1
            rows.append({"eps": eps, "jump": float(np.linalg.norm(st["T"] - np.eye(n), 2)),
                         "err": float(np.abs(u1 - ex).max()), "dE": float(dE)})
        out["across" if across else "within"] = rows
    return out


# -- P0-2/P0-3/P0-5: sweeps over the interface position (and the CFL number) -------------------------------------
def sweep(alpha, ratio, cfl=0.37, n=48, p=4, samples=12, mode="ls", extra=1):
    """Max over interface positions of cond, ||T||_inf, ||T||_2, amp, worst energy loss, min sigma, and the
    one-step error of the 'phases' q."""
    h = 1.0 / n
    res = {"cond": 0.0, "inf": 0.0, "two": 0.0, "amp": 0.0, "loss": 0.0, "sigma": np.inf, "err": 0.0}
    for s0 in positions(n, samples):
        st = step_stats(n, p, s0, cfl * h, alpha, ratio, mode, extra)
        u0 = exact_averages(n, s0, st["old"].mass, "phases", ratio)
        u1 = st["T"] @ u0
        ex = exact_averages(n, s0 + cfl * h, st["new"].mass, "phases", ratio)
        res["cond"] = max(res["cond"], st["cond"])
        res["inf"] = max(res["inf"], st["inf"])
        res["two"] = max(res["two"], st["two"])
        res["amp"] = max(res["amp"], st["amp"])
        res["loss"] = max(res["loss"], 1.0 - st["smin"] ** 2)
        res["sigma"] = min(res["sigma"], st["sigma"])
        res["err"] = max(res["err"], float(np.abs(u1 - ex).max()))
    return res


def rank_scan(alpha, ratio, n=48, p=4, samples=400, extra=1):
    """P0-2: min over many interface positions of sigma_min(W^1/2 A Z) / sigma_max of the fits."""
    h = 1.0 / n
    k = int(math.floor(A0 / h)) + 1
    smin = np.inf
    for t in np.linspace(0.0, 1.0, samples, endpoint=False):
        r = Reconstruction(n, p, k * h + t * h - A0, "ls", ratio, alpha, extra)
        smin = min(smin, r.sigma_min)
    return smin


# -- P0-4/P0-6/P0-7: runs --------------------------------------------------------------------------------------
def run(n, p, alpha, ratio, kind="phases", t_end=0.137, cfl=0.37, samples=(0.05, 0.1), mode="ls", extra=1):
    """Max error at the sample times and t_end; energy E_h(t)/E_h(0) at t_end; max ||T||_inf and product norms."""
    steps = int(math.ceil(t_end / (cfl / n)))
    ds = t_end / steps
    at = {int(round(t / ds)): t for t in samples}
    errs, u, E0 = {}, None, None
    P = np.eye(n)
    inf1, infP = 0.0, 0.0
    for k in range(steps):
        G, C, old, new = matrices(n, p, k * ds, ds, mode, ratio=ratio, alpha=alpha, extra=extra)
        if u is None:
            u = exact_averages(n, 0.0, old.mass, kind, ratio)
            G0 = matrices(n, p, -ds, ds, mode, ratio=ratio, alpha=alpha, extra=extra)[0]
            E0 = float(u @ G0 @ u)
        T = np.linalg.solve(G, C)
        u = T @ u
        P = T @ P
        inf1 = max(inf1, float(np.abs(T).sum(1).max()))
        infP = max(infP, float(np.abs(P).sum(1).max()))
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(u - exact_averages(n, (k + 1) * ds, new.mass, kind, ratio)).max())
    errs[t_end] = float(np.abs(u - exact_averages(n, t_end, new.mass, kind, ratio)).max())
    return {"errors": errs, "energy": float(u @ G @ u) / E0, "steps": steps, "inf_step": inf1, "inf_product": infP}


def loss_scan(alpha, ratio, cfl, n=48, p=4, samples=6, extra=1):
    """P0-6: worst-case one-step energy loss 1 - smin^2 over interface positions, and per unit time (U = 1)."""
    h = 1.0 / n
    worst = 0.0
    for s0 in positions(n, samples):
        worst = max(worst, 1.0 - step_stats(n, p, s0, cfl * h, alpha, ratio, "ls", extra)["smin"] ** 2)
    return {"loss": worst, "per_time": worst / (cfl * h)}


def _job(args):
    name, fn, kw = args
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="docs/figures/galerkin_cls.json")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    jobs = []
    if only is None or "P0-1" in only:
        for al in (0.0, 0.75, 1.0):
            jobs.append(("P0-1", p0_1, {"alpha": al, "ratio": 1e3}))
        jobs.append(("P0-1", p0_1, {"alpha": 0.0, "ratio": 1e3, "mode": True}))
    if only is None or "P0-2" in only:
        for al in ALPHAS:
            for r in RATIOS:
                jobs.append(("P0-2", rank_scan, {"alpha": al, "ratio": r}))
    if only is None or "P0-3" in only:
        for al in ALPHAS:
            for r in RATIOS:
                jobs.append(("P0-3", sweep, {"alpha": al, "ratio": r}))
        for r in RATIOS:
            jobs.append(("P0-3", sweep, {"alpha": 0.0, "ratio": r, "mode": True}))
    if only is None or "P0-5" in only:
        for al in ALPHAS:
            for c in (0.05, 0.1, 0.2, 0.5, 0.75, 1.0):
                jobs.append(("P0-5", sweep, {"alpha": al, "ratio": 1e3, "cfl": c}))
    if only is None or "P0-4" in only:
        for al in (0.0, 0.75, 1.0):
            for kind in ("smooth", "kink", "phases"):
                for n in (16, 32, 64, 128, 256):
                    jobs.append(("P0-4", run, {"n": n, "p": 4, "alpha": al, "ratio": 1e3, "kind": kind}))
    if only is None or "P0-6" in only:
        for al in (0.0, 0.75, 1.0):
            for c in (0.37, 0.185, 0.0925, 0.04625):
                jobs.append(("P0-6", run, {"n": 64, "p": 4, "alpha": al, "ratio": 1e3, "kind": "phases",
                                           "cfl": c, "samples": ()}))
    if only is None or "P0-6" in only:
        for r in (1.0, 1e3):
            for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.37):
                jobs.append(("P0-6w", loss_scan, {"alpha": 0.75, "ratio": r, "cfl": c}))
    if only is None or "P0-3" in only:
        for al in (0.0, 0.75, 1.0):
            for r in RATIOS:
                jobs.append(("P0-3e0", sweep, {"alpha": al, "ratio": r, "extra": 0}))
    if only is None or "P0-7" in only:
        for al in (0.0, 0.75, 1.0):
            for n in (32, 64):
                jobs.append(("P0-7", run, {"n": n, "p": 4, "alpha": al, "ratio": 1e3, "kind": "phases",
                                           "t_end": 1.0, "samples": (0.25, 0.5, 0.75)}))
    results = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, res in ex.map(_job, [(j[0], j[1], j[2]) for j in jobs]):
            kw = {k: (v if not isinstance(v, bool) else str(v)) for k, v in kw.items()}
            results.append({"test": name, "args": kw, "result": res})
            print(json.dumps({"test": name, "args": kw, "result": res}), flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

"""docs/notes/86 (the user's review, P0/P1): what matters beyond the worst case.

* physical-mode loss: the largest one-step energy loss over the span Q of physically meaningful velocities (cell
  averages of smooth functions of wavelength >= 8 cells, global and with a gradient kink at the interfaces):
  loss_phys = 1 - lambda_min(B^T T^T G1 T B, B^T G0 B);
* the enrichment on both sides: psi = (I - R A) b with b an interface-following function translated with the flow
  (b^{n+1} = U b^n), added to the old and the new spaces; the worst loss of the extended transport
  (b: light-side tent, heavy-side tent, two-phase tent);
* the three transports in runs: the direct high-order remap T_geo = M1^-1 Cg, the Galerkin T_G = G1^-1 C and the
  adjoint-corrected one, with fit="shift".

    python -m cases.galerkin_phys [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cases.galerkin_amp import place
from cases.galerkin_cls import corrected
from cases.galerkin_shift import _quadrature, _values
from cases.galerkin_transport import A0, B0, _integrate, exact_averages, interfaces, matrices, rho_at

KW = {"alpha": 0.75, "extra": 1}


def _averages(n, s, ratio, mass, f):
    h = 1.0 / n
    cuts = [z + m for z in interfaces(s) for m in (-1, 0, 1)]
    return np.array([_integrate(lambda x: rho_at(x, s, ratio) * f(x), k * h, (k + 1) * h, cuts, 10)
                     for k in range(n)]) / mass


def physical_basis(n, s, ratio, mass):
    """Cell averages of sin/cos(2 pi k (x - s)) (k <= n/8) and of the same times the band's bump (z - A0)(B0 - z)
    inside the heavy band (continuous, gradient kink at the interfaces): columns."""
    cols = []
    for k in range(1, n // 8 + 1):
        for f in (np.sin, np.cos):
            cols.append(_averages(n, s, ratio, mass, lambda x, f=f, k=k: f(2 * np.pi * k * (x - s))))
            def kink(x, f=f, k=k):
                z = np.mod(x - s, 1.0)
                return np.where((z > A0) & (z < B0), (z - A0) * (B0 - z), 0.0) * f(2 * np.pi * k * z)
            cols.append(_averages(n, s, ratio, mass, kink))
    return np.array(cols).T


def physical_loss(n, ratio, zeta, which, cfl, fit, beta=0.0, tol=1e-4):
    """The largest one-step energy loss over the physical span B, orthonormalised in the G0 energy and truncated at
    the relative singular value ``tol`` (the span is nearly dependent: its combinations of tiny energy, large smooth
    functions cancelling to a grid-scale remainder, are not physical and are dropped)."""
    from cases.galerkin_cls import _sqrtm

    h = 1.0 / n
    s = place(n, ratio, zeta, which)
    ds = cfl * h
    G, C, old, new = matrices(n, 4, s, ds, "ls", ratio=ratio, fit=fit, **KW)
    G0 = matrices(n, 4, s - ds, ds, "ls", ratio=ratio, fit=fit, **KW)[0]
    T = corrected(np.linalg.solve(G, C), G0, C, beta)
    B = physical_basis(n, s, ratio, old.mass)
    G1h, _ = _sqrtm(G)
    G0h, _ = _sqrtm(G0)
    U, S, _ = np.linalg.svd(G0h @ B, full_matrices=False)
    k = int(np.sum(S > tol * S[0]))
    Tn = G1h @ T @ np.linalg.solve(G0h, U[:, :k])
    phys = float(1 - np.linalg.svd(Tn, compute_uv=False)[-1] ** 2)
    allv = float(1 - np.linalg.svd(G1h @ T @ np.linalg.inv(G0h), compute_uv=False)[-1] ** 2)
    return {"phys": phys, "phys_per_time": phys / (cfl * h), "all": allv, "kept": k, "of": int(B.shape[1])}


def _b(kind, x, zz, h, heavy_right):
    d = (x - zz) / h
    heavy = d > 0 if heavy_right else d < 0
    tent = np.maximum(0.0, 1.0 - np.abs(d))
    if kind == "light":
        return np.where(heavy, 0.0, tent)
    if kind == "heavy":
        return np.where(heavy, tent, 0.0)
    return tent  # "both": continuous across the interface


def enriched_loss(n, ratio, zeta, which, cfl, fit, kinds=("none", "light", "heavy", "both")):
    h = 1.0 / n
    s = place(n, ratio, zeta, which)
    ds = cfl * h
    G, C, old, new = matrices(n, 4, s, ds, "ls", ratio=ratio, fit=fit, **KW)
    pts = _quadrature(n, s, ds, ratio)
    sw = np.concatenate([np.sqrt(w) for _, _, _, w in pts])
    xs = np.concatenate([x for _, _, x, _ in pts])
    cell = np.concatenate([np.full(x.size, i) for i, _, x, _ in pts])
    F = np.vstack([np.sqrt(w)[:, None] * _values(old, x - ds, n) for _, _, x, w in pts])
    P1 = np.vstack([np.sqrt(w)[:, None] * _values(new, x, n) for _, _, x, w in pts])
    zi = 0 if which == "A0" else 1
    heavy_right = which == "A0"
    z0 = interfaces(s)[zi]
    z1 = interfaces(s + ds)[zi]
    out = {}
    for kind in kinds:
        if kind == "none":
            Fe, Pe = F, P1
        else:
            # psi = (I - R A) b, at the old time (evaluated at x - ds) and at the new time (at x)
            def psi(rec, xq, zz, ss):
                bv = _b(kind, xq, zz, h, heavy_right)
                # cell averages of b at time ss (the quadrature weights are rho^{n+1}(x) = rho^n(x - ds))
                ci = np.floor(xq / h + 1e-12).astype(int) % n  # the cell of xq at its own time
                bbar = np.array([np.sum((sw ** 2)[ci == i] * bv[ci == i]) / np.sum((sw ** 2)[ci == i])
                                 for i in range(n)])
                return bv - _values_all(rec, xq, n, pts, ss) @ bbar
            Fe = np.hstack([F, (sw * psi(old, xs - ds, z0, s))[:, None]])
            Pe = np.hstack([P1, (sw * psi(new, xs, z1, s + ds))[:, None]])
        QF, _ = np.linalg.qr(Fe)
        Q1, _ = np.linalg.qr(Pe)
        out[kind] = float(np.linalg.svd(QF - Q1 @ (Q1.T @ QF), compute_uv=False)[0] ** 2)
    return out


def _values_all(rec, xq, n, pts, ss):
    """Basis values at all quadrature points (piece by piece, the pieces lie in single cells of rec)."""
    rows, k = [], 0
    for _, _, x, _ in pts:
        m = x.size
        rows.append(_values(rec, xq[k:k + m], n))
        k += m
    return np.vstack(rows)


def run3(n, ratio, transport, t_end=0.137, cfl=0.37, kind="phases", fit="shift", samples=(0.05, 0.1)):
    """Errors, E(t)/E(0), max one-step ||T||_inf and the largest |u| over the run for the three transports."""
    steps = int(math.ceil(t_end / (cfl / n)))
    ds = t_end / steps
    at = {int(round(t / ds)): t for t in samples}
    u, E0, errs, inf, umax = None, None, {}, 0.0, 0.0
    for k in range(steps):
        G, C, old, new, Cg = matrices(n, 4, k * ds, ds, "ls", ratio=ratio, fit=fit, remap=True, **KW)
        G0 = matrices(n, 4, (k - 1) * ds, ds, "ls", ratio=ratio, fit=fit, **KW)[0]
        if u is None:
            u = exact_averages(n, 0.0, old.mass, kind, ratio)
            E0 = float(u @ G0 @ u)
            u0max = float(np.abs(u).max())
        if transport == "geo":
            T = Cg / new.mass[:, None]
        else:
            T = corrected(np.linalg.solve(G, C), G0, C, 0.5 if transport == "corr" else 0.0)
        u = T @ u
        inf = max(inf, float(np.abs(T).sum(1).max()))
        umax = max(umax, float(np.abs(u).max()))
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(u - exact_averages(n, (k + 1) * ds, new.mass, kind, ratio)).max())
    errs[t_end] = float(np.abs(u - exact_averages(n, t_end, new.mass, kind, ratio)).max())
    return {"errors": errs, "energy": float(u @ G @ u) / E0, "inf_step": inf, "umax_ratio": umax / u0max}


def _job(a):
    name, fn, kw = a
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    jobs = []
    cases = ((1.0, 1.0, "A0"), (1e3, 10.0, "A0"), (1e6, 100.0, "A0"), (1e6, 31.6, "B0"))
    for fit in ("constrained", "shift"):
        for r, z, w in cases:
            for c in (0.01, 0.37):
                for beta in (0.0, 0.5):
                    jobs.append(("phys", physical_loss, {"n": 48, "ratio": r, "zeta": z, "which": w, "cfl": c,
                                                         "fit": fit, "beta": beta}))
    for r, z, w in cases:
        for c in (0.01, 0.37):
            jobs.append(("enrich", enriched_loss, {"n": 48, "ratio": r, "zeta": z, "which": w, "cfl": c,
                                                   "fit": "shift"}))
    for tr in ("geo", "galerkin", "corr"):
        for n in (16, 32, 64, 128, 256):
            jobs.append(("order", run3, {"n": n, "ratio": 1e3, "transport": tr}))
        for r in (1e3, 1e6):
            jobs.append(("long", run3, {"n": 64, "ratio": r, "transport": tr, "t_end": 1.0,
                                        "samples": (0.25, 0.5)}))
            jobs.append(("grid", run3, {"n": 64, "ratio": r, "transport": tr, "kind": "grid", "samples": ()}))
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, r in ex.map(_job, jobs):
            res.append({"test": name, "args": kw, "result": r})
            print(name, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin_phys.json", "w"), indent=1)


if __name__ == "__main__":
    main()

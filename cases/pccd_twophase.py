"""docs/notes/95: the jump-separated PCCD in time: a two-phase band translated at constant velocity (1-D periodic),
direct remap of the mass-weighted averages.

Each step: the reconstruction u_h = R q_w + J from the mass-weighted averages (jumps estimated jointly, note 94 D, or
the exact ones); the new cell [x_i, x_{i+1}] holds the old field on [x_i - D, x_{i+1} - D], D = C h; its momentum is
the integral of rho u_h over the old-coordinate pieces (split at the old cell faces and the old interfaces), its mass
the integral of rho: ubar_i^{n+1} = momentum / mass.

    python -m cases.pccd_twophase
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, _mono
from cases.pccd_jump import A_, B_, J, exact, reconstruct


def averages(N, a, b, ratio, U):
    """Mass-weighted averages of the function U(x) with the heavy phase on (a, b) (periodic)."""
    h = 1.0 / N
    out = np.zeros(N)
    for i in range(N):
        num = den = 0.0
        for lo, hi, hv in _pieces(i * h, (i + 1) * h, a, b):
            r = 1.0 if hv else 1.0 / ratio
            x = lo + (hi - lo) * XG
            w = (hi - lo) * WG
            num += r * (w @ U(x))
            den += r * w.sum()
        out[i] = num / den
    return out


def _heavy(x, a, b):
    return ((x - a) % 1.0) < ((b - a) % 1.0)


def _pieces(lo, hi, a, b, extra=()):
    cuts = [lo, hi] + [c for c in extra if lo < c < hi]
    for z in (a, b):
        k = math.floor((lo - z)) + 1
        while z + k < hi:
            if lo < z + k < hi:
                cuts.append(z + k)
            k += 1
    cuts = sorted(set(cuts))
    return [(p, q, bool(_heavy(0.5 * (p + q), a, b))) for p, q in zip(cuts[:-1], cuts[1:]) if q - p > 0]


def evaluate(N, coef, terms, x):
    """u_h at x (any real x): the cell polynomial of floor(x/h) plus J."""
    h = 1.0 / N
    cell = np.floor(x / h).astype(int)
    s = x / h - cell
    return np.einsum("ip,ip->i", coef[cell % N], _mono(s)) + (J(x % 1.0, terms) if terms else 0.0)


def run(N, ratio, C=0.37, t_end=1.0, samples=(0.25, 0.5), method="joint"):
    h = 1.0 / N
    steps = int(math.ceil(t_end / (C * h)))
    dt = t_end / steps
    D = dt                                     # unit speed
    u0 = lambda x: exact(x % 1.0)  # noqa: E731
    ub = averages(N, A_, B_, ratio, u0)
    errs = {}
    at = {int(round(t / dt)): t for t in samples}
    mass0 = None
    for k in range(steps):
        a, b = (A_ + k * dt) % 1.0, (B_ + k * dt) % 1.0
        r = reconstruct(N, ratio, True, a=a, b=b, estimate=True, ubar_in=ub, method=method)
        coef, terms = r["coef"], r["terms"]
        new = np.zeros(N)
        for i in range(N):
            lo, hi = i * h - D, (i + 1) * h - D
            num = den = 0.0
            for p, q, hv in _pieces(lo, hi, a, b, extra=[i * h]):
                rr = 1.0 if hv else 1.0 / ratio
                x = p + (q - p) * XG
                w = (q - p) * WG
                num += rr * (w @ evaluate(N, coef, terms, x))
                den += rr * w.sum()
            new[i] = num / den
        if mass0 is None:
            mass0 = sum(ub[i] * _cellmass(i, N, a, b, ratio) for i in range(N))
        ub = new
        if k + 1 in at:
            t = (k + 1) * dt
            errs[at[k + 1]] = float(np.abs(ub - averages(N, (A_ + t) % 1.0, (B_ + t) % 1.0, ratio,
                                                         lambda x: u0(x - t))).max())
    t = steps * dt
    errs[t_end] = float(np.abs(ub - averages(N, (A_ + t) % 1.0, (B_ + t) % 1.0, ratio, lambda x: u0(x - t))).max())
    a, b = (A_ + t) % 1.0, (B_ + t) % 1.0
    mom = sum(ub[i] * _cellmass(i, N, a, b, ratio) for i in range(N))
    return {"errors": errs, "momentum_drift": float(abs(mom - mass0) / abs(mass0)), "steps": steps}


def _cellmass(i, N, a, b, ratio):
    h = 1.0 / N
    return sum((q - p) * (1.0 if hv else 1.0 / ratio) for p, q, hv in _pieces(i * h, (i + 1) * h, a, b))


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(N, ratio) for ratio in (1.0, 1e3, 1e6) for N in (32, 64, 128, 256)]
    res = []
    with ProcessPoolExecutor(12) as ex:
        futs = [(j, ex.submit(run, *j)) for j in jobs]
        for (N, ratio), f in futs:
            r = f.result()
            res.append({"N": N, "ratio": ratio, **r})
            print(N, ratio, r, flush=True)
    json.dump(res, open("docs/figures/pccd_twophase.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()


def basis(N, ratio, a, b, method="joint"):
    """[(coef_j, terms_j)] of the unit averages e_j (the reconstruction is linear in the data)."""
    out = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        r = reconstruct(N, ratio, True, a=a, b=b, estimate=True, ubar_in=e, method=method)
        out.append((r["coef"], r["terms"]))
    return out


def galerkin_run(N, ratio, C=0.37, t_end=1.0, samples=(0.25, 0.5), method="joint"):
    """The Galerkin transport of the same reconstruction: G1 u^{n+1} = C u^n on the common partition (new cells split
    at the moved old faces and at the interfaces), rho-weighted; energy cannot grow (note 87 D)."""
    h = 1.0 / N
    steps = int(math.ceil(t_end / (C * h)))
    dt = t_end / steps
    u0 = lambda x: exact(x % 1.0)  # noqa: E731
    ub = averages(N, A_, B_, ratio, u0)
    errs, rad = {}, 0.0
    at = {int(round(t / dt)): t for t in samples}
    old = basis(N, ratio, A_, B_, method)
    E0 = Elast = None
    mass = lambda a, b: np.array([_cellmass(i, N, a, b, ratio) for i in range(N)]) * h  # noqa: E731
    mom0 = float(mass(A_, B_) @ ub)
    for k in range(steps):
        a0, b0 = (A_ + k * dt) % 1.0, (B_ + k * dt) % 1.0
        a1, b1 = (A_ + (k + 1) * dt) % 1.0, (B_ + (k + 1) * dt) % 1.0
        new = basis(N, ratio, a1, b1, method)
        X, W = [], []
        for i in range(N):
            for p, q, hv in _pieces(i * h, (i + 1) * h, a1, b1, extra=[i * h + dt]):
                X.append(p + (q - p) * XG)
                W.append((q - p) * WG * (1.0 if hv else 1.0 / ratio))
        X, W = np.concatenate(X), np.concatenate(W)
        B1 = np.stack([evaluate(N, c, t, X) for c, t in new], axis=1)
        B0 = np.stack([evaluate(N, c, t, X - dt) for c, t in old], axis=1)
        G1 = B1.T @ (W[:, None] * B1)
        Cm = B1.T @ (W[:, None] * B0)
        if E0 is None:
            G0 = B0.T @ (W[:, None] * B0)
            E0 = float(ub @ G0 @ ub)
        T = np.linalg.solve(G1, Cm)
        rad = max(rad, float(max(abs(np.linalg.eigvals(T)))))
        ub = T @ ub
        Elast = float(ub @ G1 @ ub)
        old = new
        if k + 1 in at:
            t = (k + 1) * dt
            errs[at[k + 1]] = float(np.abs(ub - averages(N, a1, b1, ratio, lambda x: u0(x - t))).max())
    t = steps * dt
    a, b = (A_ + t) % 1.0, (B_ + t) % 1.0
    errs[t_end] = float(np.abs(ub - averages(N, a, b, ratio, lambda x: u0(x - t))).max())
    return {"errors": errs, "energy": Elast / E0 - 1.0, "spectral_radius_max": rad,
            "momentum_drift": float(abs(mass(a, b) @ ub - mom0) / abs(mom0)), "steps": steps}

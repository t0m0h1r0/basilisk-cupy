"""docs/notes/92: two nearby interfaces (a thin band). Note 88: the reconstruction jumps where the nearest interface
changes, in the middle of a band thinner than twice the piecewise range. Two remedies:
* "pair": the fits about both interfaces, weighted by the distance difference (1/2 each at equal distances);
* "two": one fit with both kinks, P0 + max(d1, 0) P1 + max(d2, 0) P2, its window centred at the cell.

    python -m cases.galerkin2d_thin
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from cases.galerkin2d import P0_EXP, P1_EXP, Geometry, Reconstruction2D, recon_jump, run

RK = {"window": "interface", "switch": "angle", "diff": True, "R": 5.5}
TWO = {**RK, "two": (5.5, 6.5)}
PAIR = {**RK, "pair": 1.0}
CONFIGS = (("single", RK), ("pair", PAIR), ("two", TWO))


def exactness_two(N, a=1, b=1, zA=0.213, zB=0.647, seed=0, rk=None, d=0.0, near=4.6):
    """P0 + max(dA, 0) P1 + max(dB, 0) P2 (both band lines, distances in cells, polynomials about the target cell)
    against the reconstruction in every cell within ``near`` cells of an interface (relative error)."""
    rk = rk or TWO
    g = Geometry(a, b, zA, zB, d=d)
    rec = Reconstruction2D(N, g, **rk)
    h = 1.0 / N
    rng = np.random.default_rng(seed)
    c0, c1, c2 = rng.normal(size=15), rng.normal(size=10), rng.normal(size=10)
    err = scale = 0.0
    for iy in range(N):
        for ix in range(N):
            c = np.array([(ix + 0.5) * h, (iy + 0.5) * h])
            s, _ = g.nearest(*c)
            if abs(s) / h > near:
                continue
            Lc = g.L(*c) - g.d
            lA = zA + g.d + round(Lc - zA)          # the copies of the two lines nearest the cell
            lB = zB + g.d + round(Lc - zB)

            def q(X):
                xi, eta = (X[:, 0] - c[0]) / h, (X[:, 1] - c[1]) / h
                L = g.L(X[:, 0], X[:, 1])
                dA = (L - lA) / g.g / h
                dB = (lB - L) / g.g / h
                p0 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c0, P0_EXP)) / 10.0 ** 4
                p1 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c1, P1_EXP)) / 10.0 ** 3
                p2 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c2, P1_EXP)) / 10.0 ** 3
                return p0 + np.maximum(dA, 0.0) * p1 + np.maximum(dB, 0.0) * p2
            i = iy * N + ix
            idx, V = rec.basis(i, rec.qx[i])
            u = {}
            for dy in range(-9, 10):
                for dx in range(-9, 10):
                    Xc, W, j = rec._cell_points(ix + dx, iy + dy)
                    u[j] = float(W @ q(Xc)) / W.sum()
            ref = q(rec.qx[i])
            err = max(err, float(np.abs(V @ np.array([u[j] for j in idx]) - ref).max()))
            scale = max(scale, float(np.abs(ref).max()))
    return err / scale


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for N in (24, 32):
        for name, rk in CONFIGS:
            jobs.append(("exactness_two", exactness_two, {"N": N, "rk": rk}, name))
            jobs.append(("jump", recon_jump, {"N": N, "rk": rk, "spacing": 4e-3}, name))
            jobs.append(("jump", recon_jump, {"N": N, "rk": rk, "spacing": 2e-3}, name))
    for N in (16, 24, 32, 48):
        for name, rk in CONFIGS:
            jobs.append(("run", run, {"N": N, "rk": rk, "samples": (0.05,)}, name))
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        futs = [(n, kw, name, ex.submit(fn, **kw)) for n, fn, kw, name in jobs]
        for n, kw, name, f in futs:
            r = f.result()
            kw = {k: v for k, v in kw.items() if k != "rk"}
            res.append({"test": n, "config": name, "args": kw, "result": r})
            print(f"{time.time() - t0:7.0f}s", n, name, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin2d_thin.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

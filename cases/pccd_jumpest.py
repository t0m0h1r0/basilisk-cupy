"""docs/notes/94 (P1-3): jumps estimated from the data, robustness. ``method``: "onesided" (quartics on the 5 nearest
pure cells of each side) or "joint" (both sides at once from every cell within 5.5 cells, continuous weights).

* ``amplification``: Lambda = max_x sum_j |d u_h(x) / d ubar_j| over the cells within 3 of the interface a, the heavy
  fraction of a's cell prescribed by zeta = r c / (1 - c) (note 90's scan);
* ``continuity``: the largest change of u_h (fixed random averages) between interface positions ``spacing`` cells
  apart while a crosses a cell face: a jump does not shrink with the spacing.

    python -m cases.pccd_jumpest
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd_jump import B_, reconstruct


def amplification(N, ratio, zeta, b=B_, method="onesided"):
    h = 1.0 / N
    c = zeta / (ratio + zeta)
    m = N // 4
    a = (m + 1 - c) * h                       # light below a: the heavy part of cell m is its right end, fraction c
    cols = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        cols.append(reconstruct(N, ratio, True, a=a, b=b, estimate=True, ubar_in=e, method=method)["values"])
    V = np.array(cols)                         # (N data, N cells, points)
    near = [(m + o) % N for o in range(-3, 4)]
    return float(np.abs(V[:, near, :]).sum(0).max())


def continuity(N, ratio, spacing, seed=0, b=B_, span=0.2, method="onesided"):
    """Interface a from (m + 1 - span/2) h to (m + 1 + span/2) h (crossing the face of cell m)."""
    h = 1.0 / N
    m = N // 4
    u = np.random.default_rng(seed).normal(size=N)
    prev, worst, where = None, 0.0, None
    for t in np.arange(-span / 2, span / 2 + 1e-12, spacing):
        a = (m + 1 + t) * h
        v = reconstruct(N, ratio, True, a=a, b=b, estimate=True, ubar_in=u, method=method)["values"]
        if prev is not None:
            d = float(np.abs(v - prev).max())
            if d > worst:
                worst, where = d, float(t)
        prev = v
    return {"max_change": worst, "at": where}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for method in ("onesided", "joint"):
        for ratio in (1e3, 1e6):
            for z in (1e-3, 0.1, 1, 10, 100, 1e3, 1e4):
                jobs.append(("amp", amplification, {"N": 64, "ratio": ratio, "zeta": z, "method": method}))
        for ratio in (1.0, 1e3, 1e6):
            for sp in (2e-2, 1e-2, 5e-3):
                jobs.append(("cont", continuity, {"N": 64, "ratio": ratio, "spacing": sp, "method": method}))
    res = []
    with ProcessPoolExecutor(12) as ex:
        futs = [(n, kw, ex.submit(fn, **kw)) for n, fn, kw in jobs]
        for n, kw, f in futs:
            r = f.result()
            res.append({"test": n, "args": kw, "result": r})
            print(n, kw, r, flush=True)
    json.dump(res, open("docs/figures/pccd_jumpest.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

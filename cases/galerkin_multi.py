"""docs/notes/86 (P0, the user's review): two interfaces inside one reconstruction window. The heavy band is
narrowed to w cells (B0 = A0 + w/N) so that the window of the note-82 reconstruction (12 cells for degree 4, window
+1) holds both interfaces; fit="shift", alpha 0.75.

* exactness of per-phase polynomials;
* continuity: the largest relative change of G between interface positions 2e-3 cells apart, over two cells
  (a jump of the reconstruction shows as a spike that does not shrink with the spacing);
* one-step error of the 'phases' q and ||T||_inf over 12 positions.

    python -m cases.galerkin_multi [--jobs N]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cases.galerkin_transport as gt

N, KW = 48, {"alpha": 0.75, "extra": 1, "fit": "shift"}


def _setup(w):
    gt.B0 = gt.A0 + w / N


def exact(w, ratio):
    _setup(w)
    R0 = gt.Reconstruction
    gt.Reconstruction = lambda *a, **k: R0(*a, **{**k, "fit": "shift"})
    try:
        return max(gt.exactness(N, 4, ratio, piecewise="ls", alpha=0.75, s=s, extra=1)[0]
                   for s in np.linspace(0.0, 1.0 / N, 5))
    finally:
        gt.Reconstruction = R0


def continuity(w, ratio, spacing):
    _setup(w)
    h = 1.0 / N
    worst, where = 0.0, None
    prev = None
    for s in np.arange(0.0, 2 * h, spacing * h):
        G = gt.matrices(N, 4, s - 1e-9, 1e-9, "ls", ratio=ratio, **KW)[0]
        if prev is not None:
            d = float(np.linalg.norm(G - prev, 2) / np.linalg.norm(G, 2))
            if d > worst:
                worst, where = d, float(s / h)
        prev = G
    return {"max_rel_change": worst, "at_cells": where}


def step(w, ratio):
    _setup(w)
    h = 1.0 / N
    err, inf = 0.0, 0.0
    for t in np.linspace(0.0, 1.0, 12, endpoint=False):
        s = t * h
        G, C, old, new = gt.matrices(N, 4, s, 0.37 * h, "ls", ratio=ratio, **KW)
        T = np.linalg.solve(G, C)
        u0 = gt.exact_averages(N, s, old.mass, "phases", ratio)
        err = max(err, float(np.abs(T @ u0 - gt.exact_averages(N, s + 0.37 * h, new.mass, "phases", ratio)).max()))
        inf = max(inf, float(np.abs(T).sum(1).max()))
    return {"err": err, "inf": inf}


def _job(a):
    name, fn, kw = a
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    jobs = []
    for w in (20, 10, 7, 5, 3):
        for r in (1e3, 1e6):
            jobs.append(("exact", exact, {"w": w, "ratio": r}))
            jobs.append(("step", step, {"w": w, "ratio": r}))
            for sp in (4e-3, 1e-3):
                jobs.append(("continuity", continuity, {"w": w, "ratio": r, "spacing": sp}))
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, r in ex.map(_job, jobs):
            res.append({"test": name, "args": kw, "result": r})
            print(name, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin_multi.json", "w"), indent=1)


if __name__ == "__main__":
    main()

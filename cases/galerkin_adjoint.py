"""docs/notes/84 (P0-10a, P0-10b): the adjoint-corrected Galerkin transport T_beta = T [I + beta (I - T^dagger T)]
on the reconstruction of note 82 (piecewise="ls", alpha 0.75, window +1), over the density ratio.

    python -m cases.galerkin_adjoint [--jobs N]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from cases.galerkin_cls import run10, sweep10

RATIOS = (1.0, 10.0, 100.0, 1e3, 1e6)


def _job(args):
    name, fn, kw = args
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--out", default="docs/figures/galerkin_adjoint.json")
    a = ap.parse_args()
    jobs = []
    for beta in (0.0, 0.25, 0.5):
        for r in RATIOS:
            jobs.append(("structure", sweep10, {"alpha": 0.75, "ratio": r, "beta": beta}))
        for c in (0.001, 0.01, 0.1):
            for r in (1.0, 1e3):
                jobs.append(("loss_cfl", sweep10, {"alpha": 0.75, "ratio": r, "beta": beta, "cfl": c, "samples": 6}))
        for n in (16, 32, 64, 128, 256):
            jobs.append(("order", run10, {"n": n, "p": 4, "alpha": 0.75, "ratio": 1e3, "beta": beta}))
        for r in (1e3, 1e6):
            jobs.append(("long", run10, {"n": 64, "p": 4, "alpha": 0.75, "ratio": r, "beta": beta, "t_end": 0.5,
                                         "samples": (0.25,), "transient": True}))
        for c in (0.37, 0.185, 0.0925):
            jobs.append(("energy", run10, {"n": 64, "p": 4, "alpha": 0.75, "ratio": 1e3, "beta": beta,
                                           "kind": "grid", "cfl": c, "samples": ()}))
    results = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, res in ex.map(_job, jobs):
            results.append({"test": name, "args": kw, "result": res})
            print(json.dumps({"test": name, "args": kw, "result": res}), flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

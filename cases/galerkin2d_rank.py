"""docs/notes/88 (P1-1d): rank-safe 2-D reconstruction. Compares the note-87 reconstruction ("cell": 7 x 7 window
around the cell, switch at 3.5-4.5 cells) with the interface-centred window ("interface": the cells within 4.5 cells
of the cell's foot on the interface, compact weights, angle-dependent switch at d_safe = 2.5 (|nx| + |ny|) cells, the
shift on velocity differences), and the same with the continuous SVD filter (tau = 1e-3).

    python -m cases.galerkin2d_rank [--jobs N]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from cases.galerkin2d import exactness, free_stream, one_step, recon_jump

CONFIGS = {"cell": {}, "interface": {"window": "interface", "switch": "angle", "diff": True},
           "interface+tau": {"window": "interface", "switch": "angle", "diff": True, "tau": 1e-3},
           "interface R5.5": {"window": "interface", "switch": "angle", "diff": True, "R": 5.5}}
RUN = ("interface+tau", "interface R5.5")
ANGLES = {"0deg": (0, 1), "26.6deg": (1, 2), "45deg": (1, 1)}


def _job(a):
    name, fn, kw = a
    return name, kw, fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    jobs = []
    for cn in RUN:
        rk = CONFIGS[cn]
        for an, (aa, bb) in ANGLES.items():
            jobs.append(("exact", exactness, {"N": 24, "a": aa, "b": bb, **rk}))
            jobs.append(("free_stream", free_stream, {"N": 24, "a": aa, "b": bb, "rk": rk}))
            for N in (16, 24, 32, 48, 64):
                jobs.append(("one_step", one_step, {"N": N, "a": aa, "b": bb, "rk": rk}))
        for an, (aa, bb) in (("0deg", (0, 1)), ("45deg", (1, 1))):
            for ratio in (1.0, 1e3):
                for sp in (4e-3, 2e-3):
                    jobs.append(("recon_jump", recon_jump, {"N": 24, "a": aa, "b": bb, "ratio": ratio, "spacing": sp,
                                                            "rk": rk}))
    for cn in ("cell", "interface"):
        for ratio in (1.0,):
            for sp in (4e-3, 2e-3):
                jobs.append(("recon_jump", recon_jump, {"N": 24, "a": 0, "b": 1, "ratio": ratio, "spacing": sp,
                                                        "rk": CONFIGS[cn]}))
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, kw, r in ex.map(_job, sorted(jobs, key=lambda j: -j[2].get("N", 0))):
            res.append({"test": name, "args": {k: v for k, v in kw.items()}, "result": r})
            print(name, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin2d_rank2.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

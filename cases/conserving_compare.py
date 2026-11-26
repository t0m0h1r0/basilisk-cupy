"""Momentum-conserving VOF (Basilisk ``navier-stokes/conserving.h``, ``TwoPhase(conserving=True)``) against Basilisk's
C build on its Rayleigh--Taylor test (``src/test/rt.c`` on a uniform 64^2 grid without adaptivity,
``tools/basilisk/rt_conserving.c``): the fields at t = 0.1 and 0.4 and the step sizes. The port relaxes
lexicographically, as Basilisk's CPU build.

    qcc -O2 -Wall rt_conserving.c -o rt_conserving -lm && ./rt_conserving 2> log   (in a scratch directory)
    python -m cases.conserving_compare --dir <that directory>
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.linear.lexicographic import set_relaxation
from basilisk_cupy.time.basilisk_centered import CenteredNS

TIMES = (0.1, 0.4)


def run(conserving: bool = True, n: int = 64):
    grid = CartesianGrid((n, n), length=(4.0, 4.0), origin=(-2.0, -2.0))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-6, DT=5e-3)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.05 * np.cos(2.0 * math.pi * x) + y),
                                BoundaryConditions())
    TwoPhase(ns, f, rho1=1.225, rho2=0.1694, mu1=0.00313, mu2=0.00313, conserving=conserving)
    ns.a[1] = -9.81 + 0.0 * ns.a[1]
    ns.start(events=list(TIMES))
    out, dts = {}, []
    while ns.clock.t < TIMES[-1] - 1e-12:
        ns.step()
        dts.append((ns.clock.i, float(ns.clock.t), float(ns.clock.dt)))
        for t in TIMES:
            if abs(ns.clock.t - t) < 1e-9:
                out[t] = {"f": to_numpy(f.interior).copy(), "ux": to_numpy(ns.u[0].interior).copy(),
                          "uy": to_numpy(ns.u[1].interior).copy(), "p": to_numpy(ns.p.interior).copy()}
    return out, dts


def load(path: str, n: int = 64):
    a = np.loadtxt(path)
    order = np.lexsort((a[:, 1], a[:, 0]))  # x major, y minor: [i, j]
    a = a[order]
    return {k: a[:, c].reshape(n, n) for k, c in (("f", 2), ("ux", 3), ("uy", 4), ("p", 5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    set_relaxation("lexicographic")
    out, dts = run(True)
    plain, _ = run(False)
    log = np.loadtxt(os.path.join(args.dir, "log"))
    print(f"steps: port {dts[-1][0]}, Basilisk {int(log[-1, 0])}; "
          f"max |dt - dt_C| over the common steps "
          f"{max(abs(d[2] - log[k + 1, 2]) for k, d in enumerate(dts[:len(log) - 1])):.2e}")  # the C log: i, t, last dt
    for t in TIMES:
        ref = load(os.path.join(args.dir, f"fields-{t:g}.txt"))
        for name, fields in (("conserving", out[t]), ("plain (no conserving.h)", plain[t])):
            diffs = {k: float(np.abs(fields[k] - ref[k]).max()) for k in ("f", "ux", "uy")}
            pd = fields["p"] - ref["p"]
            diffs["p (mean removed)"] = float(np.abs(pd - pd.mean()).max())
            print(f"t = {t}: {name:24s}", "  ".join(f"{k} {v:.2e}" for k, v in diffs.items()),
                  f"(|u|max {np.abs(ref['uy']).max():.3f})", flush=True)


if __name__ == "__main__":
    main()

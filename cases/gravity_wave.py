"""Gravity wave (port of Basilisk ``src/test/gravity.c``): two-phase VOF, reduced gravity.

Domain 2 x 2 (``y`` in [-1, 1]), free-slip walls, ``rho1 = 1`` (``f = 1``, above), ``rho2 = 0.1``,
``mu1 = mu2 = 0.0182571749236``, ``G.y = 50``, initial interface ``y = 0.01 cos(2 pi x)``.
The amplitude (largest height-function position of the interface) is compared with
Prosperetti's solution every 0.00225584983639310905 up to t = 1.66481717925811447992; the
relative RMS error against ``gravity.ref``.

    python -m cases.gravity_wave [--nmax 128] [--backend numpy|cupy]
"""

from __future__ import annotations

import argparse
import math

from basilisk_cupy.constants import PI
from pathlib import Path

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA, position
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

STEP = 0.00225584983639310905
T_END = 1.66481717925811447992
#: gravity.ref: N / L0 -> relative RMS error
REFERENCE = {8: 0.126713, 16: 0.031758, 32: 0.00807759, 64: 0.00455815}
PROSPERETTI = [tuple(map(float, line.split())) for line in
               (Path(__file__).with_name("prosperetti_gravity.txt").read_text().splitlines()[1:])]


def event_times():
    times, t = [], 0.0
    while t <= T_END:
        times.append(t)
        t += STEP
    return times


def run(n: int, backend: str = "numpy", log=None, tolerance: float = 1e-6, vof_first: bool = True, measure_after_vof: bool = True):
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
    xp = grid.xp
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=tolerance)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.01 * xp.cos(2.0 * PI * x)),
                                BoundaryConditions())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=0.1, mu1=0.0182571749236, mu2=0.0182571749236, vof_first=vof_first)
    tp.add_reduced_gravity(G=(0.0, 50.0))
    times = event_times()
    amps = []

    def amplitude(_ns):
        if len(amps) < len(times) and abs(_ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, _ns.clock.t):
            pos = position(f.interior, f.bc, grid, (0.0, 1.0))
            amps.append(float(xp.max(xp.where(pos == NODATA, -1e30, pos))))
            if log:
                log(f"{_ns.clock.t * 16.032448313657:g} {amps[-1]:g}")

    # Basilisk runs the user's amplitude event right after the VOF advection moved "first"
    # (before the acceleration event clamps f): measure there.
    ns.on("set_dtmax", amplitude, after=measure_after_vof)
    ns.start(events=times[1:])
    while len(amps) < len(times):
        ns.step()
    m = min(len(amps), len(PROSPERETTI))
    err = math.sqrt(sum((amps[k] - PROSPERETTI[k][1]) ** 2 for k in range(m)) / m) / 0.01
    return {"n_per_wavelength": n / 2.0, "error": err, "amplitudes": amps, "steps": ns.clock.i}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=128)
    ap.add_argument("--backend", default="numpy")
    args = ap.parse_args()
    n = 16
    while n <= args.nmax:
        r = run(n, args.backend)
        print(f"{r['n_per_wavelength']:g} {r['error']:g}   ref {REFERENCE.get(int(r['n_per_wavelength']))}  steps {r['steps']}",
              flush=True)
        n *= 2


if __name__ == "__main__":
    main()

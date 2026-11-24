"""Capillary wave (port of Basilisk ``src/test/capwave.c``): VOF interface with surface tension.

Domain 2 x 2 (``y`` in [-1, 1]), free-slip walls, uniform density (``vof.h`` + ``tension.h``,
no ``two-phase.h``), ``mu = 0.0182571749236``, ``sigma = 1``, initial interface
``y = 0.01 cos(2 pi x)``. The amplitude (largest height-function position of the interface)
is compared with Prosperetti's solution every 3.04290519077e-3 up to t = 2.2426211256; the
relative RMS error against ``capwave.ref``.

    python -m cases.capillary_wave [--nmax 128] [--backend numpy|cupy] [--lexicographic]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.constants import PI
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA, position
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

STEP = 3.04290519077e-3
T_END = 2.2426211256
MU = 0.0182571749236
#: capwave.ref: N / L0 -> relative RMS error
REFERENCE = {8: 0.13865, 16: 0.0242375, 32: 0.00639279, 64: 0.00226914}
PROSPERETTI = [tuple(map(float, line.split())) for line in
               Path(__file__).with_name("prosperetti_capillary.txt").read_text().splitlines()[1:]]


def event_times():
    times, t = [], 0.0
    while t <= T_END:
        times.append(t)
        t += STEP
    return times


def _cls_amplitude(xp, psi, grid):
    """Largest ``y`` of the ``psi = 1/2`` crossings along the columns (linear interpolation)."""
    _, y = grid.meshgrid()
    a, b = psi[:, :-1] - 0.5, psi[:, 1:] - 0.5
    cross = (a * b <= 0) & (a != b)
    t = xp.where(cross, a / xp.where(cross, a - b, 1.0), 0.0)
    yc = y[:, :-1] + t * float(grid.h[1])
    return float(xp.max(xp.where(cross, yc, -1e30)))


def run(n: int, backend: str = "numpy", tolerance: float = 1e-6, interface: str = "vof", **cls_kw):
    """``interface="cls"``: TwoPhaseFlow's CLS (:class:`TwoPhaseCLS`, keywords ``cls_kw``), the
    amplitude from the ``psi = 1/2`` crossings."""
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
    xp = grid.xp
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=tolerance)
    if interface == "cls":
        x, y = grid.meshgrid()
        k = 2.0 * PI
        # signed distance to y = 0.01 cos(k x) to O(a^2 k^2): (y - eta) / sqrt(1 + eta'^2)
        dist = (y - 0.01 * xp.cos(k * x)) / xp.sqrt(1.0 + (0.01 * k * xp.sin(k * x)) ** 2)
        cls = CLSInterface.from_level_set(grid, dist, BoundaryConditions())
        f = cls.psi
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=MU, mu2=MU, vof_first=True, **cls_kw)
    else:
        f = CellField.from_interior(grid, fraction(grid, lambda x, y: y - 0.01 * xp.cos(2.0 * PI * x)),
                                    BoundaryConditions())
        tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0, mu1=MU, mu2=MU, vof_first=True)
    tp.add_surface_tension(1.0)
    times = event_times()
    amps = []

    def amplitude(_ns):
        if len(amps) < len(times) and abs(_ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, _ns.clock.t):
            if interface == "cls":
                amps.append(_cls_amplitude(xp, f.interior, grid))
            else:
                pos = position(f.interior, f.bc, grid, (0.0, 1.0))
                amps.append(float(xp.max(xp.where(pos == NODATA, -1e30, pos))))

    ns.on("set_dtmax", amplitude, after=True)  # right after the VOF advection moved "first"
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
    ap.add_argument("--lexicographic", action="store_true", help="relax as Basilisk's CPU build")
    args = ap.parse_args()
    if args.lexicographic:
        from basilisk_cupy.linear.lexicographic import set_relaxation  # noqa: PLC0415

        set_relaxation("lexicographic")
    n = 16
    while n <= args.nmax:
        r = run(n, args.backend)
        print(f"{r['n_per_wavelength']:g} {r['error']:g}   ref {REFERENCE.get(int(r['n_per_wavelength']))}  "
              f"steps {r['steps']}", flush=True)
        n *= 2


if __name__ == "__main__":
    main()

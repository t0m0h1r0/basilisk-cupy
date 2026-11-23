"""Taylor--Green vortices (port of Basilisk ``src/test/taylor-green.c``).

Incompressible Euler on the periodic unit square centred on the origin.
At t = 2 the error on |u| is printed as ``N avg rms max``, the same as
Basilisk's ``taylor-green.ref``.

    python -m cases.taylor_green [--backend numpy|cupy] [--nmax 256]
"""

from __future__ import annotations

import argparse
import math

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.basilisk_centered import CenteredNS

#: Basilisk reference (src/test/taylor-green.ref): N -> (avg, rms, max)
REFERENCE = {
    32: (0.0296821, 0.0325644, 0.058379),
    64: (0.00414356, 0.00455412, 0.00725007),
    128: (0.000534878, 0.00058883, 0.000907885),
    256: (6.78923e-05, 7.48498e-05, 0.000119386),
}


def setup(n: int, backend: str = "numpy") -> CenteredNS:
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(0, 1))
    x, y = grid.meshgrid()
    ns.u[0].interior = -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y)
    ns.u[1].interior = xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4.0
    ns.p.fill_ghosts()
    # g.x = -(p[1] - p[-1])/(2 Delta)
    for ax in range(2):
        ns.g[ax].interior = -(ns.p.data[grid.at({ax: 1})] - ns.p.data[grid.at({ax: -1})]) / (2 * grid.h[ax])
    ns.g.fill_ghosts()
    return ns


def error_norms(ns: CenteredNS) -> tuple[float, float, float]:
    """Basilisk ``normf(e)`` of e = |u| - |u_exact| (avg, rms, max)."""
    grid = ns.grid
    xp = grid.xp
    x, y = grid.meshgrid()
    u0 = -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y)
    v0 = xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)
    e = xp.sqrt(ns.u[0].interior ** 2 + ns.u[1].interior ** 2) - xp.sqrt(u0 ** 2 + v0 ** 2)
    return (float(xp.mean(xp.abs(e))), float(xp.sqrt(xp.mean(e ** 2))), float(xp.max(xp.abs(e))))


def run(n: int, backend: str = "numpy", t_end: float = 2.0) -> tuple[float, float, float]:
    ns = setup(n, backend)
    ns.start(events=[t_end])
    ns.run(t_end)
    return error_norms(ns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--nmax", type=int, default=256)
    args = ap.parse_args()
    n = 32
    while n <= args.nmax:
        avg, rms, mx = run(n, args.backend)
        ref = REFERENCE.get(n)
        print(f"{n} {avg:g} {rms:g} {mx:g}" + (f"   (Basilisk {ref[0]:g} {ref[1]:g} {ref[2]:g})" if ref else ""))
        n *= 2


if __name__ == "__main__":
    main()

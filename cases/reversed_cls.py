"""Time-reversed advection in a vortex with the conservative level set (TwoPhaseFlow's CLS), against
the VOF result of ``cases/reversed_vof.py`` (Basilisk ``src/test/reversed.c``) on the same velocity.

``psi = H_eps(R - |x - x_c|)`` (eps = 1.5 Delta), TVD-RK3 transport with FCCD face values, optional
compression--diffusion re-initialisation every ``reinit_every`` steps. Errors: norms of
``fraction(circle) - psi`` at t = 0 and t = T, and of the sharp indicator ``(psi > 1/2) - fraction``.

    python -m cases.reversed_cls [--nmax 7] [--backend numpy|cupy] [--reinit-every 0]
"""

from __future__ import annotations

import argparse
import math

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.time.stepping import Clock, dtnext, timestep

from .reversed_vof import DT, T, circle, velocity

CFL = 0.5


def norms(grid, e):
    xp = grid.xp
    a = xp.abs(e)
    dv = float(grid.h[0] * grid.h[1])
    return float(a.sum()) * dv, math.sqrt(float((a * a).sum()) * dv), float(a.max())


def run(n: int, backend: str = "numpy", reinit_every: int = 0, reinit_steps: int = 4, t_end: float = T):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    dist = 0.2 - xp.sqrt((x + 0.2) ** 2 + (y + 0.236338) ** 2)
    cls = CLSInterface.from_level_set(grid, dist, BoundaryConditions())
    f0 = fraction(grid, circle)
    e0 = norms(grid, f0 - cls.psi.interior)
    m0 = cls.mass()
    clock = Clock(events=[t_end / 4 * k for k in range(1, 5)])
    psi_min = psi_max = 0.5
    while clock.t < t_end - 1e-9 * t_end:
        uf = velocity(grid, clock.t)
        dt = dtnext(timestep(uf, DT, 0.8 if clock.i == 0 else CFL, clock), clock)
        cls.advect(uf, dt)
        if reinit_every and (clock.i + 1) % reinit_every == 0:
            cls.reinitialize(reinit_steps)
        psi_min = min(psi_min, float(xp.min(cls.psi.interior)))
        psi_max = max(psi_max, float(xp.max(cls.psi.interior)))
        clock.t = clock.tnext
        clock.i += 1
    e = norms(grid, f0 - cls.psi.interior)
    sharp = norms(grid, (cls.psi.interior > 0.5).astype(f0.dtype) - f0)
    return {"initial": e0, "final": e, "sharp": sharp, "mass_change": cls.mass() / m0 - 1.0,
            "psi_range": (psi_min, psi_max), "steps": clock.i}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=7)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--reinit-every", type=int, default=0)
    args = ap.parse_args()
    for level in range(5, args.nmax + 1):
        n = 1 << level
        r = run(n, args.backend, args.reinit_every)
        print(f"{n} psi-f avg/rms/max t=0 {r['initial'][0]:.4g}/{r['initial'][1]:.4g}/{r['initial'][2]:.3g}  "
              f"t=T {r['final'][0]:.4g}/{r['final'][1]:.4g}/{r['final'][2]:.3g}  sharp {r['sharp'][0]:.4g}/"
              f"{r['sharp'][1]:.4g}/{r['sharp'][2]:.3g}  mass {r['mass_change']:+.1e}  steps {r['steps']}", flush=True)


if __name__ == "__main__":
    main()

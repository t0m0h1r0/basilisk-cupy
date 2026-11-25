"""Stability of the explicit capillary loop: growth of the parasitic velocity of a static drop.

Drop R = 0.25, sigma = 1, rho1 = 1, walls, N x N on [-1/2, 1/2]^2, CLS with the GFM jump
(:class:`TwoPhaseCLS`, ``pressure_jump=True``), time step ``factor`` times tension.h's bound
``sqrt(rho_m Delta^3 / (pi sigma))``. The growth rate per step is fitted to ``log max|u|`` over
the second half of the run. ``curvature="fd"`` replaces the CCD curvature by 2nd-order central
differences of ``psi`` (same formula and band), for comparison.

    python -m cases.capillary_dt_scan --n 64 --rho2 0.01 --factors 0.9 0.95 1.0
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.interface.heights import pad_bc
from basilisk_cupy.time.basilisk_centered import CenteredNS


def _fd_curvature(psi):
    xp = psi.grid.xp
    q = psi.interior
    h = float(psi.grid.h[0])
    p = pad_bc(q, psi.bc, 1)
    c = p[1:-1, 1:-1]
    gx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * h)
    gy = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * h)
    gxx = (p[2:, 1:-1] - 2 * c + p[:-2, 1:-1]) / (h * h)
    gyy = (p[1:-1, 2:] - 2 * c + p[1:-1, :-2]) / (h * h)
    gxy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / (4 * h * h)
    k = -(gy * gy * gxx - 2 * gx * gy * gxy + gx * gx * gyy) / (gx * gx + gy * gy + 1e-30) ** 1.5
    return xp.where((q > 0.01) & (q < 0.99), k, 0.0)


def growth(n: int, rho2: float, factor: float, steps: int = 800, mu: float = 0.1, curvature: str = "ccd",
           backend: str = "numpy", eps_factor: float | None = None, cls_kw: dict | None = None,
           ns_kw: dict | None = None, **tp_kw):
    """``tp_kw`` go to :class:`TwoPhaseCLS` (e.g. ``reinit_every=1, reinit_method="compression"``)."""
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8,
                    **(ns_kw or {}))
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, 0.25 - xp.sqrt(x * x + y * y), BoundaryConditions(),
                                      eps=None if eps_factor is None else eps_factor * float(grid.h[0]),
                                      **(cls_kw or {}))
    if curvature == "fd":
        cls.curvature = lambda **kw: _fd_curvature(cls.psi)
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=mu, mu2=mu, pressure_jump=True,
                     capillary_dt_factor=factor, **tp_kw)
    tp.add_surface_tension(1.0)
    ns.start()
    us = []
    for _ in range(steps):
        ns.step()
        u = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
        us.append(u)
        if not math.isfinite(u) or u > 1.0:
            break
    us = np.array(us)
    tail = np.log(np.maximum(us[len(us) // 2:], 1e-300))
    rate = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 5 else float("inf")
    return {"rate": rate, "umax": float(us[-1]), "steps": len(us), "dt": float(to_numpy(ns.clock.dt)),
            "stable": len(us) == steps and rate < 1e-3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--rho2", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--curvature", default="ccd")
    ap.add_argument("--factors", type=float, nargs="+", default=[0.9, 0.95, 1.0])
    args = ap.parse_args()
    for f in args.factors:
        r = growth(args.n, args.rho2, f, args.steps, args.mu, args.curvature)
        print(f"factor {f:.3f}: rate/step {r['rate']:+.4f} umax {r['umax']:.2e} steps {r['steps']} "
              f"{'stable' if r['stable'] else 'UNSTABLE'}", flush=True)


if __name__ == "__main__":
    main()

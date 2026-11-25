"""Phase 2: static drop with three methods and density ratios 1 ... 1000 (docs/notes/56).

R = 0.25 in the unit box with walls, sigma = 1, mu = 0.1 (both phases), rho1 = 1 inside, rho2 = 1/ratio, to t = 1:

* ``vof``: Basilisk's VOF + tension.h (height-function curvature, CSF in iforce.h form) on ``CenteredNS``;
* ``centered``: ``CenteredNS`` + CLS + GFM pressure jump (sharp), CCD viscosity;
* ``imex``: ``TwoPhaseIMEX`` (the paper's stages with this port's parts, defaults).

    python -m cases.phase2_static [--jobs N] [--t-end 1.0]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS
from basilisk_cupy.time.two_phase_imex import WALLS, TwoPhaseIMEX

R, SIGMA, MU = 0.25, 1.0, 0.1


def run(method: str, n: int, ratio: float, t_end: float = 1.0) -> dict:
    warnings.filterwarnings("ignore")
    rho2 = 1.0 / ratio
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    if method == "imex":
        ns = TwoPhaseIMEX(grid, R - np.hypot(x, y), rho1=1.0, rho2=rho2, mu1=MU, mu2=MU, sigma=SIGMA, tolerance=1e-8)
        f = ns.cls.psi

        def pressure():
            return ns.pressure()
    else:
        kw = dict(uf_normal={w: 0.0 for w in WALLS}, tolerance=1e-8)
        if method == "centered":
            ns = CenteredNS(grid, viscous=CCDViscosity(), **kw)
            cls = CLSInterface.from_level_set(grid, R - np.hypot(x, y), BoundaryConditions(), divergence="fccd")
            tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=MU, mu2=MU, pressure_jump=True)
            f = cls.psi
        else:
            ns = CenteredNS(grid, **kw)
            f = CellField.from_interior(grid, fraction(grid, lambda x, y: R * R - x * x - y * y), BoundaryConditions())
            tp = TwoPhase(ns, f, rho1=1.0, rho2=rho2, mu1=MU, mu2=MU)
        tp.add_surface_tension(SIGMA)

        def pressure():
            return tp.full_pressure(ns) if hasattr(tp, "full_pressure") else ns.p.interior
    ns.start(events=[t_end])
    v0 = float(np.sum(f.interior))
    umax, t0, failed = 0.0, time.time(), False
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        u = max(float(np.abs(c.interior).max()) for c in ns.u)
        if not math.isfinite(u) or u > 10.0:
            failed = True
            break
        umax = max(umax, u)
    p = pressure()
    r = np.hypot(x, y)
    return {"jump": float(p[r < 0.1].mean() - p[r > 0.4].mean()), "u_end": u, "u_max": umax,
            "volume": float(np.sum(f.interior)) / v0 - 1.0, "steps": ns.clock.i, "t": ns.clock.t,
            "wall": time.time() - t0, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--t-end", type=float, default=1.0)
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(method=m, n=n, ratio=r, t_end=args.t_end) for n in (32, 64) for r in (1.0, 10.0, 100.0, 1000.0)
             for m in ("vof", "centered", "imex")]
    print("method N ratio: jump (error) u_end u_max volume steps wall")
    for it, r in zip(items, pmap(run, items, args.jobs)):
        print(f"{it['method']:8s} {it['n']:3d} {it['ratio']:6g}: {r['jump']:.5f} ({r['jump'] / 4 - 1:+.2e}) "
              f"{r['u_end']:.2e} {r['u_max']:.2e} {r['volume']:+.1e} {r['steps']} {r['wall']:.0f}s"
              f"{'  FAILED at t = %.3f' % r['t'] if r['failed'] else ''}")


if __name__ == "__main__":
    main()

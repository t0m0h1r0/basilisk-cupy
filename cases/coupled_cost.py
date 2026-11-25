"""Cost of the coupled iterations for a fixed physical time (docs/notes/36).

Oscillating drop (R = 0.25 (1 + 0.05 cos 2 theta), sigma = 1, mu = 0.1, walls, CLS + GFM) to ``t_end``,
each scheme at its own factor of tension.h's step. Reported: steps, outer iterations, wall time, and the
error of the deformation ``D(t_end) = int psi (x^2 - y^2)`` and of ``max |u|`` against a reference
(converged iterations at 0.25 x).

    python -m cases.coupled_cost
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS

CONVERGED = {"coupled_iterations": 30, "coupled_tolerance": 1e-10, "coupled_on_failure": "ignore"}


def oscillation(n, rho2, factor, t_end=0.3, backend="numpy", jump_mode="sharp", **ns_kw):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10, **ns_kw)
    th = xp.arctan2(y, x)
    cls = CLSInterface.from_level_set(grid, 0.25 * (1 + 0.05 * xp.cos(2 * th)) - xp.sqrt(x * x + y * y),
                                      BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, pressure_jump=True, capillary_dt_factor=factor,
                     jump_mode=jump_mode)
    tp.add_surface_tension(1.0)
    ns.start()
    its, t0 = 0, time.perf_counter()
    ok = True
    while ns.clock.t < t_end - 1e-12:
        ns.DT = t_end - ns.clock.t  # the last step lands on t_end
        ns.step()
        its += ns.coupled_info.get("iterations", 1) if ns.coupled_iterations else 1
        u = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
        if not np.isfinite(u) or u > 10:
            ok = False
            break
    wall = time.perf_counter() - t0
    dA = float(grid.h[0] * grid.h[1])
    D = float(xp.sum(cls.psi.interior * (x * x - y * y))) * dA
    return {"steps": ns.clock.i, "iterations": its, "wall": wall, "D": D, "umax": u, "ok": ok}


RTOL = {"coupled_iterations": 30, "coupled_tolerance": 1e-13, "coupled_rtol": 1e-2, "coupled_on_failure": "ignore"}
RUNS = [("K0", 0.95, {}, "sharp"), ("K2", 1.6, {"coupled_iterations": 2}, "sharp"),
        ("tol 1e-10", 1.7, CONVERGED, "sharp"),
        ("rtol 1e-2", 1.7, RTOL, "sharp"), ("rtol 1e-2 + anderson2", 1.7, dict(RTOL, coupled_anderson=2), "sharp"),
        ("K0", 0.95, {}, "smooth"), ("rtol 1e-2", 1.7, RTOL, "smooth"),
        ("rtol 1e-2 + predictor + anderson2", 1.7, dict(RTOL, coupled_anderson=2, coupled_predictor="linear"), "smooth")]


def main() -> None:
    warnings.filterwarnings("ignore")
    n, rho2 = 32, 0.01
    refs = {}
    for name, f, kw, jump in RUNS:
        if jump not in refs:
            refs[jump] = oscillation(n, rho2, 0.25, jump_mode=jump, **CONVERGED)
            r = refs[jump]
            print(f"N={n} ratio {1 / rho2:.0f} {jump} reference (0.25 x): D {r['D']:.6e} max|u| {r['umax']:.4e} "
                  f"steps {r['steps']}", flush=True)
        ref = refs[jump]
        r = oscillation(n, rho2, f, jump_mode=jump, **kw)
        print(f"  {jump:6s} {name:34s} f {f}: steps {r['steps']} iterations {r['iterations']} wall {r['wall']:.1f}s "
              f"D err {abs(r['D'] - ref['D']) / abs(ref['D']):.2e} max|u| err "
              f"{abs(r['umax'] - ref['umax']) / ref['umax']:.2e} ok {r['ok']}", flush=True)


if __name__ == "__main__":
    main()

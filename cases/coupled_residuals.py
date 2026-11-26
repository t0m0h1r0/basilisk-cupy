"""How well the outer iterations of ``CenteredNS(coupled_iterations=K)`` solve the coupled step (docs/notes/36).

An oscillating drop (R = 0.25 (1 + 0.05 cos 2 theta), sigma = 1, mu = 0.1, rho1 = 1, walls, CLS + GFM) is run
``warm`` steps with converged iterations at ``factor`` times tension.h's step; from that state one step is
taken with K = 1 .. ``kmax`` iterations and with ``kref`` (the reference). Per iteration k:

* ``change``: ``dt max |g_k - g_{k-1}|`` (what ``coupled_criterion="change"`` bounds);
* ``residual``: the momentum residual ``max |u_adv + dt g - (I - dt/rho L) u|`` of the state the step would
  end with (``coupled_criterion="residual"``);
* ``error``: ``max |u_K - u_ref|``, the distance of the K-iteration step to the converged one.

    python -m cases.coupled_residuals
"""

from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.constants import PI
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS


def drop(n=32, rho2=0.01, mu=0.1, factor=1.6, warm=40, mu2=None, jump_mode="sharp", cell_density="diffuse", **ns_kw):
    """``cell_density``: the study (docs/notes/36) used two-phase.h's diffuse cell density; with the sharp one (the
    default since docs/notes/69) the outer iterations converge more slowly."""
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    kw = {"coupled_iterations": 30, "coupled_tolerance": 1e-13, "coupled_on_failure": "ignore"}
    kw.update(ns_kw)
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-14, **kw)
    th = np.arctan2(y, x)
    cls = CLSInterface.from_level_set(grid, 0.25 * (1 + 0.05 * np.cos(2 * th)) - np.hypot(x, y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=mu, mu2=mu if mu2 is None else mu2, pressure_jump=True,
                     capillary_dt_factor=factor, jump_mode=jump_mode, cell_density=cell_density)
    tp.add_surface_tension(1.0)
    ns.start()
    for _ in range(warm):
        ns.step()
    return ns


def one_step(ns, K, **attrs):
    m = copy.deepcopy(ns, {id(ns.grid): ns.grid})  # the grid (it holds the array module) is shared
    m.coupled_iterations = K
    m.coupled_tolerance = None
    for k, v in attrs.items():
        setattr(m, k, v)
    t = time.perf_counter()
    m.step()
    return m, time.perf_counter() - t


def iterations(ns, kmax=12, kref=40, **attrs):
    """Per-iteration change, residual and error of one step from ``ns`` (not modified)."""
    ref, _ = one_step(ns, kref)
    uref = [c.interior.copy() for c in ref.u]
    m, _ = one_step(ns, kmax, coupled_criterion="residual", **attrs)
    dt = float(m.clock.dt)
    out = []
    for K in range(1, kmax + 1):
        mk, _ = one_step(ns, K, **attrs)
        err = max(float(np.abs(a.interior - b).max()) for a, b in zip(mk.u, uref))
        out.append({"k": K, "change": dt * m.coupled_history[K - 1], "residual": m.coupled_residuals[K - 1],
                    "error": err, "div": mk.divergence_residuals()})
    return out, float(max(float(np.abs(c).max()) for c in uref))


def main() -> None:
    warnings.filterwarnings("ignore")
    for n in (32, 64):
        for rho2 in (1.0, 0.01, 0.001):
            ns = drop(n, rho2)
            rows, umax = iterations(ns)
            nu = 0.1 / rho2
            dt = float(ns.clock.dt)
            print(f"N={n} ratio {1 / rho2:.0f}: max|u| {umax:.2e}, dt {dt:.3e}, nu_max dt/h^2 {nu * dt * n * n:.1f}")
            for r in rows:
                print(f"  k={r['k']:2d} change {r['change']:.2e} residual {r['residual']:.2e} error {r['error']:.2e}"
                      f"  div face {r['div'][0]:.1e} cell {r['div'][1]:.1e}", flush=True)


if __name__ == "__main__":
    main()

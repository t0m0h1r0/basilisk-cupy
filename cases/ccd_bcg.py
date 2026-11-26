"""CCD-BCG prototype (docs/notes/72, idea A): Basilisk's Bell--Colella--Glaz advection (``bcg.h``: upwind face values
from a space--time Taylor extrapolation of the upwind cell) with the cell gradient from CCD instead of centred
differences, and optionally the second-order term from the same CCD solve:

    face from the left cell  u + (h/2)(1 - C) u' + (h^2/12)(1 - C)(1 - 2C) u''
    face from the right cell u - (h/2)(1 - |C|) u' + (h^2/12)(1 - |C|)(1 - 2|C|) u''      (C = u_n dt / h)

(``mode="ccd"``: the first two terms; ``"ccd2"``: all three, the mean over the swept interval of the
mean-preserving parabola ``q + q' xi + q''/2 (xi^2 - h^2/12)``, third order for 1-D constant advection and linearly
stable for 0 < C <= 1; ``"ccd2-point"``: the point value at the face and t + dt/2, ``(h^2/8)(1 - C)^2 u''``, the
first prototype, linearly unstable for small C, ``cases/ccd_bcg_stability.py``). The derivatives come from one compact solve per line (no wider stencil). ``enable(mode)``
patches ``basilisk_cupy.terms.convection.bcg`` in this process (prototype; the library is untouched).

    python -m cases.ccd_bcg [--jobs N]    # Taylor--Green convergence and the two-phase shear layer
"""

from __future__ import annotations

import argparse

_STATE = {"mode": None, "orig": None, "d2": {}}


def enable(mode: str | None) -> None:
    """``None``: Basilisk's BCG; ``"ccd"``; ``"ccd2"``; ``"ccd2-point"``."""
    import basilisk_cupy.terms.convection.bcg as bcg

    if _STATE["orig"] is None:
        _STATE["orig"] = (bcg.gradients, bcg._bcg_faces)
    gradients0, faces0 = _STATE["orig"]
    _STATE["mode"] = mode
    if mode is None:
        bcg.gradients, bcg._bcg_faces = gradients0, faces0
        return

    from basilisk_cupy.boundary import gradient_bcs
    from basilisk_cupy.discretization.ccd import CCDCellDerivative
    from basilisk_cupy.fields import CellField, VectorField

    ccd = CCDCellDerivative()

    def gradients(f, limiter=None):
        grid = f.grid
        g = VectorField.zeros(grid, gradient_bcs(f.bc, grid.ndim), name=f"grad({f.name})")
        d2 = []
        for ax in range(grid.ndim):
            g1, g2 = ccd.derivatives(f, ax)
            g[ax].interior = g1
            c2 = CellField(grid, bc=gradient_bcs(f.bc, grid.ndim)[ax], name="d2")
            c2.interior = g2
            d2.append(c2.fill_ghosts())
        _STATE["d2"][id(f.data)] = d2
        return g.fill_ghosts()

    def faces(f, axis, un, grad_axis, transverse, src, dt):
        face = faces0(f, axis, un, grad_axis, transverse, src, dt)
        if mode not in ("ccd2", "ccd2-point"):
            return face
        d2 = _STATE["d2"].get(id(f.data))
        if d2 is None:
            return face
        grid = f.grid
        xp = grid.xp
        h = grid.h[axis]
        n = grid.n[axis]
        sc = bcg._cells_along(d2[axis].data, grid, axis)
        lo = [slice(None)] * sc.ndim
        hi = list(lo)
        lo[axis] = slice(0, n + 1)
        hi[axis] = slice(1, n + 2)
        if mode == "ccd2-point":
            left = (1.0 - un) ** 2 * sc[tuple(lo)] * h * h / 8.0
            right = (1.0 + un) ** 2 * sc[tuple(hi)] * h * h / 8.0
        else:  # the swept mean of the mean-preserving parabola (un < 0 on the right: |C| = -un)
            left = (1.0 - un) * (1.0 - 2.0 * un) * sc[tuple(lo)] * h * h / 12.0
            right = (1.0 + un) * (1.0 + 2.0 * un) * sc[tuple(hi)] * h * h / 12.0
        return face + xp.where(un > 0, left, right)

    bcg.gradients, bcg._bcg_faces = gradients, faces


def taylor_green(mode, n: int, t_end: float = 0.5):
    """Inviscid Taylor--Green vortices on the periodic [0, 1]^2 (stationary): the error of u at t_end."""
    import math
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    enable(mode)
    grid = CartesianGrid((n, n))
    x, y = (to_numpy(a) for a in grid.meshgrid())
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    u0 = -np.cos(2 * math.pi * x) * np.sin(2 * math.pi * y)
    v0 = np.sin(2 * math.pi * x) * np.cos(2 * math.pi * y)
    ns.u[0].interior, ns.u[1].interior = grid.xp.asarray(u0), grid.xp.asarray(v0)
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()
    err = max(float(np.abs(to_numpy(ns.u[0].interior) - u0).max()), float(np.abs(to_numpy(ns.u[1].interior) - v0).max()))
    return {"mode": mode, "n": n, "err": err, "steps": ns.clock.i}


def shear(mode, n: int, ratio: float):
    from cases.convection_stability import shear as run

    enable(mode)
    return run("centered-vof", n, ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    import math

    from cases.parallel import pmap

    modes = (None, "ccd", "ccd2")
    tg = [dict(mode=m, n=n) for m in modes for n in (16, 32, 64)]
    res = pmap(taylor_green, tg, args.jobs)
    for m in modes:
        e = [r["err"] for r in res if r["mode"] == m]
        rates = [math.log2(e[k] / e[k + 1]) for k in range(len(e) - 1)]
        print(f"Taylor-Green {str(m):5s}: err {' '.join(f'{v:.2e}' for v in e)}  order {' '.join(f'{r:.2f}' for r in rates)}",
              flush=True)
    sh = [dict(mode=m, n=n, ratio=r) for m in modes for n in (64, 128) for r in (10.0, 100.0)]

    def show(k, it, r):
        s = r["series"]
        ke = [d["ke"] for d in s if "ke" in d]
        print(f"shear BCG {str(it['mode']):5s} N {it['n']:3d} ratio {it['ratio']:5g}: "
              + (f"FAILED at t {s[-1]['t']:.2f} " if r["failed"] else "")
              + (f"ke max {max(ke) / r['ke0']:.4f} end {ke[-1] / r['ke0']:.4f}" if ke else ""), flush=True)

    pmap(shear, sh, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

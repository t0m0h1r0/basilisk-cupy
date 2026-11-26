"""Curvature of a translated interface without a flow solve (docs/notes/76): a drop (R = 0.15) on the periodic
[0, 1]^2 moved with u = (1, 0), CFL 0.5, one period; the relative curvature error at t = 0, 1/4, 1/2, 3/4, 1.

* ``--vof``: the VOF field moved by vof.h's split PLIC transport; tension.h's height-function curvature and the CCD
  column curvature (``vof_curvature.ccd_curvature``) against 1/R in the interfacial cells;
* ``--filter``: the CCD column curvature with its run positions filtered, ``(I + alpha (-D2)^m) e_f = e``;
* ``--cls``: the CLS psi moved by its transport (SSP-RK3, mass closure), with or without xi-SDF re-initialisation
  every 10 steps; psi's curvature against 1/r (r the distance of the cell centre from the drop centre: each cell lies
  on its own level line) in the cells 0.1 < psi < 0.9; and one re-initialisation of the exact profile, no transport.

    python -m cases.moving_curvature [--vof] [--filter] [--cls] [--jobs 18]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

R = 0.15


def _grid(n):
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.boundary import Periodic

    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    return grid, bc


def vof(n: int, method: str, alpha: float = 0.0, m: int = 2, cfl: float = 0.5):
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import CellField
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.curvature import curvature
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA
    from basilisk_cupy.interface.vof import VOFInterface, vof_advection
    from basilisk_cupy.interface.vof_curvature import ccd_curvature

    grid, bc = _grid(n)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: R ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8),
                                bc)
    uf = FaceField(grid, name="u")
    uf.components[0][...] = 1.0
    itf = VOFInterface(f)
    dt = cfl / n
    steps = int(round(1.0 / dt))

    def err():
        phi = curvature(f.interior, f.bc, grid, 1.0) if method == "heights" else \
            ccd_curvature(f, 1.0, filter_alpha=alpha, filter_order=m)
        phi, F = np.asarray(phi), np.asarray(f.interior)
        sel = (phi != NODATA) & (F > 1e-6) & (F < 1 - 1e-6)
        e = np.abs(phi[sel]) * R - 1
        return float(np.sqrt(np.mean(e ** 2)))

    out = [err()]
    for i in range(steps):
        vof_advection([itf], uf, dt, i)
        if (i + 1) % (steps // 4) == 0:
            out.append(err())
    return out


def cls(n: int, reinit: int, cfl: float = 0.5):
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.cls import CLSInterface

    grid, bc = _grid(n)
    h = 1.0 / n
    xc = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    itf = CLSInterface.from_level_set(grid, R - np.hypot(X - 0.5, Y - 0.5), bc)
    uf = FaceField(grid, name="u")
    uf.components[0][...] = 1.0
    dt = cfl * h
    steps = int(round(1.0 / dt))

    def err(t):
        dx = (X - (0.5 + t) % 1.0 + 0.5) % 1.0 - 0.5
        r = np.hypot(dx, Y - 0.5)
        k, P = np.asarray(itf.curvature("psi")), np.asarray(itf.psi.interior)
        sel = (P > 0.1) & (P < 0.9)
        e = np.abs(k[sel]) * r[sel] - 1
        return float(np.sqrt(np.mean(e ** 2)))

    if reinit < 0:  # one re-initialisation of the exact profile
        before = err(0.0)
        itf.reinitialize()
        return [before, err(0.0)]
    out = [err(0.0)]
    for i in range(steps):
        itf.advect(uf, dt)
        if reinit and (i + 1) % reinit == 0:
            itf.reinitialize()
        if (i + 1) % (steps // 4) == 0:
            out.append(err((i + 1) * dt))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vof", action="store_true")
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--cls", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(label):
        return lambda k, it, r: print(f"{label} {it}: " + " ".join(f"{100 * e:.3g}%" for e in r), flush=True)

    if args.vof:
        pmap(vof, [dict(n=n, method=mm) for n in (128, 64, 32) for mm in ("heights", "ccd")], args.jobs,
             on_result=show("vof   rms |kappa R| - 1 at t = 0, 1/4, 1/2, 3/4, 1"))
    if args.filter:
        items = [dict(n=n, method="ccd", alpha=a, m=mm) for n in (128, 64, 32)
                 for a, mm in ((0.01, 2), (0.05, 2), (0.2, 2), (0.01, 3), (0.05, 3))]
        pmap(vof, items, args.jobs, on_result=show("filter rms |kappa R| - 1 at t = 0, 1/4, 1/2, 3/4, 1"))
    if args.cls:
        items = [dict(n=n, reinit=ri) for n in (128, 64, 32) for ri in (0, 10, -1)]
        pmap(cls, items, args.jobs, on_result=show("cls   rms |kappa r| - 1 (reinit -1: once, before / after)"))


if __name__ == "__main__":
    main()

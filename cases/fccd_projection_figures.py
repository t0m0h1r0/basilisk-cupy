"""FCCD projection with the smooth-pressure jump (docs/notes/32): ``fccd_projection.png``.

Top: static drop (equal densities, N = 64, 10 steps) with FCCD faces: the jump in the pressure (sharp)
against the smooth pressure p~ = p - H j_ext, and the full pressure rebuilt from p~. Bottom: capillary
wave damping and period errors against delta / h for fd2 faces and FCCD faces + CCD viscosity + smooth jump.

    python -m cases.fccd_projection_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS


def drop(n, fccd, smooth, steps=10):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8,
                    ops=FCCDFaceComplex() if fccd else None, viscous=CCDViscosity() if fccd else None)
    cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions(),
                                      divergence="full" if fccd else "fccd", ops=FCCDFaceComplex(order=6))
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1, pressure_jump=True,
                     jump_mode="smooth" if smooth else "sharp")
    tp.add_surface_tension(1.0)
    ns.start()
    for _ in range(steps):
        ns.step()
    return ns, tp, x, y


def capwave(n, f, fccd):
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay
    from cases.capwave_modes import period_decay

    saved = cw.MU
    cw.MU = saved * f
    try:
        kw = dict(interface="cls", pressure_jump=True, face_order=6)
        if fccd:
            kw.update(divergence="full", jump_mode="smooth", viscous=CCDViscosity(), ops_ns=FCCDFaceComplex())
        else:
            kw.update(divergence="fccd")
        r = cw.run(n, "numpy", **kw)
    finally:
        cw.MU = saved
    P = np.array(cw.PROSPERETTI)
    a = np.array(r["amplitudes"])
    p, d = decay(a, P[:len(a), 0])
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, saved * f)
    dh = math.sqrt(2 * saved * f / math.sqrt((2 * math.pi) ** 3 / 2)) / (2.0 / n)
    return dh, math.log(d) / math.log(dt_) - 1, p / pt - 1, r["error"]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ext = (-0.5, 0.5, -0.5, 0.5)
    for k, (smooth, title) in enumerate(((False, "FCCD faces, sharp jump"), (True, "FCCD faces, smooth p~"))):
        ns, tp, x, y = drop(64, True, smooth)
        sp = np.hypot(ns.u[0].interior, ns.u[1].interior)
        a = fig.add_subplot(gs[0, 2 * k])
        im = a.imshow(sp.T, origin="lower", extent=ext, cmap="magma")
        fig.colorbar(im, ax=a, shrink=0.8, format="%.0e")
        a.set_title(f"{title}\n|u| (max {sp.max():.1e})", fontsize=9)
        a = fig.add_subplot(gs[0, 2 * k + 1])
        p = ns.p.interior - ns.p.interior[np.hypot(x, y) > 0.4].mean()
        im = a.imshow(p.T, origin="lower", extent=ext, cmap="viridis")
        fig.colorbar(im, ax=a, shrink=0.8)
        a.set_title("solved pressure (p~ for smooth)" if smooth else "solved pressure", fontsize=10)
    rows = {False: [], True: []}
    for fccd in (False, True):
        for f in (0.5, 1.0):
            for n in (32, 64):
                rows[fccd].append(capwave(n, f, fccd))
    a0 = fig.add_subplot(gs[1, 0:2])
    a1 = fig.add_subplot(gs[1, 2:4])
    for fccd, style, label in ((False, "o", "fd2 faces"), (True, "s", "FCCD faces + CCD viscosity + smooth jump")):
        r = rows[fccd]
        a0.plot([q[0] for q in r], [q[1] for q in r], style, ms=8, label=label)
        a1.plot([q[0] for q in r], [q[2] for q in r], style, ms=8, label=label)
    for a, t in ((a0, "capillary wave: damping-rate error"), (a1, "capillary wave: period error")):
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel("delta / h")
        a.set_title(t)
        a.legend()
        a.grid(alpha=0.3)
    fig.savefig(f"{args.out}/fccd_projection.png", dpi=100)
    print("wrote", f"{args.out}/fccd_projection.png")


if __name__ == "__main__":
    main()

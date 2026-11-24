"""Figures of the re-initialisation comparison (docs/notes/26):

1. ``reinit_static_circle.png``: a circle of 20 cells (N = 64) after 1000 re-initialisations of one
   pseudo-step, change of psi and curvature error along the interface, per method;
2. ``reinit_reversed.png``: Basilisk's reversed vortex (N = 64), psi = 1/2 at T/2 and T, per method
   (re-initialisation every 10 steps);
3. ``reinit_static_drop.png``: static drop with the GFM jump (N = 64, densities 1 / 0.01, mu = 0.1),
   density, velocity and pressure after 400 steps at tension.h's time step x sqrt(5/12), per method
   (re-initialisation after every step).

    python -m cases.acls_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface

METHODS = [("none", "no re-initialisation"), ("compression", "compression-diffusion (FCCD)"),
           ("acls2008", "ACLS 2008 (2nd order)"), ("acls2017", "ACLS 2017 (2nd order)"),
           ("acls-fccd", "ACLS 2008 form, FCCD")]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def static_circle(out):
    plt = _plt()
    n, R, cx, cy = 64, 10 / 64, 0.0123, -0.0217
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    r = np.hypot(x - cx, y - cy)
    th = np.arctan2(y - cy, x - cx)
    meths = [m for m in METHODS if m[0] != "none"]
    fig, ax = plt.subplots(2, len(meths), figsize=(4.2 * len(meths), 8), constrained_layout=True)
    for k, (m, label) in enumerate(meths):
        cls = CLSInterface.from_level_set(grid, R - r, BoundaryConditions())
        q0 = cls.psi.interior.copy()
        for _ in range(1000):
            cls.reinitialize(1, method=m)
        q = cls.psi.interior
        im = ax[0, k].imshow((q - q0).T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="RdBu_r",
                             vmin=-3e-3, vmax=3e-3)
        ax[0, k].contour(x, y, q, [0.5], colors="k", linewidths=0.8)
        ax[0, k].set_xlim(cx - 2.2 * R, cx + 2.2 * R)
        ax[0, k].set_ylim(cy - 2.2 * R, cy + 2.2 * R)
        ax[0, k].set_title(f"{label}\nmax|psi - psi0| = {np.abs(q - q0).max():.1e}", fontsize=10)
        k_ = cls.curvature()
        band = np.abs(r - R) < cls.eps
        ax[1, k].plot(np.degrees(th[band]), (k_[band] * r[band]) - 1, ".", ms=2)
        ax[1, k].set_ylim(-0.6, 0.6)
        ax[1, k].set_xlabel("angle (deg)")
        ax[1, k].set_title(f"kappa r - 1 in |phi| < eps (max {np.abs(k_[band] * R - R / r[band]).max():.3f})",
                           fontsize=9)
        ax[1, k].grid(alpha=0.3)
    fig.colorbar(im, ax=ax[0, :], shrink=0.8, label="psi - psi0")
    fig.suptitle("Static circle (20 cells), 1000 re-initialisations of one pseudo-step (ACLS 2017: dtau / 4)")
    fig.savefig(f"{out}/reinit_static_circle.png", dpi=100)


def reversed_vortex(out):
    plt = _plt()
    from cases import reversed_cls as rc

    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    fig, ax = plt.subplots(2, len(METHODS), figsize=(3.6 * len(METHODS), 7.2), constrained_layout=True)
    for k, (m, label) in enumerate(METHODS):
        for row, t_end in enumerate((rc.T / 2, rc.T)):
            r = rc.run(n, "numpy", reinit_every=0 if m == "none" else 10, method=m if m != "none" else "compression",
                       t_end=t_end)
            a = ax[row, k]
            a.imshow(r["psi"].T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="Blues", vmin=0, vmax=1)
            a.contour(x, y, r["psi"], [0.5], colors="r", linewidths=0.8)
            if row == 1:
                tt = np.linspace(0, 2 * math.pi, 200)
                a.plot(-0.2 + 0.2 * np.cos(tt), -0.236338 + 0.2 * np.sin(tt), "k--", lw=0.6)
                a.set_title(f"t = T, sharp error {r['sharp'][0]:.4f}", fontsize=9)
            else:
                a.set_title(f"{label}\nt = T/2", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Reversed vortex (N = 64), re-initialisation every 10 steps (dashed: exact final circle)")
    fig.savefig(f"{out}/reinit_reversed.png", dpi=100)


def static_drop(out):
    plt = _plt()
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    n = 64
    fig, ax = plt.subplots(len(METHODS), 3, figsize=(12, 3.6 * len(METHODS)), constrained_layout=True)
    for k, (m, label) in enumerate(METHODS):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8)
        x, y = grid.meshgrid()
        rr = np.hypot(x, y)
        cls = CLSInterface.from_level_set(grid, 0.25 - rr, BoundaryConditions())
        kw = {} if m == "none" else {"reinit_every": 1, "reinit_method": m}
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=0.01, mu1=0.1, mu2=0.1, pressure_jump=True, **kw)
        tp.add_surface_tension(1.0)
        ns.start()
        for _ in range(400):
            ns.step()
        u, v, p = ns.u[0].interior, ns.u[1].interior, ns.p.interior
        p = p - p[rr > 0.4].mean()
        sp = np.hypot(u, v)
        ext = (-0.5, 0.5, -0.5, 0.5)
        a = ax[k, 0].imshow(tp.rho(cls.psi.interior).T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(a, ax=ax[k, 0], shrink=0.8)
        b = ax[k, 1].imshow(sp.T, origin="lower", extent=ext, cmap="magma")
        s = 3
        ax[k, 1].quiver(x[::s, ::s], y[::s, ::s], u[::s, ::s], v[::s, ::s], color="w", width=0.004)
        fig.colorbar(b, ax=ax[k, 1], shrink=0.8, format="%.0e")
        c = ax[k, 2].imshow(p.T, origin="lower", extent=ext, cmap="viridis", vmin=-0.2, vmax=4.4)
        fig.colorbar(c, ax=ax[k, 2], shrink=0.8)
        ax[k, 0].set_ylabel(label, fontsize=10)
        ax[k, 1].set_title(f"|u| (max {sp.max():.1e})", fontsize=10)
        ax[k, 2].set_title(f"p - p_out (jump {p[rr < 0.1].mean():.4f})", fontsize=10)
        if k == 0:
            ax[k, 0].set_title("density", fontsize=10)
    fig.suptitle("Static drop, GFM jump, density ratio 100, N = 64, 400 steps (re-initialisation after every step)")
    fig.savefig(f"{out}/reinit_static_drop.png", dpi=100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    for name, f in (("circle", static_circle), ("reversed", reversed_vortex), ("drop", static_drop)):
        if args.only in ("all", name):
            f(args.out)
            print("done", name, flush=True)


if __name__ == "__main__":
    main()

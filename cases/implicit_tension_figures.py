"""Figures for the semi-implicit surface tension (docs/notes/27): ``implicit_tension.png``.

Left: static drop with the GFM jump (density ratio 100, mu = 0.1, N = 64) at tension.h's time step
(factor 1, where the explicit capillary loop grows): max |u| against the step for theta = 0, 1, 4, 16.
Middle: the same drop after 800 steps with theta = 1: velocity and pressure. Right: capillary-wave
amplitude (16 points per wavelength) against Prosperetti's solution for theta = 0, 1, 4.

    python -m cases.implicit_tension_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS


def drop(theta, steps=800, n=64):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8)
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=0.01, mu1=0.1, mu2=0.1, pressure_jump=True,
                     capillary_dt_factor=1.0, implicit_tension=theta)
    tp.add_surface_tension(1.0)
    ns.start()
    hist = []
    for _ in range(steps):
        ns.step()
        u = max(float(np.max(np.abs(c.interior))) for c in ns.u)
        hist.append(u)
        if not np.isfinite(u) or u > 1.0:
            break
    return np.array(hist), ns, x, y


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases import capillary_wave as cw

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig = plt.figure(figsize=(17, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    ax0 = fig.add_subplot(gs[0, 0])
    keep = None
    for theta in (0, 1, 4, 16):
        hist, ns, x, y = drop(theta)
        ax0.semilogy(hist, label=f"theta = {theta}")
        if theta == 1:
            keep = (ns, x, y)
    ax0.set_xlabel("step")
    ax0.set_ylabel("max |u|")
    ax0.set_title("static drop, ratio 100, N = 64, tension.h's dt")
    ax0.legend()
    ax0.grid(alpha=0.3)
    ns, x, y = keep
    u, v = ns.u[0].interior, ns.u[1].interior
    r = np.hypot(x, y)
    p = ns.p.interior - ns.p.interior[r > 0.4].mean()
    ax1 = fig.add_subplot(gs[0, 1])
    sp = np.hypot(u, v)
    im = ax1.imshow(sp.T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="magma")
    s = 3
    ax1.quiver(x[::s, ::s], y[::s, ::s], u[::s, ::s], v[::s, ::s], color="w", width=0.004)
    fig.colorbar(im, ax=ax1, shrink=0.8, format="%.0e")
    ax1.set_title(f"theta = 1, step 800: |u| (max {sp.max():.1e})")
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(p.T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="viridis", vmin=-0.2, vmax=4.4)
    fig.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title(f"pressure - p_out (jump {p[r < 0.1].mean():.4f})")
    ax3 = fig.add_subplot(gs[0, 3])
    P = np.array(cw.PROSPERETTI)
    for theta in (0, 1, 4):
        res = cw.run(32, "numpy", interface="cls", pressure_jump=True, divergence="fccd", face_order=6,
                     implicit_tension=theta)
        a = np.array(res["amplitudes"])
        ax3.plot(P[:len(a), 0], a, label=f"theta = {theta} (error {res['error']:.4f})")
    ax3.plot(P[:, 0], P[:, 1], "k--", lw=1, label="Prosperetti")
    ax3.set_xlabel("omega_0 t")
    ax3.set_ylabel("amplitude")
    ax3.set_title("capillary wave, 16 pts per wavelength", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.savefig(f"{args.out}/implicit_tension.png", dpi=100)
    print("wrote", f"{args.out}/implicit_tension.png")


if __name__ == "__main__":
    main()

"""Coupled viscosity-force-projection iterations in CenteredNS (docs/notes/34): ``coupled_viscous.png``.

Left: measured stability limit (multiple of tension.h's step) against the viscosity, static drop (GFM, equal
densities, N = 64), Basilisk's splitting (K = 0) and one coupled iteration (K = 1); the model limits
(omega dt) rescaled to the measured inviscid limit. Middle: max |u| of the static drop at 1.6 times the step,
mu = 0.1. Right: capillary-wave amplitude with K = 1 at 0.65 and 4 times the step.

    python -m cases.coupled_viscous_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: Measured by bisection (600 steps; stable / unstable factor).
LIMITS = {0: {0.0: (1.600, 1.650), 0.01: (1.400, 1.450), 0.1: (1.150, 1.200), 1.0: (1.000, 1.050)},
          1: {0.0: (1.438, 1.656), 0.01: (1.656, 1.875), 0.1: (2.094, 2.312), 1.0: (2.531, 2.750)}}


def drop_history(K: int, factor: float = 1.6, steps: int = 400):
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.interface.cls import CLSInterface
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8,
                    coupled_iterations=K)
    cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1, pressure_jump=True, capillary_dt_factor=factor)
    tp.add_surface_tension(1.0)
    ns.start()
    out = []
    for _ in range(steps):
        ns.step()
        u = max(float(np.abs(c.interior).max()) for c in ns.u)
        out.append(u)
        if not np.isfinite(u) or u > 1.0:
            break
    return np.array(out)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import basilisk_cupy.time.basilisk_centered as bc
    from cases import capillary_wave as cw
    from cases.coupled_stability import limit

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    mus = [0.003, 0.01, 0.1, 1.0]
    for K, style in ((0, "o-"), (1, "s-")):
        xs = [m if m > 0 else 0.003 for m in LIMITS[K]]
        mid = [np.mean(v) for v in LIMITS[K].values()]
        ax[0].semilogx(xs, mid, style, label=f"measured, K = {K}")
        ax[0].fill_between(xs, [v[0] for v in LIMITS[K].values()], [v[1] for v in LIMITS[K].values()], alpha=0.2)
    ax[0].set_xlabel("mu (0 plotted at 0.003)")
    ax[0].set_ylabel("stable factor of tension.h's step")
    ax[0].set_title("static drop, N = 64: viscosity now stabilises")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    for K in (0, 1):
        h = drop_history(K)
        ax[1].semilogy(h, label=f"K = {K}")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("max |u|")
    ax[1].set_title("static drop, mu = 0.1, 1.6 x tension.h's step")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    P = np.array(cw.PROSPERETTI)
    orig = bc.CenteredNS.__init__

    def init(self, *a, **k):
        k.setdefault("coupled_iterations", 1)
        orig(self, *a, **k)

    bc.CenteredNS.__init__ = init
    try:
        for fac in (0.6455, 4.0):
            r = cw.run(64, "numpy", interface="cls", pressure_jump=True, divergence="fccd", face_order=6,
                       capillary_dt_factor=fac)
            a = np.array(r["amplitudes"])
            ax[2].plot(P[:len(a), 0], a, label=f"K = 1, {fac} x step ({r['steps']} steps, error {r['error']:.4f})")
    finally:
        bc.CenteredNS.__init__ = orig
    ax[2].plot(P[:, 0], P[:, 1], "k:", label="Prosperetti")
    ax[2].set_xlabel("omega_0 t")
    ax[2].set_title("capillary wave, 32 points per wavelength")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    del limit, mus
    fig.savefig(f"{args.out}/coupled_viscous.png", dpi=100)
    print("wrote", f"{args.out}/coupled_viscous.png")


if __name__ == "__main__":
    main()

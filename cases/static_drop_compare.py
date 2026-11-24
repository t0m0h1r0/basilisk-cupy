"""Static drop (R = 0.25, sigma = 1, mu = 0.1, rho1 = 1 inside, rho2 = 0.1 outside, N = 64, walls) at t = 0.2:
Basilisk (C, fields dumped by ``tools/basilisk/static_drop.c``) against this port with VOF, CLS + CSF
and CLS + GFM (default capillary time-step factor sqrt(5/12), hence more steps). The VOF run relaxes
lexicographically, as Basilisk's CPU build. Needs matplotlib (``pip install .[plot]``).

    qcc -O2 -Wall tools/basilisk/static_drop.c -o static_drop -lm && ./static_drop 2> log
    python -m cases.static_drop_compare --basilisk fields-0.2.txt --basilisk-log log
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.linear.lexicographic import set_relaxation
from basilisk_cupy.time.basilisk_centered import CenteredNS

N, T, RHO1, RHO2, MU, SIGMA, R = 64, 0.2, 1.0, 0.1, 0.1, 1.0, 0.25


def run(kind: str):
    grid = CartesianGrid((N, N), origin=(-0.5, -0.5))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-8)
    x, y = grid.meshgrid()
    if kind == "vof":
        f = CellField.from_interior(grid, fraction(grid, lambda x, y: R * R - x * x - y * y), BoundaryConditions())
        tp = TwoPhase(ns, f, rho1=RHO1, rho2=RHO2, mu1=MU, mu2=MU)
    else:
        cls = CLSInterface.from_level_set(grid, R - np.sqrt(x * x + y * y), BoundaryConditions())
        tp = TwoPhaseCLS(ns, cls, rho1=RHO1, rho2=RHO2, mu1=MU, mu2=MU, pressure_jump=(kind == "gfm"))
        f = cls.psi
    tp.add_surface_tension(SIGMA)
    hist = []
    ns.on("end_timestep", lambda s: hist.append(
        (s.clock.t, max(float(np.max(np.abs(c.interior))) for c in s.u))), after=True)
    ns.start(events=[T])
    while ns.clock.t < T - 1e-12:
        ns.step()
    rho = np.clip(f.interior, 0, 1) * (RHO1 - RHO2) + RHO2
    return {"x": x, "y": y, "rho": rho, "u": ns.u[0].interior.copy(), "v": ns.u[1].interior.copy(),
            "p": ns.p.interior.copy(), "hist": np.array(hist), "steps": ns.clock.i}


def load_basilisk(path, log):
    d = np.loadtxt(path)
    order = np.lexsort((d[:, 1], d[:, 0]))  # x major, then y -> [i, j]
    d = d[order]
    g = lambda c: d[:, c].reshape(N, N)
    h = np.loadtxt(log)
    return {"x": g(0), "y": g(1), "rho": np.clip(g(2), 0, 1) * (RHO1 - RHO2) + RHO2, "u": g(3), "v": g(4),
            "p": g(5), "hist": h[1:, [1, 3]], "steps": int(h[-1, 0])}


def figure(runs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 3.9 * len(names)), constrained_layout=True)
    pref = runs[names[0]]
    for row, name in enumerate(names):
        r = runs[name]
        x, y = r["x"], r["y"]
        ext = (-0.5, 0.5, -0.5, 0.5)
        p = r["p"] - r["p"][np.hypot(x, y) > 0.4].mean()  # outside pressure as reference
        sp = np.hypot(r["u"], r["v"])
        a = axes[row, 0].imshow(r["rho"].T, origin="lower", extent=ext, cmap="Blues", vmin=RHO2, vmax=RHO1)
        fig.colorbar(a, ax=axes[row, 0], shrink=0.8)
        b = axes[row, 1].imshow(sp.T, origin="lower", extent=ext, cmap="magma")
        k = 3
        axes[row, 1].quiver(x[::k, ::k], y[::k, ::k], r["u"][::k, ::k], r["v"][::k, ::k], color="w", scale=None,
                            width=0.004)
        fig.colorbar(b, ax=axes[row, 1], shrink=0.8, format="%.1e")
        c = axes[row, 2].imshow(p.T, origin="lower", extent=ext, cmap="viridis", vmin=-0.2, vmax=4.4)
        fig.colorbar(c, ax=axes[row, 2], shrink=0.8)
        jump = float(p[np.hypot(x, y) < 0.1].mean())
        axes[row, 0].set_ylabel(f"{name}\n({r['steps']} steps)", fontsize=12)
        axes[row, 0].set_title("density" if row == 0 else "")
        axes[row, 1].set_title(f"|u| (max {sp.max():.2e}) and direction")
        axes[row, 2].set_title(f"pressure - p_out (jump {jump:.4f}, sigma/R = 4)")
        for ax in axes[row]:
            ax.set_aspect("equal")
            th = np.linspace(0, 2 * math.pi, 200)
            ax.plot(R * np.cos(th), R * np.sin(th), "r--", lw=0.6)
    fig.suptitle(f"Static drop, t = {T}, N = {N}, rho 1 / 0.1, mu = {MU}, sigma = {SIGMA} (red: initial interface)",
                 fontsize=13)
    fig.savefig(out, dpi=110)

    fig2, ax = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for name in names:
        h = runs[name]["hist"]
        ax[0].semilogy(h[:, 0], h[:, 1], label=name)
        r = runs[name]
        j = N // 2
        pc = r["p"][:, j] - r["p"][np.hypot(r["x"], r["y"]) > 0.4].mean()
        ax[1].plot(r["x"][:, j], pc, ".-", ms=3, label=name)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("max |u|"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("parasitic velocity")
    ax[1].axhline(4.0, color="k", lw=0.5); ax[1].set_xlabel("x (y = h/2)"); ax[1].set_ylabel("p - p_out")
    ax[1].set_title("pressure across the drop"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig2.savefig(out.replace(".png", "_lines.png"), dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basilisk", required=True)
    ap.add_argument("--basilisk-log", required=True)
    ap.add_argument("--out", default="docs/figures/static_drop_compare.png")
    args = ap.parse_args()
    runs = {"Basilisk (C)": load_basilisk(args.basilisk, args.basilisk_log)}
    set_relaxation("lexicographic")  # as Basilisk's CPU build
    runs["basilisk-cupy VOF"] = run("vof")
    set_relaxation("redblack")
    runs["CLS + CSF"] = run("csf")
    runs["CLS + GFM"] = run("gfm")
    b, v = runs["Basilisk (C)"], runs["basilisk-cupy VOF"]
    for key in ("u", "v", "p"):
        dp = (v[key] - v[key].mean()) - (b[key] - b[key].mean()) if key == "p" else v[key] - b[key]
        print(f"VOF vs Basilisk max |d{key}| = {np.abs(dp).max():.2e}")
    figure(runs, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

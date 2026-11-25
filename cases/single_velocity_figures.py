"""One velocity field with a consistent projection (docs/notes/48): ``single_velocity.png``. Measured with
:mod:`cases.single_velocity` (N = 32, x 1, exact curvature, band-corrected kinematics) and copied here; panel (c)
is recomputed.

    python -m cases.single_velocity_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

DIVS = [1, 2, 4]
#: damping error at dt / dt0 = 1, 1/2, 1/4
STEPPED = {"CenteredNS (approx. projection), fd2": [0.0772, 0.0636, 0.0519],
           "CenteredNS, CCD": [0.0443, 0.03582, 0.0280],
           "exact cell projection + BE, fd2": [0.16107, 0.09693, 0.06466],
           "exact cell projection + CN, fd2": [0.03203, 0.03221, 0.03226],
           "exact cell projection + SDIRK2, fd2": [0.03215, 0.03224, 0.03226],
           "exact cell projection + CN, CCD": [0.01113, 0.01130, 0.01135],
           "exact cell projection + SDIRK2, CCD": [0.01125, 0.01133, 0.01136]}
#: semi-discrete (damping, frequency)
SEMI = {"CenteredNS dt -> 0 (fd2)": (0.032, -0.002), "exact cell, fd2": (0.0323, -0.0018), "exact cell, CCD": (0.0114, -0.0111),
        "MAC, fd2": (0.1288, 0.0282)}
#: static drop: divergence-free share of the GFM force (N = 32, 64)
DROP = {"faces (CenteredNS u_f)": (8.11e-4, 1.91e-4), "cells, exact projection": (8.61e-4, 2.07e-4)}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.single_velocity import interp_div
    from basilisk_cupy import CartesianGrid

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)
    x = [1 / d for d in DIVS]
    for k, (lab, y) in enumerate(STEPPED.items()):
        ax[0].plot(x, [100 * v for v in y], ("*-" if "CenteredNS" in lab else "o-" if "BE" in lab else "s-" if "CN" in lab else "^--"),
                   label=lab)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("dt / dt0")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) N = 32, x 1: exact cell projection makes the integrators 2nd order")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    labs = list(SEMI)
    xx = np.arange(len(labs))
    ax[1].bar(xx - 0.2, [100 * SEMI[l][0] for l in labs], 0.4, label="damping")
    ax[1].bar(xx + 0.2, [100 * SEMI[l][1] for l in labs], 0.4, label="frequency")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(labs, fontsize=8, rotation=10)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("error [%] (semi-discrete)")
    ax[1].set_title("(b) semi-discrete eigenvalue")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    grid = CartesianGrid((32, 32), length=(2.0, 2.0), origin=(0.0, -1.0))
    I, D, _ = interp_div(grid)
    n = 32
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mods = [(-1.0) ** (ii + jj), (-1.0) ** ii, (-1.0) ** jj]
    for name, S, mk, c in (("cell: D_c D_c^T (D_c = D_f I)", (D @ I) @ (D @ I).T, "o", "C0"), ("MAC: D_f D_f^T", D @ D.T, "s", "C1")):
        w, V = np.linalg.eigh(S)
        k = np.arange(1, 9)
        share = []
        for q in k:
            v = V[:, q].reshape(n, n)
            share.append(max(float(np.sum((v * m).reshape(n // 2, 2, n // 2, 2).sum(axis=(1, 3)) ** 2) / np.sum(v * v) / 4.0)
                             for m in mods))
        ax[2].plot(k, w[1:9], mk + "-", color=c, label=f"{name}: eigenvalue")
        for q, sh in zip(k, share):
            if sh > 0.2:
                ax[2].annotate(f"checkerboard {sh:.2f}", (q, w[q]), fontsize=7, xytext=(3, 6), textcoords="offset points", color=c)
    ax[2].set_xlabel("mode (after the constant)")
    ax[2].set_ylabel("eigenvalue of the pressure operator")
    ax[2].set_title("(c) checkerboard-modulated pressure modes as cheap as the smoothest (cell form)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")

    labs = list(DROP)
    xx = np.arange(2)
    for k, lab in enumerate(labs):
        ax[3].bar(xx + (k - 0.5) * 0.4, DROP[lab], 0.4, label=lab)
    ax[3].set_xticks(xx)
    ax[3].set_xticklabels(["N = 32", "N = 64"])
    ax[3].set_ylabel("|div-free part of the GFM force| / |force|")
    ax[3].set_title("(d) static drop: the balance is kept")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3, axis="y")
    fig.savefig(f"{args.out}/single_velocity.png", dpi=100)
    print("wrote", f"{args.out}/single_velocity.png")


if __name__ == "__main__":
    main()

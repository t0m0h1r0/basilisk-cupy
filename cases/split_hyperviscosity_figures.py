"""Evaluation of the spectrally split damping gamma Q^T M Q (docs/notes/42): ``split_hyperviscosity.png``.

Measured with :mod:`cases.split_hyperviscosity` in :mod:`cases.capwave_jacobian` (capillary wave, N = 32, x 1)
and :class:`cases.step_jacobian.StepMap` (static drop, density ratio 100, N = 32, K = 1, 1.6 x tension.h's
step); copied here. Panel (a) is recomputed from the CCD second derivative.

    python -m cases.split_hyperviscosity_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: gamma dt -> (capillary wave damping error, frequency error), m = 4, alpha = 10
CAPWAVE = {0.0: (-0.1643, -0.0357), 0.1: (-0.1647, -0.0358), 1.0: (-0.1672, -0.0361), 10.0: (-0.1750, -0.0372),
           100.0: (-0.1888, -0.0394)}
CAPWAVE_M2 = {10.0: (-0.1848, -0.0690)}
#: gamma dt -> |lambda| of the leading mode, ratio 100, K = 1, 1.6 x
DROP = {0.0: 1.1885, 1.0: 1.1851, 10.0: 1.1744, 100.0: 1.1574}
#: share of the velocity energy in Q's band
QSHARE = {"unstable mode, ratio 100": 0.073, "white noise": 0.104}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.split_hyperviscosity import laplacian_1d

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.0), constrained_layout=True)
    n = 64
    L = laplacian_1d(n, "periodic")
    lam = np.abs(np.linalg.eigvals(L)).max()
    kh = np.linspace(0.01, np.pi, 200)
    js = np.arange(1, n // 2 + 1)
    kj = 2 * np.pi * js / n
    symj = np.array([((L @ np.cos(k * np.arange(n)))[0] / 1.0) for k in kj])
    for m, a, st in ((4, 10.0, "-"), (2, 10.0, "--"), (8, 10.0, ":")):
        s = symj / lam
        ax[0].semilogy(kj, a * s ** m / (1 + a * s ** m), st, label=f"Q, m = {m}, alpha = {a:g} (CCD symbol)")
    for x, lab in ((2 * np.pi * 2.0 / 32, "capillary wave in x (16 pts/wavelength)"), (1 / 0.92, "vortex layer, 1/delta, delta/h = 0.92"),
                   (1 / 0.65, "vortex layer, delta/h = 0.65")):
        ax[0].axvline(x, color="k", ls=":", lw=1)
        ax[0].annotate(lab, (x, 1e-6), rotation=90, fontsize=7, xytext=(3, 0), textcoords="offset points")
    ax[0].set_ylim(1e-8, 2)
    ax[0].set_xlabel("k h")
    ax[0].set_ylabel("Q(k)")
    ax[0].set_title("(a) high-wavenumber band from the CCD Laplacian")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    g = [k for k in CAPWAVE if k > 0]
    ax[1].semilogx([0.01] + g, [100 * CAPWAVE[0.0][0]] + [100 * CAPWAVE[k][0] for k in g], "o-", label="damping error, m = 4")
    ax[1].semilogx([0.01] + g, [100 * CAPWAVE[0.0][1]] + [100 * CAPWAVE[k][1] for k in g], "s--", label="frequency error, m = 4")
    ax[1].semilogx([10.0], [100 * CAPWAVE_M2[10.0][0]], "o", color="C3", label="damping error, m = 2")
    ax[1].set_xlabel("gamma dt (0 drawn at 0.01)")
    ax[1].set_ylabel("error of the capillary mode [%]")
    ax[1].set_title("(b) capillary wave, N = 32, x 1: more damping term, less damping")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    gd = [k for k in DROP if k > 0]
    ax[2].semilogx([0.01] + gd, [DROP[0.0]] + [DROP[k] for k in gd], "o-")
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].set_xlabel("gamma dt (0 drawn at 0.01)")
    ax[2].set_ylabel("|lambda| of the leading mode")
    ax[2].set_title("(c) static drop, density ratio 100, K = 1, 1.6 x: still unstable")
    ax[2].grid(alpha=0.3, which="both")

    ax[3].bar(list(QSHARE), list(QSHARE.values()), color=["C3", "C7"])
    ax[3].set_ylabel("share of velocity energy in Q's band")
    ax[3].set_title("(d) the unstable mode is not a grid-scale mode")
    ax[3].grid(alpha=0.3, axis="y")
    fig.savefig(f"{args.out}/split_hyperviscosity.png", dpi=100)
    print("wrote", f"{args.out}/split_hyperviscosity.png")


if __name__ == "__main__":
    main()

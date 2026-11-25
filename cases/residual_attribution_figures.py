"""Attribution of the remaining over-damping after the band correction (docs/notes/46): ``residual_attribution.png``.

Eigenvalues of one step's Jacobian restricted to the k = 2 pi family (:mod:`cases.capwave_jacobian`), capillary
wave, x 1 viscosity, exact linear curvature, band-corrected kinematics (:mod:`cases.velocity_extension`);
measured numbers copied here.

    python -m cases.residual_attribution_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: dt / dt0 -> damping error, N = 32: A (fd2 viscosity), C (CCD viscosity)
DT_A = {1: 0.0772, 1 / 2: 0.0636, 1 / 4: 0.0519, 1 / 8: 0.0435, 1 / 16: 0.0383, 1 / 32: 0.03539, 1 / 64: 0.03386}
DT_C = {1: 0.0443, 1 / 2: 0.03582, 1 / 4: 0.0280, 1 / 8: 0.02164, 1 / 16: 0.01722, 1 / 32: 0.01451, 1 / 64: 0.0130}
#: eigenvector overlap with dt/16 (A)
OVERLAP = {1: 0.99503, 1 / 2: 0.997947, 1 / 4: 0.99942, 1 / 8: 0.999917, 1 / 16: 1.0}
#: (N, dt / dt0) -> damping error of A, C (CCD viscosity), D (converged coupled iterations), E (both)
ABCDE = {(32, 1): (0.0772, 0.0443, 0.0761, 0.0433), (32, 1 / 4): (0.0519, 0.0280, 0.0518, 0.0279),
         (64, 1): (0.0555, 0.0297, 0.0545, 0.0287)}


def extrapolate(d):
    k = sorted(d)[:3]  # the three smallest steps
    a, b, c = (d[x] for x in k)  # a: smallest
    r = (a - b) / (b - c)
    return a + (a - b) * r / (1 - r)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    for d, lab, c in ((DT_A, "A: fd2 viscosity", "C0"), (DT_C, "C: CCD viscosity", "C1")):
        k = sorted(d)
        lim = extrapolate(d)
        ax[0].plot(k, [100 * d[x] for x in k], "o-", color=c, label=f"{lab} (dt -> 0: {100 * lim:.1f} %)")
        ax[0].axhline(100 * lim, color=c, ls=":", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("dt / dt0")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) N = 32, x 1: time step; eigenvector overlap >= 0.995")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    keys = list(ABCDE)
    labs = [f"N = {n}, dt/{int(round(1 / f))}" if f != 1 else f"N = {n}, dt" for n, f in keys]
    parts = []
    for k in keys:
        A, C, D, E = ABCDE[k]
        parts.append((C - A, D - A, E - C - D + A))
    x = np.arange(len(keys))
    for j, (name, col) in enumerate((("viscosity fd2 -> CCD", "C1"), ("splitting removed (coupled)", "C2"), ("interaction", "C3"))):
        ax[1].bar(x + (j - 1) * 0.27, [100 * p[j] for p in parts], 0.27, label=name, color=col)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labs, fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("change of the damping-rate error [% points]")
    ax[1].set_title("(b) counterfactual changes from A")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    limA, limC = extrapolate(DT_A), extrapolate(DT_C)
    A0 = DT_A[1]
    segs = [("time discretisation (fd2)", A0 - limA), ("viscosity fd2 -> CCD (dt -> 0)", limA - limC), ("remaining (CCD, dt -> 0)", limC)]
    left = 0.0
    for i, (name, w) in enumerate(segs):
        ax[2].barh([0], [100 * w], left=[100 * left], height=0.6, color=f"C{i + 4}", label=f"{name}: {100 * w:+.1f}")
        left += w
    ax[2].set_yticks([])
    ax[2].set_xlabel("damping-rate error [%] (A at dt: +7.7 %)")
    ax[2].set_title("(c) N = 32, x 1: what the +7.7 % is made of")
    ax[2].set_ylim(-0.6, 1.4)
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].grid(alpha=0.3, axis="x")
    fig.savefig(f"{args.out}/residual_attribution.png", dpi=100)
    print("wrote", f"{args.out}/residual_attribution.png", "limits", round(limA, 4), round(limC, 4))


if __name__ == "__main__":
    main()

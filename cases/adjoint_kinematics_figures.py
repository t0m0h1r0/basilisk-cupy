"""Energy consistency of the corrected CLS kinematics with the GFM force (docs/notes/45): ``adjoint_kinematics.png``.
Measured numbers (cases.adjoint_kinematics_measure, cases.capwave_jacobian, cases.step_jacobian) copied here.

    python -m cases.adjoint_kinematics_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: (case) -> (prediction with the area energy, prediction with the sharp energy, measured eigenvalue), exact curvature
PRED = {"N=32 transport": (-0.5382, -0.1970, -0.1663), "N=64 transport": (-0.3795, -0.0584, -0.0469),
        "N=32 band corr.": (-0.1021, -0.0217, 0.0772), "N=64 band corr.": (-0.0903, -0.0162, 0.0555)}
#: N = 32, area energy: |Delta F| / |F_GFM|: (gradient part, divergence-free part)
SPLIT = {"transport": (436.93, 0.281), "band correction": (436.93, 0.209)}
#: band correction, exact curvature, N = 32: dt fraction -> damping error
DT = {1.0: 0.0772, 0.5: 0.0636, 0.25: 0.0519}
#: static drop, leading |lambda| of one step's Jacobian, N = 32 / 64
DROP = {"N=32 transport": 0.99957, "N=32 band corr.": 0.99933, "N=64 transport": 0.99575, "N=64 band corr.": 0.99489}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.0), constrained_layout=True)
    labs = list(PRED)
    x = np.arange(len(labs))
    for k, (name, c) in enumerate((("prediction, area energy sigma A_h(psi)", "C7"), ("prediction, sharp energy sigma L(eta)", "C0"),
                                   ("measured (eigenvalue, dt)", "C3"))):
        ax[0].bar(x + (k - 1) * 0.27, [100 * PRED[l][k] for l in labs], 0.27, label=name, color=c)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("damping change [%]")
    ax[0].set_title("(a) damping explained by the work mismatch (GFM vs adjoint)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    names = list(SPLIT)
    ax[1].bar(np.arange(2) - 0.2, [SPLIT[n][0] for n in names], 0.4, label="gradient part (taken by the pressure)")
    ax[1].bar(np.arange(2) + 0.2, [SPLIT[n][1] for n in names], 0.4, label="divergence-free part")
    ax[1].set_yscale("log")
    ax[1].set_xticks(np.arange(2))
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("|Delta F| / |F_GFM|")
    ax[1].set_title("(b) Delta F = F_adj(area) - F_GFM, N = 32")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y", which="both")

    d = sorted(DT)
    ax[2].plot(d, [100 * DT[k] for k in d], "o-", label="band correction, exact curvature")
    ax[2].plot([0.0, 0.25], [100 * (2 * DT[0.25] - DT[0.5]), 100 * DT[0.25]], "k:", label="linear extrapolation")
    ax[2].axhline(3.82, color="C2", ls="--", lw=1, label="sharp kinematics (note 41), dt")
    ax[2].set_xlabel("dt / dt_0")
    ax[2].set_ylabel("damping-rate error [%]")
    ax[2].set_title("(c) the remaining over-damping: half from the time step")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    ax[3].bar(list(DROP), [1 - v for v in DROP.values()], color=["C0", "C1", "C0", "C1"])
    ax[3].set_ylabel("1 - |lambda| (decay per step of the slowest mode)")
    ax[3].set_title("(d) static drop: the slowest mode is not slower with the correction")
    ax[3].grid(alpha=0.3, axis="y")
    for t in ax[3].get_xticklabels():
        t.set_fontsize(8)
    fig.savefig(f"{args.out}/adjoint_kinematics.png", dpi=100)
    print("wrote", f"{args.out}/adjoint_kinematics.png")


if __name__ == "__main__":
    main()

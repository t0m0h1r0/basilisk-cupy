"""Band correction of the CLS kinematics (velocity extension) with the sharp GFM jump (docs/notes/44):
``velocity_extension.png``. Measured numbers copied from :mod:`cases.capwave_normal_mode` (time integration from
the normal mode, advection on, psi curvature), :mod:`cases.velocity_extension_checks` and
:mod:`cases.reversed_cls`.

    python -m cases.velocity_extension_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: (N, viscosity factor) -> delta/h, (damping, frequency) transport, (damping, frequency) band correction
WAVE = {(32, 0.5): (0.65, (-0.2783, -0.0405), (-0.0225, 0.0072)), (32, 1.0): (0.92, (-0.1635, -0.0337), (0.0344, 0.0052)),
        (32, 2.0): (1.30, (-0.0906, -0.0297), (0.0322, -0.0008)), (64, 0.5): (1.30, (-0.0743, None), (0.0565, 0.0076)),
        (64, 1.0): (1.83, (-0.0375, -0.0057), (0.0354, 0.0033)), (64, 2.0): (2.59, (-0.0203, -0.0064), (0.0168, 0.0003)),
        (128, 1.0): (3.66, (0.0051, -0.0010), (0.0201, 0.0003))}
SHARP_REFERENCE = (0.0382, -0.0105)  # sharp kinematics + sharp GFM, eigenvalue, N = 32, x 1 (note 41)
#: static drop max|u| after 10 / 50 / 200 / 400 steps
DROP = {"N = 32, ratio 1": ((2.85e-4, 1.45e-4, 6.07e-6, 1.78e-7), (2.81e-4, 1.53e-4, 1.81e-5, 2.07e-6)),
        "N = 64, ratio 1": ((4.44e-5, 3.57e-5, 6.66e-6, 6.78e-7), (4.44e-5, 4.02e-5, 1.73e-5, 6.25e-6)),
        "N = 32, ratio 100": ((4.82e-4, 7.79e-5, 1.89e-6, 8.83e-9), (4.78e-4, 1.10e-4, 6.74e-6, 1.34e-7)),
        "N = 64, ratio 100": ((6.18e-5, 4.05e-5, 7.28e-6, 4.41e-7), (6.17e-5, 4.17e-5, 1.53e-5, 4.78e-6))}
#: interface transport tests: (transport, band correction)
SHAPE = {"translation N=32, psi L1": (6.00e-3, 6.00e-3), "translation N=64, psi L1": (5.04e-3, 5.04e-3),
         "reversed vortex N=64, sharp L1": (3.056e-2, 2.775e-2), "reversed vortex N=128, sharp L1": (1.394e-2, 8.40e-3)}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)
    for j, (lab, mk) in enumerate((("CLS transport (default)", "o"), ("band correction (extension)", "s"))):
        for N, st in ((32, "-"), (64, "--"), (128, ":")):
            pts = sorted((v[0], v[1 + j][0]) for k, v in WAVE.items() if k[0] == N)
            ax[0].plot([p[0] for p in pts], [100 * p[1] for p in pts], mk + st, color=f"C{j}", label=f"{lab}, N = {N}")
    ax[0].axhline(100 * SHARP_REFERENCE[0], color="k", ls=":", lw=1, label="sharp reference (eigenvalue, N = 32)")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("delta / h")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) capillary wave: damping")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for j, (lab, mk) in enumerate((("CLS transport", "o-"), ("band correction", "s-"))):
        ns_ = [32, 64, 128]
        ax[1].plot(ns_, [100 * WAVE[(n, 1.0)][1 + j][1] for n in ns_], mk, label=lab)
        ax[1].plot(ns_, [100 * WAVE[(n, 1.0)][1 + j][0] for n in ns_], mk.replace("-", "--"), color=f"C{j}", label=f"{lab}: damping")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("N (x 1)")
    ax[1].set_ylabel("error [%]")
    ax[1].set_title("(b) frequency (solid) and damping (dashed) against N")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    steps = [10, 50, 200, 400]
    for i, (lab, (a, b)) in enumerate(DROP.items()):
        ax[2].semilogy(steps, a, "o-", color=f"C{i}", label=f"{lab}: transport")
        ax[2].semilogy(steps, b, "s--", color=f"C{i}", label=f"{lab}: band correction")
    ax[2].set_xlabel("steps")
    ax[2].set_ylabel("max |u|")
    ax[2].set_title("(c) static drop: the jump is unchanged, the decay is slower")
    ax[2].legend(fontsize=7, ncol=2)
    ax[2].grid(alpha=0.3, which="both")

    labs = list(SHAPE)
    y = np.arange(len(labs))
    ax[3].barh(y - 0.2, [SHAPE[k][0] for k in labs], 0.4, label="CLS transport")
    ax[3].barh(y + 0.2, [SHAPE[k][1] for k in labs], 0.4, label="band correction")
    ax[3].set_yticks(y)
    ax[3].set_yticklabels(labs, fontsize=8)
    ax[3].invert_yaxis()
    ax[3].set_xlabel("error")
    ax[3].set_title("(d) interface transport tests")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3, axis="x")
    fig.savefig(f"{args.out}/velocity_extension.png", dpi=100)
    print("wrote", f"{args.out}/velocity_extension.png")


if __name__ == "__main__":
    main()

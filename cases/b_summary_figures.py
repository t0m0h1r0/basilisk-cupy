"""Summary of B (capillary-wave damping, docs/notes/49): ``b_summary.png``. All numbers are measurements
from notes 38-48, copied here.

    python -m cases.b_summary_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: N = 32, x 1 (delta/h = 0.92): damping-rate error of each approach (note); None: unstable
APPROACHES = [("default: CLS transport + GFM (38)", -0.1635), ("height curvature (40)", -0.1136),
              ("CCD viscosity (31)", -0.1656), ("CCD + FCCD faces (32)", -0.1321),
              ("jump-corrected viscosity (40)", None), ("high-wavenumber damping, gamma dt = 100 (42)", -0.1888),
              ("adjoint surface tension psi grad mu (43)", -0.6394), ("y-refined map, alpha = 8 (39)", -0.0217),
              ("band velocity correction (44)", 0.0344), ("exact cell projection + CN, fd2 (48)", 0.0320),
              ("exact cell projection + CN, CCD (48)", 0.0111), ("MAC (48)", 0.1288)]
#: damping error against N, x 1
RESOLUTION = {"default (CLS transport)": {32: -0.1635, 64: -0.0375, 128: 0.0051},
              "band velocity correction": {32: 0.0344, 64: 0.0354, 128: 0.0201}}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    names = [a for a, _ in APPROACHES]
    vals = [v for _, v in APPROACHES]
    y = np.arange(len(names))
    colors = ["C7" if v is None else ("C2" if abs(v) < 0.05 else "C3") for v in vals]
    ax[0].barh(y, [0 if v is None else 100 * v for v in vals], color=colors)
    for k, v in enumerate(vals):
        if v is not None and v < -0.3:  # long bar: label inside
            ax[0].annotate(f"{100 * v:+.1f} %", (100 * v, k), fontsize=8, color="white", xytext=(6, -3), textcoords="offset points")
            continue
        ax[0].annotate("unstable" if v is None else f"{100 * v:+.1f} %", (0 if v is None else 100 * v, k), fontsize=8,
                       xytext=(4 if (v or 0) >= 0 else -40, -3), textcoords="offset points")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].axvspan(-5, 5, color="C2", alpha=0.08)
    ax[0].set_xlabel("damping-rate error [%] (N = 32, x 1; green: within 5 %)")
    ax[0].set_title("(a) what was tried (notes 38-48)")
    ax[0].grid(alpha=0.3, axis="x")

    for lab, d in RESOLUTION.items():
        ns_ = sorted(d)
        ax[1].plot(ns_, [100 * d[n] for n in ns_], "o-", label=lab)
    ax[1].axhspan(-1, 1, color="C2", alpha=0.1, label="within 1 %")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("N (16 / 32 / 64 points per wavelength)")
    ax[1].set_ylabel("damping-rate error [%]")
    ax[1].set_title("(b) resolution against method")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.savefig(f"{args.out}/b_summary.png", dpi=100)
    print("wrote", f"{args.out}/b_summary.png")


if __name__ == "__main__":
    main()

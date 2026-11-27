"""Figure for docs/notes/90 (python -m cases.galerkin2d_shiftfix first).

    python -m cases.galerkin2d_shiftfix_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    R = [r for r in json.load(open("docs/figures/galerkin2d_shiftfix.json")) if r["test"] == "amp"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, ratio in enumerate((1e6, 1e3)):
        for angle, ls in ((0, "-"), (45, "--")):
            for lsq, col, lab in (("shift", "C0", "shift (corrected)"), ("constrained", "C3", "notes 88-89 form")):
                rows = sorted([r for r in R if r["args"]["ratio"] == ratio and r["args"]["angle"] == angle
                               and r["args"]["lsq"] == lsq], key=lambda r: r["args"]["zeta"])
                ax[k].loglog([r["args"]["zeta"] for r in rows], [r["result"]["Lambda"] for r in rows], ls,
                             marker="o", color=col, label=f"{lab}, {angle} deg")
        ax[k].set_xlabel("phase-mass ratio zeta = r c / (1 - c) of the cut cell")
        ax[k].set_ylabel("Lambda = max sum_j |phi_ij| (5 x 5 cells around)")
        ax[k].set_title(f"density ratio {ratio:.0e}, N = 24, isolated interface")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin2d_shiftfix.png", dpi=130)


if __name__ == "__main__":
    main()

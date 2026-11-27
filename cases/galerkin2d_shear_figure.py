"""Figure for docs/notes/90 (python -m cases.galerkin2d_shear first): errors against N, and the deformed disc.

    python -m cases.galerkin2d_shear_figure
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cases.galerkin2d_shear import ShearedDisc  # noqa: E402


def main() -> None:
    R = [r for r in json.load(open("docs/figures/galerkin2d_shear.json")) if r["test"] == "run"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    th = np.linspace(0, 2 * np.pi, 400)
    for t, ls in ((0.0, ":"), (0.125, "--"), (0.25, "-")):
        g = ShearedDisc(S=1.0 * 0.5 / math.pi * math.sin(math.pi * t / 0.5))
        P = g.curve(th, (g.cx, g.cy))
        ax[0].plot(P[:, 0], P[:, 1], ls, color="k", label=f"t = {t}")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_title("interface: u = (sin 2 pi y cos(pi t / T), 0), T = 0.5")
    ax[0].legend(fontsize=7)
    for i, t in enumerate(("0.125", "0.25")):
        for (hz, tr), col, lab in ((("exact", "galerkin"), "C1", "Galerkin, defining-function hinge"),
                                   (("quadratic", "galerkin"), "C2", "Galerkin, quadratic hinge"),
                                   (("tangent", "galerkin"), "C3", "Galerkin, tangent-line hinge"),
                                   (("exact", "geo"), "C0", "direct remap, defining-function hinge")):
            rows = sorted([r for r in R if r["args"]["hinge"] == hz and r["args"].get("transport", "galerkin") == tr],
                          key=lambda r: r["args"]["N"])
            if rows:
                ax[i + 1].loglog([r["args"]["N"] for r in rows], [r["result"]["errors"][t] for r in rows],
                                 marker="o", color=col, label=lab)
        ax[i + 1].loglog([24, 64], [1e-3, 1e-3 * (24 / 64) ** 4], ":", color="gray", label="4th order")
        ax[i + 1].loglog([24, 64], [1e-2, 1e-2 * (24 / 64) ** 2], "--", color="gray", label="2nd order")
        ax[i + 1].set_xlabel("N")
        ax[i + 1].set_ylabel(f"max error at t = {t}")
        ax[i + 1].set_title(f"sheared disc, density ratio 1000, t = {t}")
        ax[i + 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin2d_shear.png", dpi=130)


if __name__ == "__main__":
    main()

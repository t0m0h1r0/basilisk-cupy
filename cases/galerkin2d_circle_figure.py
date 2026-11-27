"""Figure for docs/notes/89 (python -m cases.galerkin2d_circle first).

    python -m cases.galerkin2d_circle_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    R = [r for r in json.load(open("docs/figures/galerkin2d_circle.json")) if r["test"] == "run"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, t in enumerate(("0.05", "0.1")):
        for (hz, tr), col, lab in ((("exact", "galerkin"), "C1", "Galerkin, exact-distance hinge"),
                                   (("quadratic", "galerkin"), "C2", "Galerkin, quadratic hinge"),
                                   (("tangent", "galerkin"), "C3", "Galerkin, tangent-line hinge"),
                                   (("exact", "geo"), "C0", "direct remap, exact-distance hinge")):
            rows = sorted([r for r in R if r["args"]["hinge"] == hz and r["args"].get("transport", "galerkin") == tr],
                          key=lambda r: r["args"]["N"])
            if rows:
                ax[i].loglog([r["args"]["N"] for r in rows], [r["result"]["errors"][t] for r in rows], marker="o",
                             color=col, label=lab)
        ax[i].loglog([24, 64], [1e-3, 1e-3 * (24 / 64) ** 4], ":", color="gray", label="4th order")
        ax[i].loglog([24, 64], [1e-2, 1e-2 * (24 / 64) ** 2], "--", color="gray", label="2nd order")
        ax[i].set_xlabel("N")
        ax[i].set_ylabel(f"max error at t = {t}")
        ax[i].set_title(f"circle R0 = 1/4, density ratio 1000, t = {t}")
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin2d_circle.png", dpi=130)


if __name__ == "__main__":
    main()

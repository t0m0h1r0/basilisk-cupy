"""Figure for docs/notes/87 (python -m cases.galerkin2d and the N = 64 / shear / angle runs first).

    python -m cases.galerkin2d_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    d = json.load(open("docs/figures/galerkin2d.json"))
    R = json.load(open("docs/figures/galerkin2d_more.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for tr, col in (("geo", "C0"), ("galerkin", "C1"), ("corr", "C2")):
        Ns = [16, 24, 32, 48, 64]
        e = [d[f"run {tr} {N}"]["errors"]["0.1"] for N in Ns[:-1]]
        e += [r["result"]["errors"]["0.1"] for r in R if r["test"] == "phases64" and r["transport"] == tr]
        ax[0].loglog(Ns, e, marker="o", color=col, label={"geo": "direct remap", "galerkin": "Galerkin",
                                                           "corr": "adjoint-corrected"}[tr])
    ax[0].loglog([16, 64], [3e-2, 3e-2 / 4 ** 4], ":", color="gray", label="4th order")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("max error at t = 0.1")
    ax[0].set_title("2-D, band (1, 1), density ratio 1000, per-phase q")
    ax[0].legend(fontsize=7)
    for (t, tr), col, lab in ((("shear", "geo"), "C0", "shear, remap"), (("shear", "galerkin"), "C1", "shear, Galerkin"),
                              (("angle12", "galerkin"), "C3", "band (1, 2), Galerkin")):
        rows = sorted([r for r in R if r["test"] == t and r["transport"] == tr], key=lambda r: r["args"]["N"])
        ax[1].loglog([r["args"]["N"] for r in rows], [r["result"]["errors"]["0.1"] for r in rows], marker="o",
                     color=col, label=lab)
    ax[1].loglog([24, 64], [1e-3, 1e-3 * (24 / 64) ** 4], ":", color="gray", label="4th order")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("max error at t = 0.1")
    ax[1].set_title("exact shear profile f(z) and another angle")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin2d.png", dpi=130)


if __name__ == "__main__":
    main()

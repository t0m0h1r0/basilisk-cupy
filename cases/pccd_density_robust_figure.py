"""Figure for docs/notes/95 (python -m cases.pccd_gram and cases.pccd_floquet first).

    python -m cases.pccd_density_robust_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAB = {"none": "no jump separation (note 92)", "onesided": "one-sided jump estimate", "joint": "joint estimate, cut-off J",
       "joint_light": "joint, J on the light side", "local": "P1-a local CCD correction", "split": "P1-b phase-split + shift",
       "split_opt": "P1-b + optimised closure"}


def main() -> None:
    G = json.load(open("docs/figures/pccd_gram.json"))
    F = json.load(open("docs/figures/pccd_floquet.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for m in ("none", "onesided", "joint", "joint_light", "local", "split"):
        rows = sorted([r for r in G if r["N"] == 32 and r["mode"] == m and "lmax" in r], key=lambda r: r["ratio"])
        ax[0].loglog([r["ratio"] for r in rows], [r["lmax"] for r in rows], marker="o", label=LAB[m])
    ax[0].set_xlabel("density ratio")
    ax[0].set_ylabel("lambda_max(M^-1/2 G M^-1/2)")
    ax[0].set_title("Gram matrix, N = 32 (lambda_min = 1 for all)")
    ax[0].legend(fontsize=7)
    for k, tr in enumerate(("remap", "galerkin")):
        for m in ("none", "joint", "local", "split", "split_opt"):
            rows = sorted([r for r in F if r["mode"] == m and r["transport"] == tr and "sv_mass" in r],
                          key=lambda r: r["ratio"])
            by = {}
            for r in rows:
                by[r["ratio"]] = max(by.get(r["ratio"], 0.0), r["sv_mass"])
            ax[k + 1].loglog(list(by), list(by.values()), marker="o", label=LAB[m])
        ax[k + 1].set_xlabel("density ratio")
        ax[k + 1].set_ylabel("max over c of ||P_rel|| (mass norm)")
        ax[k + 1].set_title(f"{tr}: growth over one Floquet period (C = 1/4, N = 32)")
        ax[k + 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/pccd_density_robust.png", dpi=130)


if __name__ == "__main__":
    main()

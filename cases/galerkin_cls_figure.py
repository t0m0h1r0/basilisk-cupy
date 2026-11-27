"""Figure for docs/notes/82 from docs/figures/galerkin_cls.json (python -m cases.galerkin_cls first).

    python -m cases.galerkin_cls_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RATIOS = (1.0, 10.0, 100.0, 1e3, 1e6)


def main() -> None:
    R = json.load(open("docs/figures/galerkin_cls.json"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    eps = [1e-2, 1e-4, 1e-6, 1e-8]
    for r in R:
        if r["test"] != "P0-1":
            continue
        a = r["args"]
        lab = "piecewise (note 81)" if a.get("mode") == "True" else f"ls, alpha = {a['alpha']}"
        ax[0, 0].loglog(eps, [x["jump"] for x in r["result"]["across"]], marker="o", label=lab)
    ax[0, 0].set_xlabel("eps (interface moves by 2 eps h across a cell boundary)")
    ax[0, 0].set_ylabel("||G^-1 C - I||_2")
    ax[0, 0].set_title("P0-1 continuity (N = 48, density ratio 1000)")
    ax[0, 0].legend(fontsize=7)
    for test, ls in (("P0-3", "-"), ("P0-3e0", ":")):
        for al in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows = {r["args"]["ratio"]: r["result"] for r in R
                    if r["test"] == test and r["args"]["alpha"] == al and r["args"].get("mode", "ls") == "ls"}
            if not rows:
                continue
            tag = "window +1" if test == "P0-3" else "window +0"
            ax[0, 1].loglog(RATIOS, [rows[q]["cond"] for q in RATIOS], ls, marker="o", label=f"alpha {al}, {tag}")
            ax[1, 0].loglog(RATIOS, [rows[q]["inf"] for q in RATIOS], ls, marker="o", label=f"alpha {al}, {tag}")
    pw = {r["args"]["ratio"]: r["result"] for r in R if r["test"] == "P0-3" and r["args"].get("mode") == "True"}
    for k, axx in (("cond", ax[0, 1]), ("inf", ax[1, 0])):
        axx.loglog(RATIOS, [pw[q][k] for q in RATIOS], "k--", marker="s", label="piecewise (note 81)")
        axx.set_xlabel("density ratio")
        axx.legend(fontsize=6)
    ax[0, 1].set_ylabel("max cond(M^-1/2 G M^-1/2)")
    ax[0, 1].set_title("P0-3 conditioning (CFL 0.37, 12 positions)")
    ax[1, 0].set_ylabel("max ||T||_inf (one step)")
    ax[1, 0].set_title("P0-3 velocity amplification")
    for ratio in (1.0, 1e3):
        rows = sorted([r for r in R if r["test"] == "P0-6w" and r["args"]["ratio"] == ratio],
                      key=lambda r: r["args"]["cfl"])
        ax[1, 1].loglog([r["args"]["cfl"] for r in rows], [r["result"]["per_time"] for r in rows], marker="o",
                        label=f"density ratio {ratio:g}")
    ax[1, 1].set_xlabel("CFL")
    ax[1, 1].set_ylabel("worst energy loss per unit time")
    ax[1, 1].set_title("P0-6 worst-case dissipation (alpha 0.75, N = 48)")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin_cls.png", dpi=130)


if __name__ == "__main__":
    main()

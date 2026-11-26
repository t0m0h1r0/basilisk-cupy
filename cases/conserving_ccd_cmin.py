"""The drop carried by a uniform flow (``cases.convection_stability.translate``) with the momentum-conserving VOF and
the CCD gradient of its concentrations, for the phase fraction from which a cell joins a CCD run
(``TwoPhase(conserving_ccd_cmin=...)``, docs/notes/75): the mean over all steps of max|u - U| against N, density
ratios 10 and 1000. Writes docs/figures/conserving_ccd_cmin.png and the numbers to conserving_ccd_cmin.json.

    python -m cases.conserving_ccd_cmin [--jobs 28] [--replot]    # --replot: redraw from the JSON
"""

from __future__ import annotations

import argparse
import json

JSON = "docs/figures/conserving_ccd_cmin.json"

VARIANTS = {"centered-vof": "Basilisk VOF (BCG)", "centered-conserving": "conserving, centred gradient",
            "centered-conserving-ccd": "conserving + CCD, c >= 0.5",
            "centered-conserving-ccd-c99": "conserving + CCD, c >= 0.99",
            "centered-conserving-ccd2": "conserving + CCD parabola, c >= 0.5",
            "centered-conserving-ccd2-c99": "conserving + CCD parabola, c >= 0.99"}
NS = (32, 64, 128)
RATIOS = (10.0, 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    from cases.convection_stability import translate
    from cases.parallel import pmap

    if args.replot:
        with open(JSON) as fh:
            rows = json.load(fh)
    else:
        items = [dict(variant=v, n=n, ratio=r) for n in NS[::-1] for r in RATIOS for v in VARIANTS]
        rows = []

        def show(k, it, r):
            rows.append(dict(variant=it["variant"], n=it["n"], ratio=it["ratio"], du_mean=r["du_mean"],
                             du_max=r["du_max"], du_end=r["du"]))
            print(f"{it['variant']:30s} N {it['n']:3d} ratio {it['ratio']:6g}: mean max|u - U| {r['du_mean']:.3e}"
                  f" (max over t {r['du_max']:.3e})", flush=True)

        pmap(translate, items, args.jobs, on_result=show)
        with open(JSON, "w") as fh:
            json.dump(rows, fh, indent=1)
    res = {(d["variant"], d["n"], d["ratio"]): d["du_mean"] for d in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    styles = ["k-o", "C0-s", "C3--^", "C3-^", "C2--v", "C2-v"]
    for ax, ratio in zip(axes, RATIOS):
        for (v, label), st in zip(VARIANTS.items(), styles):
            ax.loglog(NS, [res[(v, n, ratio)] for n in NS], st, label=label)
        ax.set_title(f"translating drop, density ratio {ratio:g}")
        ax.set_xlabel("N")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("mean over steps of max|u - U|")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/figures/conserving_ccd_cmin.png", dpi=120)


if __name__ == "__main__":
    main()

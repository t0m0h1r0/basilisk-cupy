"""Spurious currents of a drop at rest and of the same drop carried by a uniform flow
(``cases.convection_stability.translate``, R = 0.15, sigma = 0.01, mu = 1e-4, periodic [0, 1]^2, t = 1,
docs/notes/76): the mean over all steps of max|u - U| and the mean relative curvature error |kappa R| - 1 in the
interfacial cells (max and RMS over the cells, sampled every 4 steps), for Basilisk's VOF and the momentum-conserving
VOF, tension.h's height-function curvature and the CCD column curvature (``curvature_method="ccd"``), N = 32 / 64 /
128, density ratios 10 and 1000. Writes docs/figures/moving_spurious.png and the numbers to moving_spurious.json.

    python -m cases.moving_spurious [--jobs 28] [--replot]
"""

from __future__ import annotations

import argparse
import json

JSON = "docs/figures/moving_spurious.json"
VARIANTS = {"centered-vof": "VOF, heights", "centered-vof+kccd": "VOF, CCD curvature",
            "centered-conserving": "conserving, heights", "centered-conserving+kccd": "conserving, CCD curvature"}
NS = (32, 64, 128)
RATIOS = (10.0, 1000.0)
SPEEDS = (0.0, 1.0)


def job(variant, n, ratio, speed):
    from cases.convection_stability import translate

    r = translate(variant, n, ratio, speed=speed, kappa_every=4)
    return {k: r[k] for k in ("failed", "du_mean", "du_max", "kappa_max", "kappa_rms", "steps")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    if args.replot:
        with open(JSON) as fh:
            rows = json.load(fh)
    else:
        from cases.parallel import pmap

        items = [dict(variant=v, n=n, ratio=r, speed=s) for n in NS[::-1] for r in RATIOS for s in SPEEDS
                 for v in VARIANTS]
        rows = []

        def show(k, it, r):
            rows.append(dict(it, **r))
            print(f"{it['variant']:26s} U {it['speed']:g} N {it['n']:3d} ratio {it['ratio']:6g}: "
                  + ("FAILED " if r["failed"] else "") + f"mean max|u - U| {r['du_mean']:.3e} (max {r['du_max']:.3e})"
                  f" kappa err max {r['kappa_max']:.3e} rms {r['kappa_rms']:.3e}", flush=True)

        pmap(job, items, args.jobs, on_result=show)
        with open(JSON, "w") as fh:
            json.dump(rows, fh, indent=1)
    res = {(d["variant"], d["n"], d["ratio"], d["speed"]): d for d in rows}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    colors = {"centered-vof": "k", "centered-vof+kccd": "C3", "centered-conserving": "C0",
              "centered-conserving+kccd": "C1"}
    for col, ratio in enumerate(RATIOS):
        for v, label in VARIANTS.items():
            for speed, ls in ((1.0, "-"), (0.0, ":")):
                kw = dict(color=colors[v], ls=ls, marker="o", label=f"{label}, {'moving' if speed else 'at rest'}")
                axes[0, col].loglog(NS, [res[(v, n, ratio, speed)]["du_mean"] for n in NS], **kw)
                axes[1, col].loglog(NS, [res[(v, n, ratio, speed)]["kappa_rms"] for n in NS], **kw)
        axes[0, col].set_title(f"density ratio {ratio:g}")
        for ax in axes[:, col]:
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(True, which="both", alpha=0.3)
        axes[1, col].set_xlabel("N")
    axes[0, 0].set_ylabel("mean over steps of max|u - U|")
    axes[1, 0].set_ylabel("RMS of |kappa R| - 1 (interfacial cells)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/moving_spurious.png", dpi=120)


if __name__ == "__main__":
    main()

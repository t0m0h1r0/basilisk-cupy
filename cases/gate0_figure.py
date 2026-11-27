"""docs/figures/gate0_records.png (docs/notes/103): C and D against N (worst over 3 interface offsets) per property,
and the boost run's difference to a CFL 1/128 run against the CFL.

    python -m cases.gate0_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def _style(ax, ticks=None, labels=None):
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)


def main() -> None:
    g = json.load(open("docs/figures/gate0_protocol.json"))
    b = json.load(open("docs/figures/coupled_boost.json"))
    from cases.gate0_protocol import NS
    cases = (("mu", 1000.0, "constant mu, 1e3"), ("water-air", 1000.0, "water-air, 1e3"),
             ("water-air", 1.0, "mu jump only, 1"), ("nu", 1000.0, "constant nu, 1e3"))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), dpi=150)
    for ax, (k, r, title) in zip(axes[:4], cases):
        for m, c in (("C", BLUE), ("D", ORANGE)):
            o = next(o for o in g["orders"] if o["test"] == "stokes" and o["kind"] == k and o["ratio"] == r
                     and o["method"] == m and o["norm"] == "max")
            ax.loglog(NS, o["errors"], "-o", color=c, lw=2, ms=4, label=f"{m} ({o['worst'][0]:.2f})")
        e0 = next(o for o in g["orders"] if o["test"] == "stokes" and o["kind"] == k and o["ratio"] == r
                  and o["method"] == "C" and o["norm"] == "max")["errors"][0]
        ax.loglog(NS, [e0 * (NS[0] / n) ** 4 for n in NS], "--", color=INK2, lw=1, label="4th order")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("N", fontsize=8, color=INK2)
        _style(ax, (32, 64, 128, 256), ("32", "64", "128", "256"))
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("max error at t = 0.01 (worst of 3 offsets)", fontsize=8, color=INK2)
    ax = axes[4]
    cols = {("mu", 1.0): BLUE, ("water-air", 1.0): ORANGE, ("mu", 1000.0): AQUA, ("water-air", 1000.0): YELLOW,
            ("nu", 1000.0): MAGENTA}
    for t in b["time"]:
        key = (t["kind"], t["ratio"])
        if key not in cols:
            continue
        qs = sorted(int(q) for q in t["diff"])
        ax.loglog([1 / q for q in qs], [t["diff"][str(q)] for q in qs], "-o", color=cols[key], lw=2, ms=4,
                  label=f"{t['kind']}, {t['ratio']:g}")
    ax.loglog([1 / 2, 1 / 32], [2e-6, 2e-6 / 16], "--", color=INK2, lw=1, label="1st order")
    ax.loglog([1 / 2, 1 / 32], [2e-6, 2e-6 / 256], ":", color=INK2, lw=1, label="2nd order")
    ax.set_title("boost: difference to CFL 1/128 (N = 64)", fontsize=9, color=INK)
    ax.set_xlabel("CFL", fontsize=8, color=INK2)
    _style(ax, (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2), ("1/32", "1/16", "1/8", "1/4", "1/2"))
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("docs/figures/gate0_records.png")


if __name__ == "__main__":
    main()

"""docs/figures/short_runs.png (docs/notes/106): light-phase error and Gram lambda_max against the pure light cells,
and the 2-D D stiffness against the 1-D one.

    python -m cases.gate1b_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = ("#257de0", "#f47230", "#00b894")


def main() -> None:
    r = json.load(open("docs/figures/short_runs.json"))
    v = json.load(open("docs/figures/viscous_2d_twophase.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=150)
    for c, ratio in zip(COLS, (1.0, 1e3, 1e6)):
        rows = sorted((o for o in r["table"] if o["kind"] == "blend" and o["ratio"] == ratio and o["f1"] == 0.4),
                      key=lambda o: o["nL"])
        n = [o["nL"] for o in rows]
        axes[0].semilogy(n, [o["kink"]["light"] for o in rows], "-o", color=c, lw=2, ms=4, label=f"light, ratio {ratio:g}")
        axes[0].semilogy(n, [o["kink"]["heavy"] for o in rows], "--", color=c, lw=1, label=f"heavy, ratio {ratio:g}")
        axes[1].plot(n, [o["lmax"] for o in rows], "-o", color=c, lw=2, ms=4, label=f"ratio {ratio:g}")
    base = sorted((o for o in r["table"] if o["kind"] == "base" and o["ratio"] == 1e3 and o["f1"] == 0.4), key=lambda o: o["nL"])
    axes[1].plot([o["nL"] for o in base if "lmax" in o], [o["lmax"] for o in base if "lmax" in o], "x", color="#666666",
                 ms=7, label="note 97 (n = 1-4 undefined)")
    axes[0].set_title("SplitShort: max error (kinked u, N = 32)", fontsize=9)
    axes[1].set_title("Gram lambda_max (M-normalised)", fontsize=9)
    for a in axes[:2]:
        a.set_xlabel("pure light cells", fontsize=8)
    rows = [o for o in v["rows"] if o["kind"] == "nu"]
    lab = [f"{o['ratio']:g}, c={o['c']:g}" for o in rows]
    x = range(len(rows))
    axes[2].semilogy(x, [o["D_1d"] for o in rows], "o", color=COLS[0], label="D, 1-D")
    axes[2].semilogy(x, [o["D"] for o in rows], "s", color=COLS[1], label="D, 2-D (max over theta_x)")
    axes[2].semilogy(x, [o["C"] for o in rows], "^", color=COLS[2], label="C, 2-D")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(lab, rotation=60, fontsize=6)
    axes[2].set_title("constant nu: stiffness x h^2 / nu_max (N = 32)", fontsize=9)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
        a.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("docs/figures/short_runs.png")


if __name__ == "__main__":
    main()

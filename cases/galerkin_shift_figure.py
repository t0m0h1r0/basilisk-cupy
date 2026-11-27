"""Figure for docs/notes/85 (python -m cases.galerkin_amp and python -m cases.galerkin_shift first).

    python -m cases.galerkin_shift_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    A = json.load(open("docs/figures/galerkin_amp.json"))
    S = json.load(open("docs/figures/galerkin_shift.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, (which, cfl) in enumerate((("A0", 0.37), ("B0", 0.05))):
        a = ax[k]
        for fit, ls in (("constrained", "-"), ("shift", "--")):
            rows = sorted([r for r in A if r["args"]["ratio"] == 1e6 and r["args"]["which"] == which
                           and r["args"]["cfl"] == cfl and r["args"].get("fit", "constrained") == fit],
                          key=lambda r: r["args"]["zeta"])
            z = [r["args"]["zeta"] for r in rows]
            for op, col in (("PC", "k"), ("geo", "C0"), ("G", "C1"), ("corr", "C2")):
                a.loglog(z, [r["result"][op]["inf"] for r in rows], ls, color=col, label=f"{op}, {fit}")
        a.set_xlabel("zeta = heavy-sliver mass / light mass in the cell")
        a.set_ylabel("one-step ||T||_inf")
        a.set_title(f"density ratio 1e6, {which} ({'sliver leaves' if which == 'A0' else 'sliver enters'}), "
                    f"CFL {cfl}")
        a.legend(fontsize=6, ncol=2)
    rows = [r for r in S if r["test"] == "lost" and r["args"]["fit"] == "shift"]
    labels, vals = [], []
    for r in rows:
        a_ = r["args"]
        labels.append(f"{a_['ratio']:g}/{a_['which']}/C{a_['cfl']}")
        vals.append([float(r["result"]["oracle"][str(k)]) for k in (0, 1, 2, 4)])
    for j, k in enumerate((0, 1, 2, 4)):
        ax[2].bar([i + 0.2 * j for i in range(len(rows))], [v[j] for v in vals], width=0.2, label=f"{k} modes")
    ax[2].set_xticks([i + 0.3 for i in range(len(rows))])
    ax[2].set_xticklabels(labels, rotation=30, fontsize=7)
    ax[2].set_ylabel("worst one-step energy loss")
    ax[2].set_title("P0-13 oracle enrichment (shift, N = 48)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin_shift.png", dpi=130)


if __name__ == "__main__":
    main()

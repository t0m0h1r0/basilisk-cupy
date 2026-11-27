"""Figure for docs/notes/97 (python -m cases.pccd_floquet --general first; the zeta scans are recomputed here).

    python -m cases.pccd_partial_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cases.pccd_closure import closure, evaluate_blend  # noqa: E402
from cases.pccd_local import explicit_closure  # noqa: E402


def main() -> None:
    Z = tuple(10.0 ** k for k in range(-6, 7))
    candB = closure(np.array(json.load(open("docs/figures/pccd_closure.json"))["runs"][2]["theta"]))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for (name, cl, mm), col in zip((("explicit closure, extrapolated mixed cell (note 95)", explicit_closure(), "extend"),
                                    ("candidate B closure, extrapolated", candB, "extend"),
                                    ("explicit closure, partial-cell form (note 97)", explicit_closure(), "partial")),
                                   ("C3", "C1", "C0")):
        rows = evaluate_blend(cl, ratio=1e6, zetas=Z, mixed_mode=mm)
        ax[0].loglog(Z, [r[2] for r in rows], marker="o", color=col, label=name)
        ax[1].semilogx(Z, [r[1] for r in rows], marker="o", color=col, label=name)
    ax[0].set_ylabel("lambda_max(M^-1/2 G M^-1/2)")
    ax[1].set_ylabel("Lambda = max sum_j |phi_j|")
    for a in ax[:2]:
        a.set_xlabel("phase-mass ratio zeta of the cut cell (density ratio 1e6, N = 32)")
        a.legend(fontsize=7)
    F = json.load(open("docs/figures/pccd_floquet_general.json"))
    cfls = ("1/16", "1/8", "1/4", "1/3", "1/2", "3/4", "9/10")
    x = [eval(c) for c in cfls]
    for tr, col in (("remap", "C0"), ("galerkin", "C1")):
        for ratio, ls in ((1.0, ":"), (1e6, "-")):
            y = [max(r["sv_one_period"] for r in F if r["transport"] == tr and r["cfl"] == c and r["ratio"] == ratio)
                 for c in cfls]
            ax[2].semilogy(x, y, ls, marker="o", color=col, label=f"{tr}, density ratio {ratio:g}")
    ax[2].set_xlabel("CFL")
    ax[2].set_ylabel("max over c of ||P_rel|| (mass norm, one period)")
    ax[2].set_title("Floquet, partial-cell phase split, c = 1e-6 .. 0.5")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/pccd_partial.png", dpi=130)


if __name__ == "__main__":
    main()

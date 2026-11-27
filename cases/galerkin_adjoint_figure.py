"""Figure for docs/notes/84 (python -m cases.galerkin_adjoint first).

    python -m cases.galerkin_adjoint_figure
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cases.galerkin_cls import step10  # noqa: E402
from cases.galerkin_fourier import dissipation  # noqa: E402
from cases.galerkin_transport import A0  # noqa: E402


def _inf(args):
    ratio, d = args
    n = 64
    h = 1.0 / n
    k = int(math.floor(A0 / h)) + 1
    return step10(n, 4, k * h - d * h - A0, 0.37 * h, 0.75, ratio, 0.0)["inf"]


def main() -> None:
    R = json.load(open("docs/figures/galerkin_adjoint.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    cfls = np.logspace(-3, np.log10(0.5), 16)
    for kind, col in (("poly", "C0"), ("spline", "C1")):
        for beta, ls in ((0.0, "-"), (0.5, "--")):
            ax[0].loglog(cfls, [max(dissipation(kind, C, math.pi / 2, beta)["d"], 1e-16) for C in cfls], ls,
                         color=col, label=f"{kind}, beta {beta}")
    ax[0].set_xlabel("CFL")
    ax[0].set_ylabel("d = (1 - |T|^2)/CFL, wavelength 4h")
    ax[0].set_title("P0-10a: constant density (Bloch)")
    ax[0].legend(fontsize=7)
    for ratio, mk in ((1.0, "o"), (1e3, "s")):
        for beta, col in ((0.0, "C0"), (0.25, "C2"), (0.5, "C3")):
            rows = sorted([r for r in R if r["test"] == "loss_cfl" and r["args"]["ratio"] == ratio
                           and r["args"]["beta"] == beta], key=lambda r: r["args"]["cfl"])
            ax[1].loglog([r["args"]["cfl"] for r in rows], [r["result"]["loss_per_time"] for r in rows],
                         marker=mk, color=col, ls="-" if ratio == 1.0 else ":",
                         label=f"ratio {ratio:g}, beta {beta}")
    ax[1].set_xlabel("CFL")
    ax[1].set_ylabel("worst energy loss per unit time")
    ax[1].set_title("P0-10b: note-82 reconstruction (N = 48)")
    ax[1].legend(fontsize=6)
    ds = [10.0 ** -e for e in range(1, 8)]
    with ProcessPoolExecutor(14) as ex:
        vals = list(ex.map(_inf, [(r, d) for r in (1e3, 1e6) for d in ds]))
    for j, ratio in enumerate((1e3, 1e6)):
        ax[2].loglog(ds, vals[j * len(ds):(j + 1) * len(ds)], marker="o", label=f"density ratio {ratio:g}")
    ax[2].set_xlabel("1 - theta (interface just before a cell boundary, light cell with a heavy sliver)")
    ax[2].set_ylabel("one-step ||T||_inf (alpha 0.75, window +1)")
    ax[2].set_title("max-norm amplification peak")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin_adjoint.png", dpi=130)


if __name__ == "__main__":
    main()

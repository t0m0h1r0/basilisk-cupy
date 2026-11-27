"""Figure for docs/notes/83 from docs/figures/galerkin_fourier.json (python -m cases.galerkin_fourier first).

    python -m cases.galerkin_fourier_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cases.galerkin_fourier import dissipation  # noqa: E402


def main() -> None:
    d = json.load(open("docs/figures/galerkin_fourier.json"))
    th = np.array(d["curve"]["theta"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for kind, col in (("poly", "C0"), ("spline", "C1")):
        for C, ls in (("0.001", "-"), ("0.37", "--")):
            ax[0].semilogy(th, np.maximum(d["curve"][kind][C], 1e-18), ls, color=col,
                           label=f"{'cell-wise quartic' if kind == 'poly' else 'continuous quartic B-spline'}, CFL {C}")
    ax[0].semilogy(th, th ** 10 / 900, ":", color="gray", label="theta^10 / 900")
    ax[0].set_xlabel("theta = k h")
    ax[0].set_ylabel("gamma h / U  (energy decay per unit time)")
    ax[0].set_ylim(1e-16, 1e2)
    ax[0].set_title("P0-8a / P0-9a: dissipation, constant density")
    ax[0].legend(fontsize=7)
    cfls = np.logspace(-3, np.log10(0.5), 20)
    for kind, col in (("poly", "C0"), ("spline", "C1")):
        for name, t, ls in (("4h", np.pi / 2, "-"), ("2h", np.pi, "--")):
            ax[1].loglog(cfls, [dissipation(kind, C, t)["d"] for C in cfls], ls, color=col,
                         label=f"{kind}, wavelength {name}")
    ax[1].set_xlabel("CFL")
    ax[1].set_ylabel("d = (1 - |T|^2) / CFL")
    ax[1].set_title("dissipation per unit time vs CFL")
    ax[1].legend(fontsize=7)
    nu = np.array([v["nu_num"] for v in d["jump"].values()])
    lam = [16, 8, 6, 4, 3, 2]
    for Re in (1, 10, 100, 1000):
        ax[2].loglog(lam, nu * Re, marker="o", label=f"Re_h = U h / nu = {Re}")
    ax[2].axhline(1.0, color="gray", ls=":")
    ax[2].invert_xaxis()
    ax[2].set_xlabel("wavelength (cells)")
    ax[2].set_ylabel("nu_num / nu  (cell-wise quartic, small CFL)")
    ax[2].set_title("numerical vs physical viscous decay")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin_fourier.png", dpi=130)


if __name__ == "__main__":
    main()

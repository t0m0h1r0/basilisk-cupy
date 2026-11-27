"""Figure for docs/notes/91 (python -m cases.pccd first).

    python -m cases.pccd_figure
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cases import galerkin_fourier as gf  # noqa: E402
from cases.pccd import symbols  # noqa: E402
from cases.pccd_compare import bcg_symbol, fuccd_lambda  # noqa: E402


def main() -> None:
    C = 0.37
    th = np.linspace(0.05, math.pi - 0.05, 120)
    thf, lam = fuccd_lambda(256)
    curves = {"BCG (Fromm)": [bcg_symbol(t, C) for t in th],
              "FUCCD4 (exact time)": list(np.exp(np.interp(th, thf[:129], lam[:129].real) * C
                                                 + 1j * np.interp(th, thf[:129], lam[:129].imag) * C)),
              "LS quartic Galerkin (notes 80-81)": [gf.symbol("poly", C, t) for t in th],
              "B-spline Galerkin": [gf.symbol("spline", C, t) for t in th]}
    S = [symbols(t, C) for t in th]
    curves.update({"PCCD remap": [r["remap"] for r in S], "PCCD Galerkin": [r["galerkin"] for r in S],
                   "PCCD adjoint-corrected": [r["corr"] for r in S]})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for (k, v), col in zip(curves.items(), ("C7", "C5", "C4", "C8", "C0", "C1", "C2")):
        v = np.array(v)
        ax[0].semilogy(th / math.pi, np.maximum((1 - abs(v) ** 2) / C, 1e-17), color=col, label=k)
        ax[1].semilogy(th / math.pi, np.maximum(abs(np.angle(v * np.exp(1j * th * C))) / (th * C), 1e-17),
                       color=col, label=k)
    ax[0].set_title(f"dissipation d = (1 - |T|^2)/C, C = {C}")
    ax[1].set_title("relative phase error |arg(T e^{i theta C})| / (theta C)")
    for a in ax[:2]:
        a.set_xlabel("theta / pi  (wavelength 2/(theta/pi) cells)")
        a.set_ylim(1e-14, 2)
    ax[0].legend(fontsize=7)
    R = json.load(open("docs/figures/pccd.json"))["runs"]
    for tr, col in (("remap", "C0"), ("galerkin", "C1"), ("corr", "C2")):
        rows = sorted([r for r in R if r["transport"] == tr and r["profile"] == "narrow"], key=lambda r: r["N"])
        ax[2].loglog([r["N"] for r in rows], [r["errors"]["1.0"] for r in rows], marker="o", color=col,
                     label=f"PCCD {tr}")
    ax[2].loglog([32, 256], [3e-3, 3e-3 / 8 ** 5], ":", color="gray", label="5th order")
    ax[2].loglog([32, 128], [1e-4, 1e-4 / 4 ** 9], "--", color="gray", label="9th order")
    ax[2].set_xlabel("N")
    ax[2].set_ylabel("max error of the averages at t = 1")
    ax[2].set_title("Gaussian sigma = 0.05, one period, C = 0.37")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/pccd.png", dpi=130)


if __name__ == "__main__":
    main()

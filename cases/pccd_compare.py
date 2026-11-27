"""docs/notes/91: PCCD against the library's BCG and FUCCD_4 and against notes 80-88's reconstructions (Fourier).

* BCG (bcg.h, 1-D, the unlimited centred gradient: Fromm's scheme on cell averages):
  u_f = u_i + (1 - C)/4 (u_{i+1} - u_{i-1}),  u^{n+1}_i = u_i - C (u_f(i) - u_f(i-1)).
* FUCCD_4 (sigma 1e-3, the library's operator on point values; its matrix on a periodic 1-D grid): the semi-discrete
  symbol lambda(theta) h, integrated exactly in time (the spatial part alone) or with SSP-RK3.
* notes 80-88: the cell-wise quartic ("poly") and the quartic B-spline Galerkin (cases.galerkin_fourier).

    python -m cases.pccd_compare
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases import galerkin_fourier as gf
from cases.pccd import symbols


def fuccd_lambda(n=64, sigma=1.0e-3):
    """lambda h (n,) of FUCCD_4 for the modes theta_k = 2 pi k / n (unit speed, periodic)."""
    from basilisk_cupy.boundary import BoundaryConditions
    from basilisk_cupy.fields import CellField, FaceField, VectorField
    from basilisk_cupy.grid.cartesian import CartesianGrid
    from basilisk_cupy.terms.convection.fuccd import FUCCDConvection

    grid = CartesianGrid((n,), length=(1.0,))
    op = FUCCDConvection(r=4, sigma=sigma)
    bc = BoundaryConditions.periodic(1)
    uf = FaceField(grid, [np.ones(n + 1)])
    col = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        u = VectorField([CellField.from_interior(grid, e, bc)])
        col.append(np.asarray(op.tendency(u, uf)[0]))
    L = np.array(col).T * grid.h[0]                 # lambda h, column j = response to e_j
    k = np.arange(n)
    th = 2 * np.pi * k / n
    # circulant: lambda(theta) = sum_j L[0, j] e^{i j theta}
    lam = np.array([np.sum(L[0] * np.exp(1j * (np.arange(n)) * t)) for t in th])
    return th, lam


def rk3(z):
    return 1 + z + z * z / 2 + z ** 3 / 6


def bcg_symbol(theta, C):
    e = np.exp(-1j * theta)
    d = (1 - e) + (1 - C) / 4 * (np.exp(1j * theta) - e - 1 + e * e)
    return 1 - C * d


def table(theta=math.pi / 2, Cs=(0.01, 0.1, 0.37, 0.5, 0.9)):
    th, lam = fuccd_lambda(64)
    lam_t = lam[np.argmin(abs(th - theta))]
    out = {}
    for C in Cs:
        ex = np.exp(-1j * theta * C)
        r = symbols(theta, C)
        T = {"BCG (Fromm)": bcg_symbol(theta, C), "FUCCD4 exact time": np.exp(lam_t * C),
             "FUCCD4 RK3": rk3(lam_t * C), "LS quartic Galerkin (notes 80-81)": gf.symbol("poly", C, theta),
             "B-spline Galerkin": gf.symbol("spline", C, theta), "PCCD remap": r["remap"],
             "PCCD Galerkin": r["galerkin"], "PCCD corrected": r["corr"]}
        out[C] = {k: {"d": float((1 - abs(v) ** 2) / C), "phase": float(np.angle(v / ex) / (theta * C))}
                  for k, v in T.items()}
    return out


def main() -> None:
    res = {"theta=pi/2": table()}
    res["theta=pi/4"] = table(math.pi / 4)
    for C, row in res["theta=pi/2"].items():
        print("C", C)
        for k, v in row.items():
            print(f"   {k:36s} d {v['d']:.3e}  relative phase error {v['phase']:+.3e}")
    json.dump(res, open("docs/figures/pccd_compare.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

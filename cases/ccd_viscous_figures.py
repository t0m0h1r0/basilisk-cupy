"""CCD viscosity in CenteredNS (docs/notes/31): ``ccd_viscous.png``.

Left: Taylor-Green-like decay (kx = 2 pi, ky = 4 pi, periodic, nu = 0.01, Stokes, dt = 2e-4, t = 0.04;
reference: exact in space, backward Euler in time) for fd2 / CCD viscosity with fd2 / FCCD face operators.
Right: capillary-wave damping-rate error against delta / h for CLS + GFM with fd2 and CCD viscosity.

    python -m cases.ccd_viscous_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS

NU, DT, T = 0.01, 2e-4, 0.04
VARIANTS = (("fd2 viscosity, fd2 faces", False, False), ("CCD viscosity, fd2 faces", True, False),
            ("fd2 viscosity, FCCD faces", False, True), ("CCD viscosity, FCCD faces", True, True))


def taylor_green(n: int, ccd: bool, fccd: bool) -> float:
    g = CartesianGrid((n, n), length=(1.0, 1.0))
    x, y = g.meshgrid()
    ns = CenteredNS(g, periodic=(0, 1), mu=NU, DT=DT, viscous=CCDViscosity() if ccd else None,
                    ops=FCCDFaceComplex() if fccd else None, tolerance=1e-10, stokes=True)
    kx, ky = 2 * math.pi, 4 * math.pi
    u0 = np.sin(kx * x) * np.cos(ky * y) * ky
    ns.u[0].interior = u0
    ns.u[1].interior = -np.cos(kx * x) * np.sin(ky * y) * kx
    ns.u.fill_ghosts()
    ns.start()
    while ns.clock.t < T - 1e-12:
        ns.step()
    dec = (1 + NU * (kx * kx + ky * ky) * DT) ** (-ns.clock.i)
    return float(np.abs(ns.u[0].interior - u0 * dec).max()) / ky


def capwave_point(n: int, f: float, ccd: bool):
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay, run
    from cases.capwave_modes import period_decay

    a, t, _ = run("cls", n, mu=cw.MU * f, **({"viscous": CCDViscosity()} if ccd else {}))
    p, d = decay(a, t)
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, cw.MU * f)
    dh = math.sqrt(2 * cw.MU * f / math.sqrt((2 * math.pi) ** 3 / 2)) / (2.0 / n)
    return dh, math.log(d) / math.log(dt_) - 1


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ns_ = (16, 32, 64)
    for label, ccd, fccd in VARIANTS:
        ax[0].loglog(ns_, [taylor_green(n, ccd, fccd) for n in ns_], "o-", label=label)
    ax[0].loglog(ns_, [2e-2 * (16 / n) ** 2 for n in ns_], "k:", lw=0.8, label="2nd order")
    ax[0].loglog(ns_, [1e-3 * (16 / n) ** 4 for n in ns_], "k--", lw=0.8, label="4th order")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("max |u - u_exact| / ky")
    ax[0].set_title("Taylor-Green (kx != ky), CenteredNS, Stokes")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for ccd, style in ((False, "o"), (True, "s")):
        pts = [capwave_point(n, f, ccd) for f in (0.5, 1.0) for n in (32, 64)]
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], style, ms=8,
                   label="CCD viscosity" if ccd else "fd2 viscosity")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("delta / h")
    ax[1].set_ylabel("damping-rate error")
    ax[1].set_title("capillary wave, CLS + GFM (fd2 faces)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.savefig(f"{args.out}/ccd_viscous.png", dpi=100)
    print("wrote", f"{args.out}/ccd_viscous.png")


if __name__ == "__main__":
    main()

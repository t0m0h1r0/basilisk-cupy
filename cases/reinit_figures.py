"""Figure of docs/notes/54: curvature along the interface after one re-initialisation, and the capillary wave with
re-initialisation every 10 steps (``docs/figures/reinit_closures.png``).

    python -m cases.reinit_figures [--out docs/figures] [--jobs N]
"""

from __future__ import annotations

import argparse

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface


def curvature_along(distance: str | None, method: str = "xi_sdf", n: int = 32):
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    c = CLSInterface.from_level_set(g, 0.25 - np.hypot(x, y), BoundaryConditions())
    if distance:
        c.reinitialize(method=method, distance=distance)
    k = c.curvature()
    q = c.psi.interior
    band = (q > 0.3) & (q < 0.7)
    return np.arctan2(y[band], x[band]), k[band]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.capillary_wave import PROSPERETTI
    from cases.parallel import pmap
    from cases.two_closure_reinit import capwave

    runs = pmap(capwave, [dict(method=m) for m in (None, "xi_sdf", "xi_sdf_vw")], args.jobs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for dist, lab, m in ((None, "exact profile", "k."), ("points", "after one ξ-SDF (points)", "C1o"),
                         ("segments", "after one ξ-SDF (segments)", "C2s")):
        th, k = curvature_along(dist)
        ax.plot(th, k, m, ms=3, label=lab)
    ax.axhline(4.0, color="gray", lw=0.8)
    ax.set_xlabel("angle along the interface")
    ax.set_ylabel("curvature in the band 0.3 < ψ < 0.7")
    ax.set_title("static drop N = 32: ψ's CCD curvature")
    ax.legend(fontsize=8)
    ax = axes[1]
    P = np.array(PROSPERETTI)
    ax.plot(P[:, 0], P[:, 1], "k-", lw=1.0, label="Prosperetti")
    for r, lab in zip(runs, ("no re-initialisation", "ξ-SDF every 10 steps", "two closures (λ_Γ, ε_eff) every 10")):
        a = np.array(r["amplitudes"])
        ax.plot(P[:len(a), 0], a, lw=1.0, label=f"{lab}: RMS {r['error']:.3f}")
    ax.set_xlabel("t ω₀")
    ax.set_ylabel("amplitude")
    ax.set_title("capillary wave, 16 points per wavelength (TwoPhaseIMEX)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/reinit_closures.png", dpi=150)
    print("wrote", f"{args.out}/reinit_closures.png")


if __name__ == "__main__":
    main()

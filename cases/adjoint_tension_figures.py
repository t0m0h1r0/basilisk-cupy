"""Adjoint (energy-consistent) surface tension against sharp coupling (docs/notes/43): ``adjoint_tension.png``.

Numbers measured with :mod:`cases.capwave_jacobian` (eigenvalue, exact linear curvature unless stated) and
:mod:`cases.capwave_normal_mode` (time integration for the psi grad mu form, whose reduced Jacobian is not
closed: residual 0.3); copied here.

    python -m cases.adjoint_tension_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: N = 32, x 1: (label, damping error, frequency error, coupling width)
VARIANTS = [("sharp kinematics + sharp GFM (adjoint pair, width 0)", 0.0382, -0.0105),
            ("CLS kinematics (weights) + sharp GFM", -0.1682, -0.0540),
            ("CLS kinematics + GFM spread with the same weights (adjoint pair, width ~h)", -0.2594, -0.0857),
            ("CLS transport + GFM, plain divergence (time integration)", -0.2066, -0.0504),
            ("CLS transport + psi grad mu (adjoint, width ~eps, time integration)", -0.6394, -0.3284)]
#: psi grad mu, time integration: N -> (eps k, frequency error, damping error), x 1
DIFFUSE = {32: (1.5 * (2.0 / 32) * 2 * np.pi, -0.3284, -0.6394), 64: (1.5 * (2.0 / 64) * 2 * np.pi, -0.1761, -0.3311)}
#: GFM (plain, time integration) N -> (frequency, damping)
SHARP = {32: (-0.0504, -0.2066), 64: (-0.0150, -0.0617)}
#: CLS weights kinematics + spread GFM against + sharp GFM: delta/h -> (sharp force, spread force)
SPREAD = {0.65: (-0.2894, -0.3928), 0.92: (-0.1682, -0.2594), 1.30: (-0.0946, -0.1581), 1.83: (-0.0502, -0.1015)}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    names = [v[0] for v in VARIANTS]
    y = np.arange(len(names))
    ax[0].barh(y - 0.2, [100 * v[1] for v in VARIANTS], 0.4, label="damping error")
    ax[0].barh(y + 0.2, [100 * v[2] for v in VARIANTS], 0.4, label="frequency error")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlabel("error [%]")
    ax[0].set_title("(a) N = 32, x 1: the wider the coupling, the larger the deficit")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="x")

    d = sorted(SPREAD)
    ax[1].plot(d, [100 * SPREAD[k][0] for k in d], "o-", label="CLS kinematics + sharp GFM")
    ax[1].plot(d, [100 * SPREAD[k][1] for k in d], "s--", label="CLS kinematics + GFM spread with the same weights")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("delta / h")
    ax[1].set_ylabel("damping-rate error [%]")
    ax[1].set_title("(b) making the pair adjoint at width h does not help")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for lab, data, mk in (("psi grad mu (width eps)", DIFFUSE, "o-"),):
        ns_ = sorted(data)
        ax[2].plot([data[n][0] for n in ns_], [100 * data[n][1] for n in ns_], mk, label=f"{lab}: frequency")
        ax[2].plot([data[n][0] for n in ns_], [100 * data[n][2] for n in ns_], mk.replace("-", "--"), label=f"{lab}: damping")
    ns_ = sorted(SHARP)
    ax[2].plot([DIFFUSE[n][0] for n in ns_], [100 * SHARP[n][0] for n in ns_], "s-", label="sharp GFM: frequency")
    ax[2].plot([DIFFUSE[n][0] for n in ns_], [100 * SHARP[n][1] for n in ns_], "s--", label="sharp GFM: damping")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("eps k (N = 64, 32)")
    ax[2].set_ylabel("error [%]")
    ax[2].set_title("(c) energy-consistent but diffuse force: errors ~ eps k")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.savefig(f"{args.out}/adjoint_tension.png", dpi=100)
    print("wrote", f"{args.out}/adjoint_tension.png")


if __name__ == "__main__":
    main()

"""Discrete dispersion relation of the capillary wave and where its damping deficit comes from (docs/notes/41):
``capwave_theory.png``.

Numbers measured with :mod:`cases.capwave_jacobian` (eigenvalue of one step's Jacobian restricted to the k = 2 pi
family) and :mod:`cases.capwave_kinematic_theory` are copied here; panels (b) and (c) are recomputed.

(a) damping error against delta/h: the scheme with psi / height / exact linear curvature, the scheme with the
    interface moved by the face velocity at the interface ("sharp kinematics") or by the CLS transport's own
    weights, and the modified dispersion relation (continuous Navier--Stokes, discrete kinematics);
(b) the weights w_f with which the CLS transport moves the psi = 1/2 crossing (N = 32; the same in units of h
    at N = 64);
(c) the exact v(y) (cos kx part, N = 32, x 1) at the faces: the interface moves with sum w_f v(y_f), not v(0);
(d) N = 32, x 1: what changes the damping error and what does not.

    python -m cases.capwave_theory_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: (N, viscosity factor) -> delta/h, damping error: psi, heights, exact curvature, sharp kinematics,
#: CLS weights (all with exact curvature except the first two), modified dispersion relation
DATA = {(32, 0.5): (0.65, -0.2871, -0.2371, -0.2882, -0.0134, -0.2894, -0.2683),
        (32, 1.0): (0.92, -0.1643, -0.1179, -0.1663, 0.0382, -0.1682, -0.1704),
        (32, 2.0): (1.30, -0.0893, -0.0560, -0.0920, 0.0271, -0.0946, -0.0961),
        (64, 0.5): (1.30, -0.0836, -0.0351, -0.0866, 0.0700, None, -0.1142),
        (64, 1.0): (1.83, -0.0435, -0.0133, -0.0469, 0.0334, -0.0502, -0.0632),
        (64, 2.0): (2.59, -0.0255, -0.0091, -0.0289, 0.0088, -0.0323, -0.0334)}
#: N = 32, x 1 (delta/h = 0.92): damping error of the scheme's eigenvalue
VARIANTS = [("psi curvature (default)", -0.1643), ("height curvature", -0.1179), ("exact linear curvature", -0.1663),
            ("exact curvature, dt / 2", -0.1684), ("exact curvature, dt / 4", -0.1697),
            ("exact curvature, CCD viscosity", -0.1680), ("exact curvature, CCD + FCCD faces", -0.1343),
            ("sharp kinematics", 0.0382), ("CLS weights as kinematics", -0.1682),
            ("theory: continuous NS + CLS weights", -0.1704)]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases import capillary_wave as cw
    from cases.capwave_jacobian import CapwaveMap, kinematic_weights
    from cases.capwave_normal_mode import K, mode

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)
    labels = ["psi curvature (default)", "height curvature", "exact curvature", "exact curvature, sharp kinematics",
              "exact curvature, CLS weights as kinematics", "theory: continuous NS + CLS weights"]
    marks = ["o-", "s-", "^-", "D-", "v--", "k:"]
    for j, (lab, mk) in enumerate(zip(labels, marks)):
        for N, alpha in ((32, 1.0), (64, 0.5)):
            pts = sorted((v[0], v[1 + j]) for k, v in DATA.items() if k[0] == N and v[1 + j] is not None)
            ax[0].plot([p[0] for p in pts], [100 * p[1] for p in pts], mk, color=None if mk != "k:" else "k",
                       alpha=alpha, label=f"{lab}" if N == 32 else None, mfc="none" if N == 64 else None)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("delta / h (open marks: N = 64)")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) eigenvalue of the scheme against delta/h")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    cm = CapwaveMap(32, 1.0, curvature="exact")
    yf, w = kinematic_weights(cm.tp)
    h = 2.0 / 32
    sel = np.abs(yf) < 6 * h
    ax[1].bar(yf[sel] / h, w[sel], width=0.6)
    ax[1].set_xlabel("face position y_f / h (interface at 0)")
    ax[1].set_ylabel("weight w_f")
    ax[1].set_title(f"(b) CLS kinematic weights: sum {w.sum():.3f}, second moment {np.sum(w * yf ** 2) / h ** 2:.3f} h^2")
    ax[1].grid(alpha=0.3, axis="y")

    s, F = mode(K, 1.0, 1.0, cw.MU)
    yy = np.linspace(-0.25, 0.25, 2001)
    V = F(np.zeros_like(yy), yy)[1]
    Vf = F(np.zeros_like(yf), yf)[1]
    for part, name, c in ((np.real, "Re", "C0"), (np.imag, "Im", "C1")):
        ax[2].plot(yy, part(V), color=c, label=f"{name} v(y)")
        ax[2].plot(yf[sel], part(Vf[sel]), "o", color=c)
        ax[2].axhline(part(np.sum(w * Vf)), color=c, ls="--", lw=1, label=f"{name} sum w_f v(y_f)")
        ax[2].axhline(part(F(np.zeros(1), np.zeros(1))[1][0]), color=c, ls=":", lw=1, label=f"{name} v(0)")
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("v, cos kx part (complex amplitude)")
    ax[2].set_title("(c) N = 32, x 1: the layer's cusp in v at the faces")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)

    names = [n for n, _ in VARIANTS]
    vals = [100 * v for _, v in VARIANTS]
    ax[3].barh(np.arange(len(names)), vals, color=["C3" if "kinematics" in n and "sharp" in n else "C0" for n in names])
    ax[3].set_yticks(np.arange(len(names)))
    ax[3].set_yticklabels(names, fontsize=8)
    ax[3].invert_yaxis()
    ax[3].axvline(0, color="k", lw=0.5)
    ax[3].set_xlabel("damping-rate error [%]")
    ax[3].set_title("(d) N = 32, x 1 (delta/h = 0.92)")
    ax[3].grid(alpha=0.3, axis="x")
    fig.savefig(f"{args.out}/capwave_theory.png", dpi=100)
    print("wrote", f"{args.out}/capwave_theory.png")


if __name__ == "__main__":
    main()

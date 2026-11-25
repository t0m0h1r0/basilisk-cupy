"""Why the capillary wave is under-damped on coarse grids (docs/notes/38): ``capwave_layer_diag.png``.

All numbers were measured with :mod:`cases.capwave_normal_mode` (the wave started on the exact normal mode,
the damping and frequency fitted over 1.5 periods) and are copied here; panel (c) is recomputed.

(a) damping error against delta / h for the operator combinations, and the one-parameter model
    ``gamma_h / gamma = 1 / sqrt(1 + c (h / delta)^2)`` (a layer broadened by ``c h^2``) fitted to fd2;
(b) what does not change it (N = 32, viscosity x 1): time step, CLS width, operators, jump form; and what
    does: the smeared force (CSF) instead of the GFM jump;
(c) rms vorticity profile (``sin kx`` part, second half of the run) against the exact mode, N = 32, x 1;
(d) damping ratio against the enstrophy ratio: of the run, and of the exact mode sampled on the grid
    (centred differences of the cell values).

    python -m cases.capwave_layer_diag_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

#: (N, viscosity factor) -> delta / h, damping error of fd2 / ccd / ccd + fccd (smooth jump)
DAMPING = {(32, 0.5): (0.65, -0.2783, -0.2626, -0.2113), (32, 1.0): (0.92, -0.1635, -0.1656, -0.1321),
           (32, 2.0): (1.30, -0.0906, -0.0988, -0.0772), (64, 0.5): (1.30, -0.0743, None, None),
           (64, 1.0): (1.83, -0.0375, -0.0468, -0.0435), (64, 2.0): (2.59, -0.0203, -0.0265, -0.0233),
           (128, 1.0): (3.66, 0.0051, None, None)}
#: N = 32, viscosity x 1 (delta / h = 0.92): damping error
INVARIANCE = [("fd2 (reference)", -0.1635), ("dt / 2", -0.1644), ("dt / 4", -0.1655), ("eps = 1.0 h", -0.1601),
              ("eps = 2.0 h", -0.1644), ("eps = 3.0 h", -0.1649), ("ccd viscosity", -0.1656),
              ("ccd + fccd, smooth", -0.1321), ("fd2, smooth jump", -0.1635), ("CSF, eps = 1.5 h", -0.6485),
              ("CSF, eps = 3.0 h", -0.9426)]
#: delta / h -> (gamma_h / gamma of the run, enstrophy of the run / exact, exact sampled on the grid / exact)
ENSTROPHY = {0.65: (0.7217, 0.644, 0.716), 0.92: (0.8365, 0.779, 0.846), 1.30: (0.9094, 0.871, 0.921),
             1.83: (0.9625, 0.936, 0.958), 2.59: (0.9797, 0.969, 0.979)}


def model_c() -> float:
    """Least squares ``c`` of ``1 / sqrt(1 + c (h / delta)^2)`` against the fd2 damping ratios."""
    x = np.array([(1.0 / d[0]) ** 2 for d in DAMPING.values()])
    r = np.array([1.0 + d[1] for d in DAMPING.values()])
    y = 1.0 / r ** 2 - 1.0
    return float(np.sum(x * y) / np.sum(x * x))


def main() -> None:
    import cmath

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases import capillary_wave as cw
    from cases.capwave_normal_mode import K, mode, run

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)

    c = model_c()
    for j, (lab, mk) in enumerate((("fd2", "o"), ("ccd viscosity", "s"), ("ccd + fccd, smooth jump", "^"))):
        pts = sorted((d[0], d[1 + j]) for d in DAMPING.values() if d[1 + j] is not None)
        ax[0].plot([p[0] for p in pts], [100 * p[1] for p in pts], mk + "-", label=lab)
    dh = np.linspace(0.6, 3.8, 200)
    ax[0].plot(dh, 100 * (1.0 / np.sqrt(1.0 + c / dh ** 2) - 1.0), "k:", label=f"1/sqrt(1 + c (h/delta)^2) - 1, c = {c:.3f}")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("delta / h, delta = sqrt(2 nu / omega_0)")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) normal-mode start: error against delta / h")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    names = [n for n, _ in INVARIANCE]
    vals = [100 * v for _, v in INVARIANCE]
    colors = ["C0"] * 9 + ["C3", "C3"]
    ax[1].barh(np.arange(len(names)), vals, color=colors)
    ax[1].set_yticks(np.arange(len(names)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("damping-rate error [%]")
    ax[1].set_title("(b) N = 32, x 1 (delta / h = 0.92): what changes it")
    ax[1].grid(alpha=0.3, axis="x")

    r = run(32, 1.0, periods=1.5, vorticity=True)
    s, sh = r["s"], r["s_h"]
    T = 2 * math.pi / s.imag
    rows = [(t, w) for t, w in r["vorticity"] if t > 0.5 * T]
    y = r["y"]
    sim = np.sqrt(np.mean([(w / math.exp(sh.real * t)) ** 2 for t, w in rows], axis=0))
    _, F = mode(K, 1.0, 1.0, cw.MU)
    yf = np.linspace(-0.3, 0.3, 3001)
    d = 1e-6

    def omega(yy, t):
        x0 = np.full_like(yy, 0.25)
        vx = (F(x0 + d, yy)[1] - F(x0 - d, yy)[1]) / (2 * d)
        uy = (F(x0, yy + d)[0] - F(x0, yy - d)[0]) / (2 * d)
        return ((vx - uy) * cmath.exp(s * t)).real / math.exp(s.real * t)

    ex = np.sqrt(np.mean([omega(yf, t) ** 2 for t, _ in rows], axis=0))
    m = np.abs(y) < 0.3
    ax[2].plot(yf, ex, "k-", label="exact mode")
    ax[2].plot(y[m], sim[m], "o-", label=f"run (fd2), damping error {100 * r['damping_err']:.1f} %")
    for yc in y[m]:
        ax[2].axvline(yc, color="0.85", lw=0.5, zorder=0)
    ax[2].set_xlabel("y (grey: cell centres)")
    ax[2].set_ylabel("rms of the sin kx part of the vorticity")
    ax[2].set_title("(c) vorticity layer, N = 32, x 1")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    dhs = sorted(ENSTROPHY)
    ax[3].plot([ENSTROPHY[k][2] for k in dhs], [ENSTROPHY[k][0] for k in dhs], "o-", label="exact mode sampled on the grid")
    ax[3].plot([ENSTROPHY[k][1] for k in dhs], [ENSTROPHY[k][0] for k in dhs], "s--", label="the run")
    for k in dhs:
        ax[3].annotate(f"{k}", (ENSTROPHY[k][2], ENSTROPHY[k][0]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax[3].plot([0.6, 1.0], [0.6, 1.0], "k:", lw=1, label="equal")
    ax[3].set_xlabel("enstrophy / exact enstrophy (centred differences)")
    ax[3].set_ylabel("gamma_h / gamma")
    ax[3].set_title("(d) damping ratio against enstrophy (labels: delta / h)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3)
    fig.savefig(f"{args.out}/capwave_layer_diag.png", dpi=100)
    print("wrote", f"{args.out}/capwave_layer_diag.png", "c =", round(c, 4))


if __name__ == "__main__":
    main()

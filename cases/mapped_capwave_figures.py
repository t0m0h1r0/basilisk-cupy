"""Capillary wave and static drop on grids refined in y about the interface (docs/notes/39): ``mapped_cls.png``.

Measured with :mod:`cases.capwave_normal_mode` (normal-mode start, Stokes, plain CLS divergence, 4th-order
faces, ``FD2Viscosity`` on the mapped grids) and :mod:`cases.mapped_static_drop`; copied here. Panel (c) is
recomputed. Walls are CPU seconds for 1.5 periods, 4 to 5 runs in parallel.

(a) damping error against delta / h_min: uniform N^2 and 32 x 32 cells with the y axis refined alpha times about
    y = 0 (``MonitorMapping(1 + (alpha - 1) exp(-((s - 1/2)/0.08)^2))``);
(b) wall time against the damping error (mapped: the capillary step of the uniform 32^2 grid, lines along y);
(c) the cell height h_y(y) of the maps;
(d) static drop: max |u| after 10 / 50 / 200 steps.

    python -m cases.mapped_capwave_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: uniform N^2 (same transport as the mapped runs): N -> (delta / h, damping error, wall s); viscosity x 1
UNIFORM = {32: (0.92, -0.2062, 5.0), 64: (1.83, -0.0613, 15.7), 128: (3.66, -0.0037, 58.0)}
UNIFORM_HALF = {32: (0.65, -0.3309)}  # viscosity x 0.5
#: 32 x 32, alpha -> (delta / h_min, damping error with the tension.h step of h_min, wall s;
#:                    damping error with the step of the uniform grid, wall s with lines along y only)
MAPPED = {2: (1.59, -0.0947, 209, -0.0952, 37.6), 4: (2.56, -0.0450, 312, -0.0464, 43.3),
          8: (3.67, -0.0217, 494, -0.0234, 51.1)}
MAPPED_HALF = {4: (1.81, -0.0728), 8: (2.59, -0.0338)}
#: static drop (R = 0.25, mu = 0.1, equal densities): label -> (max|u| at 10, 50, 200 steps, jump (exact 4))
DROP = {"uniform 32^2": (3.53e-4, 1.12e-4, 4.75e-6, 4.0101), "uniform 64^2": (5.48e-5, 3.00e-5, 5.66e-6, 4.0030),
        "32^2, alpha 4, eps 1.5 h_min": (2.53e-2, 2.55e-2, 2.63e-2, 4.1109),
        "32^2, alpha 4, eps 1.5 h_max": (3.54e-4, 6.88e-4, 1.21e-3, 4.0104),
        "48^2, alpha 4, eps 1.5 h_max": (1.36e-4, 2.52e-4, 2.62e-4, 4.0042)}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from basilisk_cupy import CartesianGrid
    from cases.mapped_static_drop import band

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)

    ax[0].plot([v[0] for v in UNIFORM.values()], [100 * v[1] for v in UNIFORM.values()], "o-", label="uniform N^2 (32, 64, 128), x 1")
    ax[0].plot([v[0] for v in UNIFORM_HALF.values()], [100 * v[1] for v in UNIFORM_HALF.values()], "o", mfc="none", color="C0", label="uniform 32^2, x 0.5")
    ax[0].plot([UNIFORM[32][0]] + [v[0] for v in MAPPED.values()], [100 * UNIFORM[32][1]] + [100 * v[1] for v in MAPPED.values()],
               "s-", color="C1", label="32 x 32, refined alpha = 1, 2, 4, 8, x 1")
    ax[0].plot([UNIFORM_HALF[32][0]] + [v[0] for v in MAPPED_HALF.values()], [100 * UNIFORM_HALF[32][1]] + [100 * v[1] for v in MAPPED_HALF.values()],
               "s--", color="C1", mfc="none", label="32 x 32, alpha = 1, 4, 8, x 0.5")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("delta / h_min (normal to the interface)")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) error against the resolution of the layer")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for n, (dh, e, w) in UNIFORM.items():
        ax[1].loglog(w, 100 * abs(e), "o", color="C0")
        ax[1].annotate(f"uniform {n}^2", (w, 100 * abs(e)), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for a, v in MAPPED.items():
        ax[1].loglog(v[4], 100 * abs(v[3]), "s", color="C1")
        ax[1].annotate(f"32^2, alpha {a}", (v[4], 100 * abs(v[3])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("wall time for 1.5 periods [s] (CPU)")
    ax[1].set_ylabel("|damping-rate error| [%]")
    ax[1].set_title("(b) cost against accuracy")
    ax[1].grid(alpha=0.3, which="both")

    for a in (1, 2, 4, 8):
        m = band(a, 0.5)
        g = CartesianGrid((32, 32), length=(2.0, 2.0), origin=(0.0, -1.0), mapping=None if m is None else (None, m))
        f = np.asarray(g.faces(1))
        ax[2].plot(0.5 * (f[1:] + f[:-1]), np.diff(f), ".-", label=f"alpha = {a}")
    ax[2].axhline(np.sqrt(2 * 0.0182571749236 / np.sqrt((2 * np.pi) ** 3 / 2)), color="k", ls=":", label="delta (x 1)")
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("cell height h_y")
    ax[2].set_title("(c) the maps (32 cells in y)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    labels = list(DROP)
    for k, step in enumerate((10, 50, 200)):
        ax[3].bar(np.arange(len(labels)) + (k - 1) * 0.27, [DROP[l][k] for l in labels], 0.27, label=f"{step} steps")
    ax[3].set_yscale("log")
    ax[3].set_xticks(np.arange(len(labels)))
    ax[3].set_xticklabels([f"{l}\njump {DROP[l][3]:.4f}" for l in labels], fontsize=7, rotation=15)
    ax[3].set_ylabel("max |u|")
    ax[3].set_title("(d) static drop (y refined about y = 0.25)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3, axis="y")
    fig.savefig(f"{args.out}/mapped_cls.png", dpi=100)
    print("wrote", f"{args.out}/mapped_cls.png")


if __name__ == "__main__":
    main()

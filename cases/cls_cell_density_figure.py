"""Figure of docs/notes/68: frequency error and damping of the CLS's capillary wave and drop against N, with the
diffuse and the sharp cell density of the viscous solve, and the VOF for reference (values from
``python -m cases.cls_cell_density`` and ``python -m cases.vof_imex_dynamics``, copied below).

    python -m cases.cls_cell_density_figure
"""

from __future__ import annotations

N = (32, 64)
SERIES = {  # (capillary wave omega %, drop (heights) omega %, capillary wave damping ratio, drop damping ratio)
    "CLS, cell_density='diffuse' (default)": ((-9.30, -4.22), (-5.13, -3.38), (0.50, 1.43), (2.51, 3.90)),
    "CLS, cell_density='sharp'": ((-4.83, -1.26), (-2.38, -1.38), (0.31, 0.75), (2.51, 0.84)),
    "VOF on IMEX (CSF)": ((-3.98, -1.33), (-1.54, -1.04), (2.05, 1.18), (None, None)),
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    titles = ("capillary wave: frequency error (%)", "drop: frequency error (%)",
              "capillary wave: damping / mode", "drop: damping / mode")
    for k, ax in enumerate(axes):
        for name, vals in SERIES.items():
            v = vals[k]
            if v[0] is not None:
                ax.plot(N, v, "o-", label=name)
        ax.axhline(0 if k < 2 else 1, color="k", lw=0.8)
        ax.set_xticks(N)
        ax.set_xlabel("N")
        ax.set_title(titles[k], fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/cls_cell_density.png", dpi=150)


if __name__ == "__main__":
    main()

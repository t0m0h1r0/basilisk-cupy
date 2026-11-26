"""Figure of docs/notes/66: rms curvature error of a VOF circle (R = 0.25, 8 centre offsets) against N, from the
output of ``python -m cases.vof_curvature_ccd`` (values copied below).

    python -m cases.vof_curvature_figure
"""

from __future__ import annotations

N = (32, 64, 128, 256)
RMS = {
    "tension.h, fraction() init": (5.51e-2, 5.15e-2, 5.37e-2, 5.54e-2),
    "tension.h, exact fractions": (8.29e-3, 1.97e-3, 4.87e-4, 1.21e-4),
    "window column sums + CCD": (9.31e-3, 1.59e-3, 3.85e-4, 9.71e-5),
    "column positions + CCD (vof_curvature='ccd')": (2.48e-2, 3.03e-3, 1.81e-4, 2.28e-6),
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, v in RMS.items():
        ax.loglog(N, v, "o-", label=name)
    ax.loglog(N, [8.3e-3 * (32 / n) ** 2 for n in N], "k:", lw=1, label="2nd order")
    ax.set_xticks(N, [str(n) for n in N])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("N (R = N/4 cells)")
    ax.set_ylabel("rms relative curvature error")
    ax.set_title("VOF circle: initialisation and curvature")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/vof_curvature.png", dpi=150)


if __name__ == "__main__":
    main()

"""Figure of docs/notes/63: static drop jump error against N for each curvature and face operator (density ratio
1000, t = 0.5), from the output of ``python -m cases.phase_ccd_check --curvature-scan`` (values copied below).

    python -m cases.curvature_scan_figure
"""

from __future__ import annotations

N = (32, 64, 128)
ERR = {  # |jump - 4| / 4
    ("psi", "fd2"): (2.63e-3, 7.54e-4, 2.66e-4),
    ("psi", "phase-ccd"): (5.38e-4, 2.14e-4, 1.68e-4),
    ("phi", "fd2"): (2.59e-3, 6.38e-4, 1.52e-4),
    ("phi", "phase-ccd"): (4.90e-4, 4.66e-5, 9.95e-6),
    ("heights-fd2", "fd2"): (3.25e-3, 8.16e-4, 2.07e-4),
    ("heights-ccd", "fd2"): (2.14e-4, 3.78e-4, 4.48e-5),
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (curv, faces), e in ERR.items():
        ax.loglog(N, e, "o-" if faces == "phase-ccd" else "s--", label=f"{curv} / {faces}")
    ax.loglog(N, [2.6e-3 * (32 / n) ** 2 for n in N], "k:", lw=1, label="2nd order")
    ax.set_xlabel("N")
    ax.set_ylabel("|jump - 4| / 4")
    ax.set_xticks(N, [str(n) for n in N])
    ax.set_title("Static drop, density ratio 1000, t = 0.5: curvature x face operator")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/figures/curvature_scan.png", dpi=150)


if __name__ == "__main__":
    main()

"""Figure of docs/notes/65: frequency error of the paper's capillary wave (water--air, 5 periods) against N for the
VOF's CSF and ghost-fluid jump with the cell-centred correction's two coefficients, from the output of
``python -m cases.vof_gfm_factors`` and ``python -m cases.vof_imex_dynamics`` (values copied below).

    python -m cases.cell_coefficient_figure
"""

from __future__ import annotations

N = (32, 64)
OMEGA = {  # % error of the angular frequency against the normal mode
    "VOF on CenteredNS (Basilisk)": (-3.98, -1.33),
    "VOF on IMEX, CSF": (-3.96, -1.35),
    'VOF-GFM, cell_coefficient="gfm" (note 64)': (-5.81, -2.31),
    'VOF-GFM, cell_coefficient="mean"': (-4.91, -1.44),
    "CLS (default)": (-9.30, -4.22),
}
BETA = {  # damping rate / normal mode's
    "VOF on CenteredNS (Basilisk)": (2.49, 1.31),
    "VOF on IMEX, CSF": (2.05, 1.18),
    'VOF-GFM, cell_coefficient="gfm" (note 64)': (-0.56, -0.44),
    'VOF-GFM, cell_coefficient="mean"': (0.25, 0.37),
    "CLS (default)": (0.50, 1.43),
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for name, v in OMEGA.items():
        a.plot(N, v, "o-", label=name)
    for name, v in BETA.items():
        b.plot(N, v, "o-", label=name)
    a.axhline(0, color="k", lw=0.8)
    b.axhline(1, color="k", lw=0.8)
    b.axhline(0, color="gray", lw=0.6, ls=":")
    for ax in (a, b):
        ax.set_xticks(N)
        ax.set_xlabel("N")
    a.set_ylabel("frequency error (%)")
    b.set_ylabel("damping / normal mode (< 0: grows)")
    a.set_title("Capillary wave, water-air: frequency")
    b.set_title("damping")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/cell_coefficient.png", dpi=150)


if __name__ == "__main__":
    main()

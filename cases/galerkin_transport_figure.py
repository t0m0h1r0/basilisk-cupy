"""Figure for docs/notes/81: P0-1 convergence (piecewise vs single polynomial, q with per-phase functions) and P0-4
the jump of the update operator when the interface crosses a cell boundary.

    python -m cases.galerkin_transport_figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cases.galerkin_transport import run, switching  # noqa: E402


def main() -> None:
    ns = (16, 32, 64, 128, 256)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, mk in ((3, "s"), (4, "o")):
        for pw, ls, lab in ((True, "-", "piecewise (shared a0)"), (False, "--", "single polynomial")):
            e = [run(n, p, kind="phases", piecewise=pw)[0] for n in ns]
            ax[0].loglog(ns, e, ls, marker=mk, label=f"degree {p}, {lab}")
    ax[0].loglog(ns, [3e-4 * (16 / n) ** 4 for n in ns], ":", color="gray", label="4th order")
    ax[0].loglog(ns, [3e-2 * (16 / n) for n in ns], "-.", color="gray", label="1st order")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("max error at t = 0.137")
    ax[0].set_title("q: different non-polynomial functions per phase")
    ax[0].legend(fontsize=7)
    eps = [10.0 ** -k for k in range(2, 9)]
    for args, lab in (((True, True), "piecewise, across a cell boundary"), ((False, True), "piecewise, within a cell"),
                      ((True, False), "single, across"), ((True, "blend"), "blend, across")):
        v = [switching(32, 4, e, args[0], args[1])[0] for e in eps]
        ax[1].loglog(eps, v, marker="o", label=lab)
    ax[1].set_xlabel("eps (interface moves by 2 eps h)")
    ax[1].set_ylabel("||G^-1 C - I||_2")
    ax[1].set_title("continuity of the update (N = 32, degree 4)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/figures/galerkin_piecewise.png", dpi=130)


if __name__ == "__main__":
    main()

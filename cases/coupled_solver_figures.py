"""Convergence, stability and cost of the coupled outer iterations (docs/notes/36): ``coupled_solver.png``.

All numbers were measured by the scripts named in each panel and are copied here.

(a) ``cases/coupled_residuals.py``: one step of the oscillating drop (N = 32, mu = 0.1, 1.6 x tension.h's
    step) against the outer iteration k: the change ``dt |g_k - g_{k-1}|``, the momentum residual and the
    error ``|u_k - u_ref|``, density ratios 1, 100, 1000.
(b) ``cases/coupled_limits.py``: stability limit of the static drop against N, K = 0, K = 1 and converged
    iterations, density ratios 1, 100, 1000 (N = 128: GPU, 4 bisections).
(c) ``cases/coupled_acceleration.py``: mean outer iterations per step to ``coupled_rtol=1e-3`` (oscillating
    drop, ratio 100, 20 steps), sharp and smooth jump forms, N = 32 and 64.
(d) ``cases/coupled_cost.py``: wall time against the error of the deformation at t = 0.3 (N = 32, ratio 100).

    python -m cases.coupled_solver_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: (a) per iteration k = 1..12: change, residual, error (N = 32).
ITER = {
    1: ([8.60e-04, 2.24e-04, 7.87e-05, 2.86e-05, 1.07e-05, 4.09e-06, 1.60e-06, 6.33e-07, 2.54e-07, 1.03e-07, 4.40e-08, 1.90e-08],
        [2.12e-03, 5.03e-04, 1.51e-04, 5.10e-05, 1.90e-05, 7.29e-06, 2.85e-06, 1.18e-06, 4.98e-07, 2.12e-07, 9.08e-08, 3.92e-08],
        [4.37e-04, 9.99e-05, 2.92e-05, 9.73e-06, 3.69e-06, 1.42e-06, 5.54e-07, 2.24e-07, 9.44e-08, 4.01e-08, 1.72e-08, 7.39e-09]),
    100: ([1.43e-01, 3.79e-02, 1.26e-02, 4.83e-03, 2.04e-03, 9.84e-04, 5.81e-04, 3.45e-04, 2.17e-04, 1.38e-04, 8.80e-05, 5.65e-05],
          [6.11e-01, 1.59e-01, 5.63e-02, 2.93e-02, 1.50e-02, 7.82e-03, 4.22e-03, 2.99e-03, 2.07e-03, 1.70e-03, 1.39e-03, 1.10e-03],
          [8.42e-02, 2.12e-02, 6.88e-03, 2.67e-03, 1.17e-03, 5.53e-04, 2.76e-04, 1.48e-04, 9.10e-05, 5.97e-05, 3.94e-05, 2.62e-05]),
    1000: ([3.82e-01, 1.03e-01, 3.45e-02, 1.33e-02, 5.63e-03, 2.85e-03, 1.71e-03, 1.05e-03, 6.90e-04, 4.55e-04, 3.02e-04, 2.03e-04],
           [1.63e+00, 4.34e-01, 1.63e-01, 8.62e-02, 4.45e-02, 2.34e-02, 1.61e-02, 1.17e-02, 9.27e-03, 8.12e-03, 6.62e-03, 5.65e-03],
           [2.23e-01, 5.75e-02, 1.89e-02, 7.38e-03, 3.27e-03, 1.58e-03, 8.11e-04, 4.77e-04, 2.87e-04, 1.80e-04, 1.22e-04, 9.47e-05]),
}
#: (b) {(ratio, scheme): {N: (stable, unstable)}}
LIMITS = {
    (1, "K0"): {16: (1.320, 1.359), 32: (1.203, 1.242), 64: (1.164, 1.203)},
    (1, "K1"): {16: (2.141, 2.180), 32: (2.219, 2.258), 64: (2.297, 2.336)},
    (1, "converged"): {16: (2.141, 2.180), 32: (2.063, 2.102), 64: (2.102, 2.141), 128: (2.225, 2.300)},
    (100, "K0"): {16: (1.125, 1.164), 32: (1.047, 1.086), 64: (0.969, 1.008)},
    (100, "K1"): {16: (1.750, 1.789), 32: (1.281, 1.320), 64: (0.930, 0.969)},
    (100, "converged"): {16: (1.828, 1.867), 32: (1.867, 1.906), 64: (1.828, 1.867), 128: (1.725, 1.800)},
    (1000, "K0"): {16: (1.125, 1.164), 32: (1.008, 1.047), 64: (0.969, 1.008)},
    (1000, "K1"): {16: (1.711, 1.750), 32: (1.242, 1.281), 64: (0.891, 0.930)},
    (1000, "converged"): {16: (1.828, 1.867), 32: (1.828, 1.867), 64: (1.828, 1.867)},
}
#: (c) mean iterations per step and relative error at rtol 1e-3: {(N, jump): {variant: (iterations, error)}}
ACCEL = {
    (32, "sharp"): {"plain": (11.55, 2.41e-5), "predictor": (13.25, 4.40e-4), "anderson1": (8.30, 1.69e-5),
                    "anderson2": (7.55, 2.42e-5), "anderson3": (7.55, 2.51e-5), "predictor+anderson2": (8.15, 2.42e-4)},
    (32, "smooth"): {"plain": (13.00, 4.58e-6), "predictor": (12.30, 1.42e-6), "anderson1": (9.25, 3.08e-6),
                     "anderson2": (8.75, 3.72e-6), "anderson3": (8.65, 3.48e-6), "predictor+anderson2": (8.20, 1.02e-6)},
    (64, "sharp"): {"plain": (18.65, 3.91e-3), "predictor": (21.40, 2.01e-3), "anderson1": (11.40, 4.80e-3),
                    "anderson2": (10.45, 4.44e-3), "anderson3": (9.80, 3.53e-3), "predictor+anderson2": (10.55, 2.80e-3)},
    (64, "smooth"): {"plain": (12.60, 4.24e-6), "predictor": (10.15, 7.49e-6), "anderson1": (9.05, 5.22e-6),
                     "anderson2": (8.50, 5.44e-6), "anderson3": (8.25, 5.07e-6), "predictor+anderson2": (7.20, 4.16e-6)},
}
#: (d) (label, wall seconds, relative error of D(t = 0.3), steps, outer iterations)
COST = [("K0, 0.95 x", 9.3, 1.60e-2, 153, 153), ("K2, 1.6 x", 12.8, 7.95e-2, 95, 190),
        ("tol 1e-10, 1.7 x", 118.3, 2.95e-2, 90, 2700), ("rtol 1e-2, 1.7 x", 30.8, 3.55e-2, 90, 589),
        ("rtol 1e-2 + Anderson 2, 1.7 x", 25.0, 2.61e-2, 90, 472),
        ("smooth rtol 1e-2, 1.7 x", 29.4, 2.95e-2, 90, 575),
        ("smooth rtol 1e-2 + predictor + Anderson 2, 1.7 x", 22.3, 2.95e-2, 90, 438)]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.4), constrained_layout=True)

    k = np.arange(1, 13)
    for ratio, c in ((1, "C0"), (100, "C1"), (1000, "C3")):
        ch, res, err = ITER[ratio]
        ax[0].semilogy(k, ch, "o-", color=c, label=f"ratio {ratio}: change dt|dg|")
        ax[0].semilogy(k, res, "s--", color=c, mfc="none", label=f"ratio {ratio}: momentum residual")
        ax[0].semilogy(k, err, "^:", color=c, label=f"ratio {ratio}: error |u_k - u_ref|")
    ax[0].set_xlabel("outer iteration k")
    ax[0].set_title("(a) one step, oscillating drop, N = 32, 1.6 x")
    ax[0].legend(fontsize=7, ncol=1)
    ax[0].grid(alpha=0.3, which="both")

    styles = {"K0": "o-", "K1": "s--", "converged": "^-"}
    colors = {1: "C0", 100: "C1", 1000: "C3"}
    for (ratio, scheme), d in LIMITS.items():
        ns = sorted(d)
        mid = [np.mean(d[n]) for n in ns]
        ax[1].semilogx(ns, mid, styles[scheme], color=colors[ratio], base=2, label=f"ratio {ratio}, {scheme}",
                       mfc="none" if scheme == "K1" else None)
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("stable factor of tension.h's step")
    ax[1].set_title("(b) static drop: limit against the grid")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3)

    variants = list(ACCEL[(32, "sharp")])
    w = 0.2
    for i, key in enumerate(ACCEL):
        ax[2].bar(np.arange(len(variants)) + (i - 1.5) * w, [ACCEL[key][v][0] for v in variants], w,
                  label=f"N = {key[0]}, {key[1]}")
    ax[2].set_xticks(np.arange(len(variants)))
    ax[2].set_xticklabels(variants, rotation=25, fontsize=8)
    ax[2].set_ylabel("outer iterations per step (rtol 1e-3)")
    ax[2].set_title("(c) acceleration, ratio 100, oscillating drop")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")

    for i, (label, wall, err, steps, its) in enumerate(COST):
        ax[3].loglog(wall, err, "o", ms=9, color=f"C{i}", label=f"{label}: {steps} steps, {its} iterations")
    ax[3].set_xlabel("wall time to t = 0.3 [s] (CPU)")
    ax[3].set_ylabel("relative error of the deformation D(t = 0.3)")
    ax[3].set_title("(d) cost for a fixed time, N = 32, ratio 100")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3, which="both")
    fig.savefig(f"{args.out}/coupled_solver.png", dpi=100)
    print("wrote", f"{args.out}/coupled_solver.png")


if __name__ == "__main__":
    main()

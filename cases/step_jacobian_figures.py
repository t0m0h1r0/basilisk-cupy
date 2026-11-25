"""Linear stability of one CenteredNS + TwoPhaseCLS step (docs/notes/35): ``step_jacobian.png`` and
``step_jacobian_modes.png``.

Static drop R = 0.25, sigma = 1, mu = 0.1, N = 32, GFM jump, ``factor`` times tension.h's step.

``step_jacobian.png``. Left: spectral radius of the step's Jacobian (matrix-free, Arnoldi; ``cases/step_jacobian.py``)
against the factor, density ratios 1 and 100, K = 0, 1, 2, 8 outer iterations (8: converged; measured with
``coupled_tolerance=1e-10``), with the measured limits (800 steps,
solvers to 1e-14) as shaded brackets. Middle: the leading Ritz values in the complex plane at 1.6 x the step,
density ratio 100, K = 1, 2, 8. Right: what one outer iteration with a lagging pressure leaves after the projection,
``|P (B - I) dt alpha grad q| / |dt grad q|`` (``cases/operator_commutator.py``), against the density ratio, and for
the cases A-E that vary the density and viscosity ratios apart.

``step_jacobian_modes.png``: the leading mode (real part of psi, speed, p) of the K = 0 capillary instability
(ratio 1, 1.4 x) and of the K = 1 oscillatory instability (ratio 100, 1.6 x).

    python -m cases.step_jacobian_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

#: Spectral radius |lambda_1| (arg/pi) by Arnoldi, N = 32, mu = 0.1; {(ratio, K): {factor: (|lambda|, arg/pi)}}.
SPECTRAL = {
    (100, 0): {0.8: (1.0003, 0), 1.0: (0.9999, 0), 1.019: (1.0000, 0), 1.075: (1.0787, 1), 1.2: (1.5284, 1),
               1.4: (2.3424, 1), 1.6: (3.2734, 1), 2.0: (5.4887, 1), 2.4: (8.1788, 1)},
    (100, 1): {0.8: (1.0024, 0), 1.0: (0.9987, 0), 1.188: (0.9993, 0), 1.2: (0.9987, 0), 1.244: (0.9987, 0),
               1.3: (1.0072, -0.207), 1.4: (1.0647, 0.214), 1.6: (1.1885, -0.220), 2.0: (1.4629, -0.216),
               2.4: (1.855, 1)},
    (100, 2): {0.8: (0.9966, 0), 1.0: (0.999, 0), 1.2: (0.9989, 0), 1.4: (0.9995, 0), 1.6: (0.999, 0),
               2.0: (0.9979, 0), 2.144: (1.0696, 1), 2.2: (1.1866, 1), 2.4: (1.6149, 1)},
    (1, 0): {0.8: (0.9994, 0), 1.0: (0.9996, 0), 1.188: (0.9997, 0), 1.2: (0.9997, 0), 1.244: (1.0393, 1),
             1.4: (1.5348, 1), 1.6: (2.2498, 1), 2.0: (3.9422, 1), 2.4: (5.9889, 1)},
    (1, 1): {0.8: (0.9985, 0), 1.0: (0.9992, 0), 1.2: (0.9995, 0), 1.4: (0.9996, 0), 1.6: (0.9996, 0),
             2.0: (0.9992, 0), 2.2: (0.999, 0), 2.256: (1.022, 1), 2.4: (1.3077, 1)},
    (1, 2): {0.8: (0.9984, 0), 1.0: (0.9993, 0), 1.2: (0.9997, 0), 1.4: (0.9998, 0), 1.6: (0.9997, 0),
             1.975: (0.9992, 0), 2.0: (0.9991, 0), 2.031: (0.9991, 0), 2.4: (1.6413, 1)},
    (100, 8): {1.6: (0.9996, 0), 1.7: (0.9993, 0), 1.8: (0.9999, 0), 1.9: (1.168, 1), 2.0: (1.3926, 1)},
    (1, 8): {1.6: (1.0002, 0), 2.05: (0.9994, 0), 2.1: (1.0689, 1), 2.15: (1.1693, 1), 2.2: (1.2701, 1)},
}
#: Measured limits (stable, unstable factor), 800 steps from the initial drop, solvers to 1e-14.
MEASURED = {(100, 0): (1.019, 1.075), (100, 1): (1.244, 1.3), (100, 2): (2.0, 2.144),
            (1, 0): (1.188, 1.244), (1, 1): (2.2, 2.256), (1, 2): (2.031, 2.1),
            # converged iterations: coupled_iterations=16, coupled_tolerance=1e-10
            (100, 8): (1.8, 1.9), (1, 8): (2.05, 2.1)}
#: Leading Ritz values (|lambda|, arg/pi), ratio 100, 1.6 x, K = 1, 2, 8.
RITZ = {1: [(1.1891, 0.22), (1.1867, 0.22), (1.1863, 0.22), (1.1765, 0.22)],
        2: [(0.9989, 0.001), (0.9629, 0.015), (0.9542, 0.003), (0.9303, 0.031)],
        8: [(0.9996, 0.0), (0.9983, 0.0), (0.9657, 0.0), (0.9596, 0.015), (0.9559, 0.0), (0.931, 0.031)]}
#: |P (B - I) G| / |G_1|, N = 32 and 64 (cases/operator_commutator.py).
LAG_RATIO = {32: {1: 3.902e-02, 10: 6.998e-01, 100: 8.062e+00, 1000: 4.654e+01},
             64: {1: 3.082e-02, 10: 5.018e-01, 100: 6.369e+00, 1000: 5.933e+01}}
LAG_CASES = {"A: rho 1, mu 1": 3.902e-02, "B: rho 100, mu 1": 8.062e+00, "C: rho 100, nu 1": 2.764e+00,
             "D: rho 1, mu 100": 2.641e-03, "E: rho 1000, mu 1": 4.654e+01}
#: Spectral radius at K = 1, 1.6 x, of the cases (Arnoldi) and the measured outcome.
LAG_K1 = {"A: rho 1, mu 1": (0.9996, "stable"), "B: rho 100, mu 1": (1.1885, "unstable"),
          "C: rho 100, nu 1": (1.0925, "unstable"), "D: rho 1, mu 100": (0.9992, "stable"),
          "E: rho 1000, mu 1": (1.2223, "unstable")}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.step_jacobian import leading

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    colors = {0: "C0", 1: "C1", 2: "C2", 8: "C3"}
    for (ratio, K), d in SPECTRAL.items():
        f = sorted(d)
        r = [d[x][0] for x in f]
        ax[0].plot(f, r, "o-" if ratio == 100 else "s--", color=colors[K], label=f"ratio {ratio}, K = {K}" + (" (converged)" if K == 8 else ""))
        osc = [x for x in f if 0.05 < abs(d[x][1]) < 0.95]
        ax[0].plot(osc, [d[x][0] for x in osc], "*", ms=14, color="k", zorder=5)
        lo, hi = MEASURED[(ratio, K)]
        ax[0].axvspan(lo, hi, ymin=0.0, ymax=0.04 if ratio == 1 else 0.08, color=colors[K], alpha=0.6)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].set_ylim(0.9, 3.0)
    ax[0].set_xlabel("factor of tension.h's step")
    ax[0].set_ylabel("|lambda_1| of the step")
    ax[0].set_title("N = 32: eigenvalue vs measured limit (bars at the bottom)\n* complex lambda (arg ~ 0.21 pi), else real")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    t = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(np.cos(t), np.sin(t), "k", lw=0.8)
    for K, mk in ((1, "o"), (2, "s"), (8, "^")):
        z = [m * np.exp(1j * np.pi * a) for m, a in RITZ[K]]
        z = z + [np.conj(v) for v in z]
        ax[1].plot(np.real(z), np.imag(z), mk, label=f"K = {K}", ms=8, mfc="none")
    ax[1].set_aspect("equal")
    ax[1].set_xlim(-0.2, 1.3)
    ax[1].set_ylim(-0.9, 0.9)
    ax[1].set_title("leading Ritz values, ratio 100, 1.6 x step")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    for n, mk in ((32, "o-"), (64, "s--")):
        x = sorted(LAG_RATIO[n])
        ax[2].loglog(x, [LAG_RATIO[n][k] for k in x], mk, label=f"mu equal, N = {n}")
    for name, v in LAG_CASES.items():
        lam, outcome = LAG_K1[name]
        ratio = {"A": 1, "B": 100, "C": 100, "D": 1, "E": 1000}[name[0]]
        ax[2].plot(ratio * (1.4 if name[0] in "CD" else 1.0), v, "*" if outcome == "unstable" else "P", ms=13,
                   color="C3" if outcome == "unstable" else "C2")
        ax[2].annotate(f"{name[0]} |lam| {lam:.3f}", (ratio * (1.4 if name[0] in "CD" else 1.0), v), fontsize=8,
                       xytext=(5, -10), textcoords="offset points")
    ax[2].set_xlabel("density ratio")
    ax[2].set_ylabel("|P (B - I) dt alpha grad q| / |dt grad q|")
    ax[2].set_title("what one iteration with a lagging pressure leaves\n(red *: unstable at K = 1, 1.6 x; green: stable)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")
    fig.savefig(f"{args.out}/step_jacobian.png", dpi=100)
    print("wrote", f"{args.out}/step_jacobian.png", flush=True)

    fig, ax = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for row, (rho2, K, fac, title) in enumerate(((1.0, 0, 1.4, "ratio 1, K = 0, 1.4 x"),
                                                 (0.01, 1, 1.6, "ratio 100, K = 1, 1.6 x"))):
        lam, X, sm = leading(rho2, K, fac, m=40, nev=2)
        part = sm.split(X[:, 0] * np.exp(-1j * np.angle(X[np.argmax(np.abs(X[:, 0])), 0])))
        lab = f"{title}: lambda = {abs(lam[0]):.3f} exp({np.angle(lam[0]) / np.pi:+.2f} i pi)"
        for col, (name, field) in enumerate((("psi", part["psi"].real),
                                             ("speed", np.hypot(part["u"].real, part["v"].real)),
                                             ("p", part["p"].real))):
            a = ax[row, col]
            v = np.abs(field).max()
            im = a.imshow(field.T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
                          cmap="viridis" if name == "speed" else "RdBu_r", vmin=0 if name == "speed" else -v, vmax=v)
            a.add_patch(plt.Circle((0, 0), 0.25, fill=False, color="k", lw=0.6))
            a.set_title(f"{name}\n{lab}" if col == 0 else name, fontsize=9)
            fig.colorbar(im, ax=a, shrink=0.8)
    fig.savefig(f"{args.out}/step_jacobian_modes.png", dpi=100)
    print("wrote", f"{args.out}/step_jacobian_modes.png")


if __name__ == "__main__":
    main()

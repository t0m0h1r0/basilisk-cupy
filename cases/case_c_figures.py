"""Second-order time integration of the coupled stage (Case C) and the per-step approximate projection
(docs/notes/47): ``case_c.png``. Measured with :mod:`cases.case_c` (eigenvalue of the step's Jacobian, N = 32,
x 1, exact curvature, band-corrected kinematics); numbers copied here.

    python -m cases.case_c_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

DIVS = [1, 2, 4, 8]
#: damping error: faces rebuilt from the cells each step (CenteredNS) / faces kept (uf^n as the face base)
INTERP = {("A (CenteredNS)", "fd2"): [0.0772, 0.0636, 0.0519, 0.0435],
          ("BE", "fd2"): [0.21466, 0.13267, 0.08646, 0.06077], ("CN", "fd2"): [0.08492, 0.06775, 0.054, 0.04455],
          ("SDIRK2", "fd2"): [0.0806, 0.06524, 0.05258, 0.04377],
          ("BE", "ccd"): [0.17825, 0.10327, 0.06182, 0.03858], ("CN", "ccd"): [0.05129, 0.03964, 0.02998, 0.02265],
          ("SDIRK2", "ccd"): [0.04742, 0.03738, 0.02868, 0.02192]}
FACES = {("BE", "fd2"): [0.22089, 0.1527, 0.11837, 0.10116], ("CN", "fd2"): [0.08364, 0.08384, 0.08389, 0.08391],
         ("SDIRK2", "fd2"): [0.08378, 0.08388, 0.0839, 0.08391], ("CN", "ccd"): [0.04441, 0.04461, 0.04465, 0.04467]}
FREQ_FACES = {"CN fd2": 0.0299, "CN ccd": 0.0148}
ITERS = {"BE": 3.5, "CN": 3.4, "SDIRK2 (per stage)": 2.9}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    x = [1 / d for d in DIVS]
    st = {"A (CenteredNS)": "k*-", "BE": "o-", "CN": "s-", "SDIRK2": "^-"}
    for (m, v), y in INTERP.items():
        ax[0].plot(x, [100 * e for e in y], st[m], color=None if m.startswith("A") else ("C0" if v == "fd2" else "C1"),
                   mfc="none" if v == "ccd" else None, label=f"{m}, {v}")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("dt / dt0")
    ax[0].set_ylabel("damping-rate error [%]")
    ax[0].set_title("(a) faces rebuilt from the cells each step (CenteredNS)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    for (m, v), y in FACES.items():
        ax[1].plot(x, [100 * e for e in y], st[m], color="C0" if v == "fd2" else "C1", label=f"{m}, {v}")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("dt / dt0")
    ax[1].set_ylabel("damping-rate error [%]")
    ax[1].set_title("(b) faces kept (u_f^n as the face base): 2nd order, another limit")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for (m, v), y in list(INTERP.items()) + [((k[0] + " faces", k[1]), v_) for k, v_ in FACES.items()]:
        d = np.abs(np.diff(y))
        if np.all(d > 0):
            ax[2].loglog(x[1:], d, "o-" if "faces" not in m else "s--", label=f"{m}, {v}")
    xs = np.array(x[1:])
    ax[2].loglog(xs, 0.02 * xs, "k:", lw=1, label="slope 1")
    ax[2].loglog(xs, 0.02 * xs ** 2, "k--", lw=1, label="slope 2")
    ax[2].set_xlabel("dt / dt0")
    ax[2].set_ylabel("|error(dt) - error(2 dt)|")
    ax[2].set_title("(c) observed order")
    ax[2].legend(fontsize=6, ncol=2)
    ax[2].grid(alpha=0.3, which="both")
    fig.savefig(f"{args.out}/case_c.png", dpi=100)
    print("wrote", f"{args.out}/case_c.png")


if __name__ == "__main__":
    main()

"""Jump-corrected viscosity (IIM) tried on the vortex layer (docs/notes/40): ``jump_correction.png``.

(a) frozen flat interface driven by a GFM jump ``j = 0.08 cos kx sin 10 t`` (``cases.jump_frozen_layer``):
    the ``sin kx`` part of u at t = 0.9, N = 32 and 64 with and without the correction, against N = 256;
(b) capillary wave from the normal mode, N = 64 (delta/h = 1.83), height-function curvature: amplitude
    with and without the correction (recomputed);
(c) measured: damping error at N = 32 (delta/h = 0.92, heights) when the correction is scaled by c;
(d) measured: damping error against delta/h, psi curvature (docs/notes/38) and height-function curvature.

    python -m cases.jump_correction_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

#: N = 32, x 1, heights: scale c of the correction -> damping error (None: blew up)
SCALE = {-1.0: None, -0.5: -0.0767, 0.0: -0.1136, 0.25: -0.1836, 0.5: -3.5954, 1.0: None}
#: (N, viscosity factor) -> (delta/h, damping error psi curvature (38), heights)
CURVATURE = {(32, 0.5): (0.65, -0.2783, -0.2293), (32, 1.0): (0.92, -0.1635, -0.1136),
             (32, 2.0): (1.30, -0.0906, -0.0507), (64, 0.5): (1.30, -0.0743, -0.0256),
             (64, 1.0): (1.83, -0.0375, -0.0059), (64, 2.0): (2.59, -0.0203, -0.0016),
             (128, 1.0): (3.66, 0.0051, 0.0152)}


def _frozen(a):
    from cases.jump_frozen_layer import run

    n, c = a
    return a, run(n, c)


def _wave(c):
    from cases.capwave_normal_mode import run

    r = run(64, 1.0, periods=1.5, cls_kw={"curvature_kw": {"method": "heights"}}, jump_correction=c)
    return c is not None, r["t"][:len(r["amplitudes"])], r["amplitudes"], r["damping_err"]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    cases = [(256, None), (32, None), (32, {}), (64, None), (64, {})]
    with ProcessPoolExecutor(7) as ex:
        fr = dict(((n, c is not None), r) for (n, c), r in ex.map(_frozen, cases))
        wv = list(ex.map(_wave, (None, {})))
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)
    yr, ur = fr[(256, False)]
    m = np.abs(yr) < 0.3
    ax[0].plot(yr[m], ur[m], "k-", label="N = 256")
    for (n, c), st in (((32, False), "o-"), ((32, True), "o--"), ((64, False), "s-"), ((64, True), "s--")):
        y, u = fr[(n, c)]
        mm = np.abs(y) < 0.3
        err = np.abs(u - np.interp(y, yr, ur))[mm].max()
        ax[0].plot(y[mm], u[mm], st, mfc="none" if c else None,
                   label=f"N = {n}{', corrected' if c else ''}: max error {err:.1e}")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("sin kx part of u, t = 0.9")
    ax[0].set_title("(a) frozen interface, prescribed jump")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for corr, t, a, e in wv:
        ax[1].plot(t, a, "--" if corr else "-",
                   label=("corrected" if corr else "no correction") + (f": damping error {100 * e:.1f} %" if np.isfinite(e) else ": blew up"))
    ax[1].set_ylim(-0.03, 0.03)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("interface amplitude")
    ax[1].set_title("(b) capillary wave, N = 64, height curvature")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    cs = list(SCALE)
    vals = [100 * SCALE[c] if SCALE[c] is not None and abs(SCALE[c]) < 1 else np.nan for c in cs]
    ax[2].bar(range(len(cs)), [v if np.isfinite(v) else 0 for v in vals], color=["C3" if not np.isfinite(v) else "C0" for v in vals])
    for i, v in enumerate(vals):
        ax[2].annotate("unstable" if not np.isfinite(v) else f"{v:.1f}", (i, (v if np.isfinite(v) else 0) - 1.5), ha="center", fontsize=8)
    ax[2].set_xticks(range(len(cs)))
    ax[2].set_xticklabels([f"c = {c:g}" for c in cs])
    ax[2].set_ylabel("damping-rate error [%]")
    ax[2].set_title("(c) N = 32, x 1: correction scaled by c")
    ax[2].grid(alpha=0.3, axis="y")

    for j, (lab, mk) in enumerate((("psi curvature (note 38)", "o"), ("height-function curvature", "s"))):
        for N, st in ((32, "-"), (64, "--"), (128, ":")):
            pts = sorted((v[0], v[1 + j]) for k, v in CURVATURE.items() if k[0] == N)
            ax[3].plot([p[0] for p in pts], [100 * p[1] for p in pts], mk + st, color=f"C{j}",
                       label=f"{lab}, N = {N}")
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set_xlabel("delta / h")
    ax[3].set_ylabel("damping-rate error [%]")
    ax[3].set_title("(d) curvature method changes the damping error")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)
    fig.savefig(f"{args.out}/jump_correction.png", dpi=100)
    print("wrote", f"{args.out}/jump_correction.png")


if __name__ == "__main__":
    main()

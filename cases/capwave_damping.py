"""Where the capillary wave's error comes from (docs/notes/29): ``capwave_damping.png``.

Amplitude against Prosperetti's solution (16 and 32 points per wavelength) for VOF (Basilisk) and CLS with
the GFM jump (corrected divergence, 6th-order faces), the period and decay per period read from the
peaks, and the inviscid wave (mu = 0) where only numerical dissipation damps it; CLS also with the
semi-implicit surface tension (theta = 4).

    python -m cases.capwave_damping [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from cases import capillary_wave as cw

CLS = dict(interface="cls", pressure_jump=True, divergence="fccd", face_order=6)


def peaks(a, t):
    pk = [(t[i], a[i]) for i in range(1, len(a) - 1) if a[i] >= a[i - 1] and a[i] > a[i + 1]]
    return pk


def run(kind, n, mu=None, **kw):
    saved = cw.MU
    if mu is not None:
        cw.MU = mu
    try:
        r = cw.run(n, "numpy", **(CLS if kind == "cls" else {}), **kw)
    finally:
        cw.MU = saved
    P = np.array(cw.PROSPERETTI)
    a = np.array(r["amplitudes"])
    return a, P[:len(a), 0], r["error"]


def decay(a, t):
    pk = peaks(a, t)
    if len(pk) < 6:
        return float("nan"), float("nan")
    return (pk[5][0] - pk[0][0]) / 5, (pk[5][1] / pk[0][1]) ** 0.2


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    P = np.array(cw.PROSPERETTI)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for k, n in enumerate((32, 64)):
        for kind, style in (("vof", "-"), ("cls", "--")):
            a, t, e = run(kind, n)
            per, dec = decay(a, t)
            ax[k].plot(t, a, style, label=f"{kind.upper()} (error {e:.4f}, period {per:.3f}, decay {dec:.4f})")
        pper, pdec = decay(P[:, 1], P[:, 0])
        ax[k].plot(P[:, 0], P[:, 1], "k:", lw=1.2, label=f"Prosperetti (period {pper:.3f}, decay {pdec:.4f})")
        ax[k].set_title(f"capillary wave, {n // 2} points per wavelength")
        ax[k].set_xlabel("omega_0 t")
        ax[k].legend(fontsize=8)
        ax[k].grid(alpha=0.3)
    for kind, kw, label in (("vof", {}, "VOF"), ("cls", {}, "CLS + GFM"),
                            ("cls", {"implicit_tension": 4.0}, "CLS + GFM, theta = 4")):
        a, t, _ = run(kind, 32, mu=0.0, **kw)
        ax[2].semilogy(t, a, label=label)
    ax[2].set_title("inviscid (mu = 0), 16 points per wavelength")
    ax[2].set_xlabel("omega_0 t")
    ax[2].set_ylabel("amplitude")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.savefig(f"{args.out}/capwave_damping.png", dpi=100)
    print("wrote", f"{args.out}/capwave_damping.png")


if __name__ == "__main__":
    main()

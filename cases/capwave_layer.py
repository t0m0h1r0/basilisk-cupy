"""Capillary wave: is the damping set by the resolution of the interfacial vorticity layer?
(docs/notes/30) ``capwave_layer.png``.

The viscosity is scaled by 0.5 / 1 / 2 / 4 at 16 / 32 / 64 points per wavelength, for CLS (GFM, corrected
divergence, 6th-order faces) and VOF. Reference: the normal mode of two identical viscous fluids
(:mod:`cases.capwave_modes`, which agrees with Prosperetti's solution at the original viscosity). Left:
damping-rate error against delta / h, delta = sqrt(2 nu / omega_0); right: period error against points per
wavelength.

    python -m cases.capwave_layer [--out docs/figures] [--data DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os

from cases import capillary_wave as cw
from cases.capwave_damping import decay, run
from cases.capwave_modes import period_decay

K, W0 = 2 * math.pi, math.sqrt((2 * math.pi) ** 3 / 2)
FACTORS = (0.5, 1.0, 2.0, 4.0)


def point(kind, n, f):
    a, t, _ = run(kind, n, mu=cw.MU * f)
    p, d = decay(a, t)
    nu = cw.MU * f
    pt, dt_ = period_decay(K, 1.0, 1.0, nu)
    return {"kind": kind, "pts": n // 2, "f": f, "delta_h": math.sqrt(2 * nu / W0) / (2.0 / n),
            "period_err": p / pt - 1, "damping_err": math.log(d) / math.log(dt_) - 1}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--data", default=None, help="directory of saved runs (kind_n_f.json) to reuse")
    args = ap.parse_args()
    rows = []
    for f in FACTORS:
        for n in (32, 64, 128):
            for kind in ("cls", "vof"):
                path = args.data and os.path.join(args.data, f"{kind}_{n}_{f}.json")
                if path and os.path.exists(path):
                    d = json.load(open(path))
                    nu = cw.MU * f
                    pt, dt_ = period_decay(K, 1.0, 1.0, nu)
                    rows.append({"kind": kind, "pts": n // 2, "f": f,
                                 "delta_h": math.sqrt(2 * nu / W0) / (2.0 / n),
                                 "period_err": d["period"] / pt - 1,
                                 "damping_err": math.log(d["decay"]) / math.log(dt_) - 1})
                else:
                    rows.append(point(kind, n, f))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    markers = {16: "o", 32: "s", 64: "^"}
    colors = {0.5: "C0", 1.0: "C1", 2.0: "C2", 4.0: "C3"}
    for col, kind in enumerate(("cls", "vof")):
        for r in rows:
            if r["kind"] != kind:
                continue
            ax[col].plot(r["delta_h"], r["damping_err"], markers[r["pts"]], color=colors[r["f"]], ms=8)
        ax[col].axhline(0, color="k", lw=0.6)
        ax[col].set_xscale("log")
        ax[col].set_xlabel("delta / h  (delta = sqrt(2 nu / omega_0))")
        ax[col].set_ylabel("damping-rate error")
        ax[col].set_title(f"{kind.upper()}: damping against the layer resolution")
        ax[col].grid(alpha=0.3)
    for r in rows:
        if r["kind"] == "cls":
            ax[2].plot(r["pts"], r["period_err"], markers[r["pts"]], color=colors[r["f"]], ms=8)
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("points per wavelength")
    ax[2].set_ylabel("period error")
    ax[2].set_title("CLS: period against the interface resolution")
    ax[2].grid(alpha=0.3)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colors[f], marker="o", ls="", label=f"nu x {f}") for f in FACTORS]
    handles += [Line2D([], [], color="k", marker=markers[p], ls="", label=f"{p} points") for p in (16, 32, 64)]
    ax[0].legend(handles=handles, fontsize=8)
    fig.suptitle("Capillary wave: damping is set by delta / h, the period by the points per wavelength "
                 "(reference: normal mode)")
    fig.savefig(f"{args.out}/capwave_layer.png", dpi=100)
    print("wrote", f"{args.out}/capwave_layer.png")


if __name__ == "__main__":
    main()

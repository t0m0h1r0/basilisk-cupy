"""Phase 3, benchmarks 2 and 3: the paper's oscillating drop (14b) and rising bubble (14c), docs/notes/59.

* drop: D(t) = (a - b)/(a + b) from the psi = 1/2 crossings on the axes, 2 periods of the 2-D Rayleigh--Lamb mode
  (omega_0 = 59.578 1/s, T_RL = 0.10546 s), fitted as ``D0 e^{beta t} cos(omega t + phi)``;
* bubble: centroid, rise velocity and deformation of the gas (1 - psi), to t = 0.06 s (the paper publishes
  t <= 0.03 s and reports a breakdown after t ~ 0.04 s).

    python -m cases.phase3_drop_bubble [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

OPTS = {
    "imex-ab2": dict(transport="ab2", reinit_every=10, reinit_method="xi_sdf"),
    "rk-ars222": dict(tableau="ars222", reinit_every=10, reinit_method="xi_sdf"),
    "imex-noreinit": {},
    "vof": dict(method="vof"),
    "imex-ab2-xi5": dict(transport="ab2", reinit_every=5, reinit_method="xi_sdf"),
    "imex-ab2-xi1": dict(transport="ab2", reinit_every=1, reinit_method="xi_sdf"),
    "imex-ab2-comp10": dict(transport="ab2", reinit_every=10, reinit_method="compression"),
    "imex-ab2-xi10-heights": dict(transport="ab2", reinit_every=10, reinit_method="xi_sdf",
                                  curvature_kw={"method": "heights", "heights_method": "fd2"}),
}


def run(case: str, variant: str, n: int):
    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run as bench

    t0 = time.time()
    if case == "drop":
        T = 0.105460663082
        r = bench("drop", n=n, t_end=2 * T, samples=200, **OPTS[variant])
    else:
        r = bench("bubble", n=n, t_end=0.06, samples=60, **OPTS[variant])
    r["wall"] = time.time() - t0
    r.pop("psi", None)
    return r


def summarise_drop(r):
    from cases.phase3_capwave import fit

    s = [d for d in r["series"] if "D" in d and math.isfinite(d.get("D", float("nan")))]
    t = np.array([d["t"] for d in s])
    D = np.array([d["D"] for d in s])
    import cases.phase3_capwave as p3

    saved = p3.T
    p3.T = 0.105460663082
    try:
        beta, om, res = fit(t, D)
    finally:
        p3.T = saved
    return beta, om, res, s[-1]["mass"] / s[0]["mass"] - 1.0, max(d["umax"] for d in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--drop-scan", action="store_true", help="re-initialisation variants on the drop only")
    args = ap.parse_args()
    from cases.parallel import pmap

    base = ("imex-ab2", "rk-ars222", "imex-noreinit", "vof")
    items = [dict(case=c, variant=v, n=n) for c in ("drop", "bubble") for n in (32, 64) for v in base]
    if args.drop_scan:
        items = [dict(case="drop", variant=v, n=n) for n in (32, 64)
                 for v in base[:2] + ("vof", "imex-ab2-xi5", "imex-ab2-xi1", "imex-ab2-comp10", "imex-ab2-xi10-heights")]
    res = pmap(run, items, args.jobs)
    w0 = 59.578473371
    print(f"drop: Rayleigh-Lamb omega_0 = {w0} 1/s")
    for it, r in zip(items, res):
        last = r["series"][-1]
        fail = f" FAILED at t = {last['t']:.4f}" if last.get("failed") else ""
        if it["case"] == "drop":
            if fail:
                print(f"drop   N {it['n']:3d} {it['variant']:14s}:{fail}")
                continue
            beta, om, fres, vol, umax = summarise_drop(r)
            print(f"drop   N {it['n']:3d} {it['variant']:14s}: beta {beta:+7.2f} omega {om:7.2f} "
                  f"({100 * (om / w0 - 1):+.2f}%) fit res {fres:.3f} volume {vol:+.1e} max|u| {umax:.3f} "
                  f"({r['steps']} steps, {r['wall']:.0f}s)")
        else:
            at = min(r["series"], key=lambda d: abs(d["t"] - 0.03))
            y0 = r["series"][0]["yc"]
            vel = (at.get("yc", float("nan")) - y0) / at["t"] if at["t"] else float("nan")
            print(f"bubble N {it['n']:3d} {it['variant']:14s}: t=0.03: yc {1e3 * at.get('yc', float('nan')):.3f} mm, "
                  f"mean rise {vel:.4f} m/s, vc {at.get('vc', float('nan')):.4f}, D {at.get('D', float('nan')):.3f};"
                  f" end t {last['t']:.4f}{fail} ({r['steps']} steps, {r['wall']:.0f}s)")


if __name__ == "__main__":
    main()

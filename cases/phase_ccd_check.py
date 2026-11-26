"""The phase-wise CCD projection (``faces="phase-ccd"``) in flows, against fd2's GFM (docs/notes/61).

* static drop (R = 0.25, sigma = 1, mu = 0.1, walls), density ratios 1 / 1000, N = 32 / 64 / 128, t = 0.5;
* the paper's capillary wave (water--air, 32^2 and 64^2, 5 periods, ab2 transport, xi-SDF every 10 steps).

    python -m cases.phase_ccd_check [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np


CURVATURES = {
    "psi": None,
    "heights-fd2": {"method": "heights", "heights_method": "fd2"},
    "heights-ccd": {"method": "heights", "heights_method": "ccd"},
    "phi": {"source": "phi"},
    "exact": "exact",
}


def drop(faces: str, n: int, ratio: float, t_end: float = 0.5, curvature: str = "psi", taylor: bool = True):
    """``taylor=False``: phase-ccd without the cut faces' Taylor terms (docs/notes/67)."""
    warnings.filterwarnings("ignore")
    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    ck = CURVATURES[curvature]
    ns = TwoPhaseIMEX(g, 0.25 - np.hypot(x, y), rho1=1.0, rho2=1.0 / ratio, mu1=0.1, mu2=0.1, sigma=1.0,
                      faces=faces, tolerance=1e-8, curvature_kw=None if ck in (None, "exact") else dict(ck))
    if ck == "exact":  # oracle: the exact curvature 1/R in the band (the drop stays at rest)
        from basilisk_cupy.interface.cls_curvature import PSI_MIN

        def exact_curvature(**kw):
            q = ns.cls.psi.interior
            return np.where((q > PSI_MIN) & (q < 1 - PSI_MIN), 4.0, 0.0)

        ns.cls.curvature = exact_curvature
    if not taylor:
        ns.ops.taylor = False
    ns.start(events=[t_end])
    umax, t0 = 0.0, time.time()
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        u = ns.max_velocity()
        if not math.isfinite(u) or u > 10:
            break
        umax = max(umax, u)
    p = ns.pressure()
    r = np.hypot(x, y)
    return {"jump": float(p[r < 0.1].mean() - p[r > 0.4].mean()), "u_end": u, "u_max": umax,
            "steps": ns.clock.i, "t": ns.clock.t, "wall": time.time() - t0}


def capwave(faces: str, n: int):
    warnings.filterwarnings("ignore")
    from cases.phase3_capwave import T, fit
    from cases.paper_benchmarks import run as bench

    t0 = time.time()
    r = bench("capwave", n=n, t_end=5 * T, samples=500, faces=faces, transport="ab2", reinit_every=10,
              reinit_method="xi_sdf")
    s = [d for d in r["series"] if "amplitude" in d]
    beta, om, res = fit([d["t"] for d in s], [d["amplitude"] for d in s])
    return {"beta": beta, "omega": om, "fit_res": res, "failed": bool(r["series"][-1].get("failed")),
            "steps": r["steps"], "wall": time.time() - t0}


def _dispatch(kind, kw):
    return drop(**kw) if kind == "drop" else capwave(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--curvature-scan", action="store_true", help="static drop, ratio 1000, curvature variants")
    args = ap.parse_args()
    if args.curvature_scan:
        from cases.parallel import pmap

        items = [dict(faces=f, n=n, ratio=1000.0, curvature=c) for c in CURVATURES for f in ("fd2", "phase-ccd")
                 for n in (32, 64, 128)]
        for it, r in zip(items, pmap(drop, items, args.jobs)):
            print(f"{it['curvature']:12s} {it['faces']:9s} N {it['n']:3d}: jump {r['jump']:.6f} "
                  f"({r['jump'] / 4 - 1:+.2e}) u_end {r['u_end']:.2e} u_max {r['u_max']:.2e} t {r['t']:.3f} "
                  f"({r['steps']} steps, {r['wall']:.0f}s)", flush=True)
        return
    from cases.parallel import pmap
    from cases.two_fluid_modes import mode

    items = [dict(kind="drop", kw=dict(faces=f, n=n, ratio=r)) for f in ("fd2", "phase-ccd") for r in (1.0, 1000.0)
             for n in (32, 64, 128)]
    items += [dict(kind="cw", kw=dict(faces=f, n=n)) for f in ("fd2", "phase-ccd") for n in (32, 64)]
    res = pmap(_dispatch, items, args.jobs)
    s = mode(2 * math.pi / 0.01, 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    for it, r in zip(items, res):
        k = it["kw"]
        if it["kind"] == "drop":
            print(f"drop {k['faces']:9s} ratio {k['ratio']:6g} N {k['n']:3d}: jump {r['jump']:.5f} "
                  f"({r['jump'] / 4 - 1:+.2e}) u_end {r['u_end']:.2e} u_max {r['u_max']:.2e} t {r['t']:.3f} "
                  f"({r['steps']} steps, {r['wall']:.0f}s)")
        else:
            print(f"capwave {k['faces']:9s} N {k['n']:3d}: beta {r['beta']:+.2f} ({r['beta'] / s.real:.2f} x) "
                  f"omega {100 * (r['omega'] / s.imag - 1):+.2f}% fit res {r['fit_res']:.3f}"
                  f"{' FAILED' if r['failed'] else ''} ({r['steps']} steps, {r['wall']:.0f}s)")


if __name__ == "__main__":
    main()

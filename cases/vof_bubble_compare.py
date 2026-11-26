"""The paper's rising bubble (water--air, t = 0.03 s) with the VOF variants of docs/notes/64-67 against Basilisk's
VOF: centroid, mean rise velocity, deformation and the largest velocity.

    python -m cases.vof_bubble_compare [--n 32 64] [--jobs N]
"""

from __future__ import annotations

import argparse

VARIANTS = {
    "centered-vof": dict(method="vof"),
    "imex-vof": dict(interface="vof", transport="ab2"),
    "imex-vof-gfm": dict(interface="vof-gfm", transport="ab2"),
    "imex-vof-gfm-pccd": dict(interface="vof-gfm", transport="ab2", faces="phase-ccd"),
}


def run(variant: str, n: int):
    import time
    import warnings

    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run as bench

    t0 = time.time()
    r = bench("bubble", n=n, t_end=0.03, samples=30, **VARIANTS[variant])
    s = r["series"]
    at = s[-1]
    return {"t": at["t"], "failed": bool(at.get("failed")), "yc": at.get("yc"), "vc": at.get("vc"), "D": at.get("D"),
            "rise": (at["yc"] - s[0]["yc"]) / at["t"] if "yc" in at else None,
            "umax": max(d.get("umax", 0.0) for d in s), "steps": r["steps"], "wall": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[32, 64])
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        if r["failed"]:
            print(f"{it['variant']:18s} N {it['n']:3d}: FAILED at t {r['t']:.4f}", flush=True)
            return
        print(f"{it['variant']:18s} N {it['n']:3d}: yc {1e3 * r['yc']:.3f} mm rise {r['rise']:.4f} vc {r['vc']:.4f} "
              f"D {r['D']:.3f} max|u| {r['umax']:.3f} ({r['steps']} steps, {r['wall']:.0f}s)", flush=True)

    items = [dict(variant=v, n=n) for n in args.n for v in VARIANTS]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

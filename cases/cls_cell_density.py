"""The CLS with the sharp cell density of the implicit viscous solve (``cell_density="sharp"``, docs/notes/68):
the paper's capillary wave (5 periods), oscillating drop (2 periods, psi and height-function curvature) and rising
bubble (t = 0.03 s), water--air, ab2, xi-SDF every 10 steps, the library defaults otherwise.

    python -m cases.cls_cell_density [--jobs N]
"""

from __future__ import annotations

import argparse

CW = (-0.802955, 134.342861)
DROP = (-0.179428, 59.554813)


def run(case: str, density: str, n: int, curvature: str = "psi"):
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from cases.paper_benchmarks import run as bench

    opts = dict(transport="ab2", reinit_every=10, reinit_method="xi_sdf", cell_density=density)
    if curvature == "heights":
        opts["curvature_kw"] = {"method": "heights", "heights_method": "fd2"}
    if case == "bubble":
        r = bench("bubble", n=n, t_end=0.03, samples=30, **opts)
        s = r["series"]
        at = s[-1]
        return {"failed": bool(at.get("failed")), "yc": at.get("yc"), "D": at.get("D"),
                "umax": max(d.get("umax", 0.0) for d in s)}
    T, key, periods, samples = ((0.046742983863, "amplitude", 5, 500) if case == "capwave"
                                else (0.105460663082, "D", 2, 200))
    r = bench(case, n=n, t_end=periods * T, samples=samples, **opts)
    s = [d for d in r["series"] if key in d]
    saved, p3.T = p3.T, T
    try:
        beta, om, _ = p3.fit([d["t"] for d in s], [d[key] for d in s])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "failed": bool(r["series"][-1].get("failed")),
            "umax": max(d.get("umax", 0.0) for d in r["series"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        tag = f"{it['case']:7s} {it['density']:7s} {it.get('curvature', 'psi'):7s} N {it['n']:3d}"
        if it["case"] == "bubble":
            print(f"{tag}: yc {1e3 * r['yc']:.3f} mm D {r['D']:.3f} max|u| {r['umax']:.3f}"
                  f"{' FAILED' if r['failed'] else ''}", flush=True)
            return
        ref = CW if it["case"] == "capwave" else DROP
        print(f"{tag}: beta {r['beta']:+.2f} ({r['beta'] / ref[0]:.2f} x) omega {100 * (r['omega'] / ref[1] - 1):+.2f}%"
              f" max|u| {r['umax']:.3f}{' FAILED' if r['failed'] else ''}", flush=True)

    items = [dict(case=c, density=d, n=n, curvature=k) for n in (32, 64) for d in ("diffuse", "sharp")
             for c, k in (("capwave", "psi"), ("drop", "psi"), ("drop", "heights"), ("bubble", "psi"))]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

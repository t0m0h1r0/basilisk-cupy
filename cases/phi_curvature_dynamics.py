"""Curvature from the logit phi (``curvature_kw={"source": "phi"}``) with fd2 / phase-wise CCD faces on moving
interfaces (docs/notes/63): the paper's capillary wave (water--air, against the normal mode) and the oscillating
drop (against the 2-D viscous mode), ab2 transport, xi-SDF every 10 steps.

    python -m cases.phi_curvature_dynamics [--jobs N]
"""

from __future__ import annotations

import argparse

CW = (-0.802955, 134.342861)
DROP = (-0.179428, 59.554813)


def run(case: str, faces: str, curvature: str, n: int):
    import warnings

    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run as bench
    from cases.phase3_capwave import fit

    opts = dict(faces=faces, transport="ab2", reinit_every=10, reinit_method="xi_sdf")
    if curvature == "phi":
        opts["curvature_kw"] = {"source": "phi"}
    if case == "capwave":
        T = 0.046742983863
        r = bench("capwave", n=n, t_end=5 * T, samples=500, **opts)
        s = [d for d in r["series"] if "amplitude" in d]
        t, a = [d["t"] for d in s], [d["amplitude"] for d in s]
    else:
        import cases.phase3_capwave as p3

        T = 0.105460663082
        r = bench("drop", n=n, t_end=2 * T, samples=200, **opts)
        s = [d for d in r["series"] if "D" in d]
        t, a = [d["t"] for d in s], [d["D"] for d in s]
        saved, p3.T = p3.T, T
    try:
        beta, om, res = fit(t, a)
    finally:
        if case == "drop":
            p3.T = saved
    return {"beta": beta, "omega": om, "res": res, "failed": bool(r["series"][-1].get("failed")),
            "umax": max(d.get("umax", 0.0) for d in r["series"]), "steps": r["steps"], "wall": r["wall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        ref = CW if it["case"] == "capwave" else DROP
        print(f"{it['case']:7s} {it['faces']:9s} {it['curvature']:3s} N {it['n']:3d}: beta {r['beta']:+.2f} "
              f"({r['beta'] / ref[0]:.2f} x) omega {100 * (r['omega'] / ref[1] - 1):+.2f}% max|u| {r['umax']:.3f}"
              f"{' FAILED' if r['failed'] else ''} ({r['steps']} steps, {r['wall']:.0f}s)", flush=True)

    items = [dict(case=c, faces=f, curvature=k, n=n) for c in ("capwave", "drop") for n in (32, 64)
             for f, k in (("fd2", "psi"), ("fd2", "phi"), ("phase-ccd", "phi"))]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

"""Is the CLS's excess damping of the oscillating drop (3.9 times the 2-D viscous mode, docs/notes/63) the diffuse
viscosity mu(psi) of the band? The paper's drop (water--air, 2 periods, ab2, xi-SDF every 10, height-function
curvature) with CCD or fd2 viscosity, and with the face viscosity made sharp: ``mu1`` or ``mu2`` by the side of
``psi = 1/2`` at the face (the mean of the two cells' phases on a cut face); ``sharp+rho``: the cell density of the
implicit viscous solve by the side of ``psi = 1/2`` too.

    python -m cases.cls_drop_viscosity [--jobs N]
"""

from __future__ import annotations

import argparse

DROP = (-0.179428, 59.554813)


def run(viscous: str, mu: str, n: int, case: str = "drop"):
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity
    from cases.paper_benchmarks import run as bench

    hook = None
    if mu == "rho":  # the cell density of the viscous solve sharp, mu(psi) kept
        def hook(ns):
            q = ns.tp.f.interior
            ns.rho = ns.grid.xp.where(q > 0.5, ns.tp.rho1, ns.tp.rho2)
    elif mu.startswith("sharp"):
        def hook(ns):
            tp = ns.tp
            for ax in range(2):
                hl, hr, cut, theta = tp._cut(ax)
                h = 0.5 * (hl.astype(float) + hr.astype(float))
                ns.mu[ax] = h * tp.mu1 + (1.0 - h) * tp.mu2
            if mu == "sharp+rho":  # the cell density of the viscous solve as well
                q = tp.f.interior
                ns.rho = ns.grid.xp.where(q > 0.5, tp.rho1, tp.rho2)
    if case == "drop":
        T, periods, samples, key, curv = 0.105460663082, 2, 200, "D", {"method": "heights", "heights_method": "fd2"}
    else:
        T, periods, samples, key, curv = 0.046742983863, 5, 500, "amplitude", None
    r = bench(case, n=n, t_end=periods * T, samples=samples, transport="ab2", reinit_every=10,
              reinit_method="xi_sdf", curvature_kw=curv,
              viscous=CCDViscosity() if viscous == "ccd" else FD2Viscosity(), after_properties=hook)
    s = [d for d in r["series"] if key in d]
    saved, p3.T = p3.T, T
    try:
        beta, om, _ = p3.fit([d["t"] for d in s], [d[key] for d in s])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "umax": max(d.get("umax", 0.0) for d in r["series"]),
            "failed": bool(r["series"][-1].get("failed"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="viscous:mu pairs, e.g. fd2:sharp+rho")
    ap.add_argument("--case", default="drop", choices=("drop", "capwave"))
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        ref = DROP if it["case"] == "drop" else (-0.802955, 134.342861)
        print(f"{it['case']:7s} {it['viscous']:3s} mu {it['mu']:9s} N {it['n']:3d}: beta {r['beta']:+.2f} "
              f"({r['beta'] / ref[0]:.2f} x) omega {100 * (r['omega'] / ref[1] - 1):+.2f}% max|u| {r['umax']:.3f}"
              f"{' FAILED' if r['failed'] else ''}", flush=True)

    ap_items = [("ccd", "diffuse"), ("fd2", "diffuse"), ("fd2", "sharp"), ("fd2", "sharp+rho")]
    if args.only:
        ap_items = [tuple(x.split(":")) for x in args.only]
    items = [dict(viscous=v, mu=m, n=n, case=args.case) for n in (32, 64) for v, m in ap_items]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

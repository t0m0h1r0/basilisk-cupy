"""Is the CLS's excess damping of the oscillating drop (3.9 times the 2-D viscous mode, docs/notes/63) the diffuse
viscosity mu(psi) of the band? The paper's drop (water--air, 2 periods, ab2, xi-SDF every 10, height-function
curvature) with CCD or fd2 viscosity, and with the face viscosity made sharp: ``mu1`` or ``mu2`` by the side of
``psi = 1/2`` at the face (the mean of the two cells' phases on a cut face).

    python -m cases.cls_drop_viscosity [--jobs N]
"""

from __future__ import annotations

import argparse

DROP = (-0.179428, 59.554813)


def run(viscous: str, mu: str, n: int):
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity
    from cases.paper_benchmarks import run as bench

    hook = None
    if mu == "sharp":
        def hook(ns):
            tp = ns.tp
            for ax in range(2):
                hl, hr, cut, theta = tp._cut(ax)
                h = 0.5 * (hl.astype(float) + hr.astype(float))
                ns.mu[ax] = h * tp.mu1 + (1.0 - h) * tp.mu2
    T = 0.105460663082
    r = bench("drop", n=n, t_end=2 * T, samples=200, transport="ab2", reinit_every=10, reinit_method="xi_sdf",
              curvature_kw={"method": "heights", "heights_method": "fd2"},
              viscous=CCDViscosity() if viscous == "ccd" else FD2Viscosity(), after_properties=hook)
    s = [d for d in r["series"] if "D" in d]
    saved, p3.T = p3.T, T
    try:
        beta, om, _ = p3.fit([d["t"] for d in s], [d["D"] for d in s])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "umax": max(d.get("umax", 0.0) for d in r["series"]),
            "failed": bool(r["series"][-1].get("failed"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        print(f"drop {it['viscous']:3s} mu {it['mu']:7s} N {it['n']:3d}: beta {r['beta']:+.2f} "
              f"({r['beta'] / DROP[0]:.2f} x) omega {100 * (r['omega'] / DROP[1] - 1):+.2f}% max|u| {r['umax']:.3f}"
              f"{' FAILED' if r['failed'] else ''}", flush=True)

    items = [dict(viscous=v, mu=m, n=n) for n in (32, 64) for v, m in (("ccd", "diffuse"), ("fd2", "diffuse"),
                                                                       ("fd2", "sharp"))]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

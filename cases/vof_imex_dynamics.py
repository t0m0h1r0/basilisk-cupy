"""VOF on the IMEX steppers (``TwoPhaseIMEX(interface="vof" | "vof-gfm")``, docs/notes/64) on the paper's capillary
wave (water--air, normal mode) and oscillating drop (2-D viscous mode), against Basilisk's VOF on ``CenteredNS``.

    python -m cases.vof_imex_dynamics [--jobs N]
"""

from __future__ import annotations

import argparse

CW = (-0.802955, 134.342861)
DROP = (-0.179428, 59.554813)


def run(case: str, variant: str, n: int, backend: str = "numpy"):
    """``variant``: a base name, then ``+ccdk`` (``vof_curvature="ccd"``), ``+pccd`` (``faces="phase-ccd"``) and/or
    ``+r<k>`` (``vof_refine=k``)."""
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from cases.paper_benchmarks import run as bench

    variant, *extra = variant.split("+")
    cmin = None
    if variant.startswith("centered-conserving-ccd") and variant[-4:-2] == "-c":  # "-c<NN>": conserving_ccd_cmin
        variant, cmin = variant[:-4], int(variant[-2:]) / 100
    opts = {"centered-vof": dict(method="vof"),
            "centered-conserving": dict(method="vof", conserving=True),
            "centered-conserving-ccd": dict(method="vof", conserving=True, conserving_gradient="ccd"),
            "centered-conserving-ccd2": dict(method="vof", conserving=True, conserving_gradient="ccd2"),
            "imex-vof": dict(interface="vof"),
            "imex-vof-ab2": dict(interface="vof", transport="ab2"),
            "imex-vof-gfm": dict(interface="vof-gfm", transport="ab2"),
            "imex-vof-gfm-plic": dict(interface="vof-gfm", transport="ab2", vof_crossing="plic")}[variant.removesuffix("-fd2v")]
    if cmin is not None:
        opts = dict(opts, conserving_ccd_cmin=cmin)
    if variant.startswith("imex"):  # "-fd2v": the fd2 viscous operator (the VOF default since notes/64), else CCD
        from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity

        opts = dict(opts, viscous=FD2Viscosity() if variant.endswith("-fd2v") else CCDViscosity())
    for e in extra:
        if e == "ccdk":
            opts = dict(opts, vof_curvature="ccd")
        elif e == "pccd":  # phase-wise CCD faces (the transport velocity reprojected with fd2's, notes/67)
            opts = dict(opts, faces="phase-ccd")
        elif e.startswith("r"):
            opts = dict(opts, vof_refine=int(e[1:]))
    hook = None
    if "notaylor" in extra:  # phase-ccd without the cut faces' Taylor terms (docs/notes/67)
        def hook(ns):
            ns.ops.taylor = False
    if case == "capwave":
        T = 0.046742983863
        r = bench("capwave", n=n, t_end=5 * T, samples=500, backend=backend, after_properties=hook, **opts)
        key = "amplitude"
    else:
        T = 0.105460663082
        r = bench("drop", n=n, t_end=2 * T, samples=200, backend=backend, after_properties=hook, **opts)
        key = "D"
    s = [d for d in r["series"] if key in d]
    saved, p3.T = p3.T, T
    try:
        beta, om, res = p3.fit([d["t"] for d in s], [d[key] for d in s])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "res": res, "failed": bool(r["series"][-1].get("failed")),
            "umax": max(d.get("umax", 0.0) for d in r["series"]), "steps": r["steps"], "wall": r["wall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n", type=int, nargs="*", default=[32, 64])
    ap.add_argument("--cases", nargs="*", default=["capwave", "drop"])
    ap.add_argument("--backend", default="numpy", help="cupy: the runs one after another on the GPU")
    ap.add_argument("--variants", nargs="*", default=["centered-vof", "imex-vof", "imex-vof-ab2", "imex-vof-gfm",
                                                       "imex-vof-gfm-plic"])
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        ref = CW if it["case"] == "capwave" else DROP
        print(f"{it['case']:7s} {it['variant']:17s} N {it['n']:3d}: beta {r['beta']:+.2f} ({r['beta'] / ref[0]:.2f} x) "
              f"omega {100 * (r['omega'] / ref[1] - 1):+.2f}% max|u| {r['umax']:.3f}{' FAILED' if r['failed'] else ''}"
              f" ({r['steps']} steps, {r['wall']:.0f}s)", flush=True)

    items = [dict(case=c, variant=v, n=n, backend=args.backend) for c in args.cases for n in args.n
             for v in args.variants]
    if args.backend == "cupy":
        for k, it in enumerate(items):
            show(k, it, run(**it))
    else:
        pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

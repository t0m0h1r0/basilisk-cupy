"""The coefficient of the cell-centred velocity correction on cut faces (``cell_coefficient``, docs/notes/65) for
the CLS: the paper's capillary wave (5 periods) and oscillating drop (2 periods), water--air, ab2 transport, xi-SDF
every 10 steps, fd2 faces, psi or height-function curvature.

    python -m cases.cls_cell_coefficient [--jobs N]
"""

from __future__ import annotations

import argparse

CW = (-0.802955, 134.342861)
DROP = (-0.179428, 59.554813)
CURV = {"psi": None, "heights": {"method": "heights", "heights_method": "fd2"}, "phi": {"source": "phi"}}


def run(case: str, cell: str, curvature: str, n: int, faces: str = "fd2", ratio1: bool = False,
        taylor: bool = True):
    """``cell``: "gfm", "mean", "cut", or "w<weight>" (the cells' coefficient ``w alpha_mean + (1 - w) alpha_gfm``,
    a diagnostic blend of the two, applied by rescaling "mean"'s factors)."""
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from cases.paper_benchmarks import run as bench

    w = float(cell[1:]) if cell.startswith("w") else None
    opts = dict(faces=faces, transport="ab2", reinit_every=10, reinit_method="xi_sdf",
                cell_coefficient="mean" if w is not None else cell, curvature_kw=CURV[curvature])
    ref = None
    if ratio1:  # equal densities and viscosities (water), against that pair's normal mode
        import math

        from cases.paper_benchmarks import WATER_AIR
        from cases.two_fluid_modes import mode

        opts.update(rho2=WATER_AIR["rho1"], mu2=WATER_AIR["mu1"])
        sm = mode(2 * math.pi / 0.01, WATER_AIR["sigma"], WATER_AIR["rho1"], WATER_AIR["mu1"], WATER_AIR["rho1"],
                  WATER_AIR["mu1"])
        ref = (sm.real, sm.imag)
    if case == "capwave":
        T, key, periods, samples = 0.046742983863, "amplitude", 5, 500
        if ref is not None:
            T = 2 * math.pi / ref[1]
    else:
        T, key, periods, samples = 0.105460663082, "D", 2, 200
    hook = None
    if w is not None:
        def hook(ns):  # blend after TwoPhaseCLS._properties set g_scale = alpha_mean / alpha_gfm
            ns.g_scale = [1.0 + w * (r - 1.0) for r in ns.g_scale]
    elif not taylor:
        def hook(ns):  # phase-ccd without the cut faces' Taylor terms (docs/notes/67)
            ns.ops.taylor = False
    if faces.endswith("-fd2t"):  # phase-ccd with fd2's cell <-> face transfers (docs/notes/67)
        opts["faces"] = "phase-ccd"
        import basilisk_cupy.time.two_phase_imex as tpi
        from basilisk_cupy.discretization.ccd import FCCDFaceComplex
        from basilisk_cupy.discretization.phase_ccd import PhaseCCDFaces

        tpi._face_ops = lambda f: None if f == "fd2" else PhaseCCDFaces(FCCDFaceComplex(), transfer="fd2")
    r = bench(case, n=n, t_end=periods * T, samples=samples, after_properties=hook, **opts)
    s = [d for d in r["series"] if key in d]
    saved, p3.T = p3.T, T
    try:
        beta, om, res = p3.fit([d["t"] for d in s], [d[key] for d in s])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "ref": ref, "failed": bool(r["series"][-1].get("failed")),
            "umax": max(d.get("umax", 0.0) for d in r["series"]), "steps": r["steps"], "wall": r["wall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--cells", nargs="*", default=["gfm", "mean", "cut"])
    ap.add_argument("--no-psi-drop", action="store_true")
    ap.add_argument("--faces", default="fd2")
    ap.add_argument("--ratio1", action="store_true", help="equal densities (the capillary wave only)")
    ap.add_argument("--capwave-only", action="store_true")
    ap.add_argument("--notaylor", action="store_true", help="phase-ccd without the cut faces' Taylor terms")
    ap.add_argument("--curvatures", nargs="*", default=None, help="the drop's curvatures (default psi, heights)")
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        ref = r.get("ref") or (CW if it["case"] == "capwave" else DROP)
        print(f"{it['case']:7s} {it['faces']:9s}{'' if it.get('taylor', True) else '-notaylor'} {it['cell']:4s} {it['curvature']:7s} N {it['n']:3d}: beta {r['beta']:+.2f} "
              f"({r['beta'] / ref[0]:.2f} x) omega {100 * (r['omega'] / ref[1] - 1):+.2f}% max|u| {r['umax']:.3f}"
              f"{' FAILED' if r['failed'] else ''} ({r['steps']} steps, {r['wall']:.0f}s)", flush=True)

    cases_ = [("capwave", "psi"), ("drop", "heights")] if args.no_psi_drop else \
        [("capwave", "psi"), ("drop", "psi"), ("drop", "heights")]
    if args.curvatures:
        cases_ = [("capwave", args.curvatures[0])] + [("drop", k) for k in args.curvatures]
    if args.ratio1 or args.capwave_only:
        cases_ = [c for c in cases_ if c[0] == "capwave"]
    items = [dict(case=c, cell=cell, curvature=k, n=n, faces=args.faces, ratio1=args.ratio1,
                  taylor=not args.notaylor) for n in (32, 64) for cell in args.cells for c, k in cases_]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

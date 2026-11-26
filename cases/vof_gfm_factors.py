"""Why the VOF ghost-fluid jump (``interface="vof-gfm"``) is less accurate than tension.h's CSF on moving
interfaces (docs/notes/64): the paper's capillary wave (5 periods, ab2, fd2 viscosity) with

* ``coef-mean``: the jump kept sharp, but two-phase.h's face coefficient ``1/rho(f_face)`` instead of the GFM's;
* ``jump-mean``: the jump as iforce.h's mean of the two cells' potentials, not theta-weighted;
* ``g-mean``: the GFM projection of the face velocity, but the cell-centred correction with two-phase.h's coefficient;
* ``ratio1``: equal densities and viscosities (water), against the normal mode of that pair.

    python -m cases.vof_gfm_factors [--jobs N]
"""

from __future__ import annotations

import argparse
import math


def cell_coefficient(ns, rest) -> None:
    """``g-mean``: the cell-centred correction with two-phase.h's coefficient (``cell_coefficient="mean"``, the
    default since docs/notes/65); otherwise the GFM's, as in docs/notes/64."""
    if hasattr(ns.tp, "cell_coefficient"):
        ns.tp.cell_coefficient = "mean" if "g-mean" in rest else "gfm"


def run_drop(variant: str, n: int):
    """The paper's oscillating drop (2 periods) against the 2-D viscous mode."""
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from cases.paper_benchmarks import diagnostics, setup

    force, *rest = variant.split("+")
    ns, info = setup("drop", n, interface="vof" if force == "csf" else "vof-gfm", transport="ab2")
    cell_coefficient(ns, rest)
    T = 0.105460663082
    times = [2 * T * (i + 1) / 200 for i in range(200)]
    ns.start(events=times)
    series = [diagnostics("drop", ns, info)]
    j = 0
    while j < len(times):
        ns.step()
        if abs(ns.clock.t - times[j]) <= 1e-9:
            series.append(diagnostics("drop", ns, info))
            j += 1
    saved, p3.T = p3.T, T
    try:
        beta, om, _ = p3.fit([d["t"] for d in series], [d["D"] for d in series])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "ref": (-0.179428, 59.554813), "umax": max(d["umax"] for d in series)}


def run(variant: str, n: int):
    import warnings

    warnings.filterwarnings("ignore")
    import cases.phase3_capwave as p3
    from cases.paper_benchmarks import WATER_AIR, setup
    from cases.two_fluid_modes import mode

    force, *rest = variant.split("+")
    kw = dict(interface="vof" if force == "csf" else "vof-gfm", transport="ab2")
    if "ratio1" in rest:
        kw.update(rho2=WATER_AIR["rho1"], mu2=WATER_AIR["mu1"])
    ns, info = setup("capwave", n, **kw)
    if "coef-mean" in rest:
        ns.tp.coefficient = "mean"
    if "jump-mean" in rest:
        ns.tp.jump_weights = "mean"
    cell_coefficient(ns, rest)
    k = info["k"]
    props = dict(WATER_AIR, **{key: kw[key] for key in ("rho2", "mu2") if key in kw})
    s = mode(k, props["sigma"], props["rho1"], props["mu1"], props["rho2"], props["mu2"])
    T = 2 * math.pi / s.imag
    from cases.paper_benchmarks import diagnostics

    times = [5 * T * (i + 1) / 500 for i in range(500)]
    ns.start(events=times)
    series = [diagnostics("capwave", ns, info)]
    j = 0
    while j < len(times):
        ns.step()
        if abs(ns.clock.t - times[j]) <= 1e-9:
            series.append(diagnostics("capwave", ns, info))
            j += 1
    saved, p3.T = p3.T, T
    try:
        beta, om, _ = p3.fit([d["t"] for d in series], [d["amplitude"] for d in series])
    finally:
        p3.T = saved
    return {"beta": beta, "omega": om, "ref": (s.real, s.imag), "umax": max(d.get("umax", 0.0) for d in series)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--drop", action="store_true", help="the oscillating drop instead of the capillary wave")
    ap.add_argument("--variants", nargs="*", default=["csf", "gfm", "gfm+coef-mean", "csf+ratio1", "gfm+ratio1"])
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        b0, w0 = r["ref"]
        print(f"{it['variant']:20s} N {it['n']:3d}: beta {r['beta']:+.2f} (ref {b0:+.3f}) omega "
              f"{100 * (r['omega'] / w0 - 1):+.2f}% max|u| {r['umax']:.3f}", flush=True)

    items = [dict(variant=v, n=n) for n in (32, 64)
             for v in args.variants]
    pmap(run_drop if args.drop else run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

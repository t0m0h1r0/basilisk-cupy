"""VOF on the IMEX steppers at N = 64 (docs/notes/64): is tension.h's capillary step too long for SBDF2? The paper's
capillary wave (CSF) and drop (ghost-fluid jump) with the largest step halved, with first-order SBDF1, or with the
fd2 viscous operator instead of CCD (the VOF viscosity jumps at the interface).

    python -m cases.vof_imex_stability [--jobs N]
"""

from __future__ import annotations

import argparse


def run(case: str, interface: str, n: int = 64, DT: float | None = None, scheme: str = "sbdf2",
        viscous: str = "ccd"):
    import math
    import warnings

    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import setup

    kw = dict(interface=interface, transport="ab2", scheme=scheme)
    if DT:
        kw["DT"] = DT
    from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity

    kw["viscous"] = FD2Viscosity() if viscous == "fd2" else CCDViscosity()  # the VOF default is fd2 since notes/64
    ns, info = setup(case, n, **kw)
    T = info["T"]
    ns.start()
    hist = []
    while ns.clock.t < 1.0 * T:
        ns.step()
        u = ns.max_velocity()
        if ns.clock.i % 50 == 0:
            hist.append((ns.clock.i, round(ns.clock.t / T, 3), round(u, 4)))
        if not math.isfinite(u) or u > 3:
            hist.append((ns.clock.i, round(ns.clock.t / T, 3), u))
            break
    return {"steps": ns.clock.i, "dt": ns.clock.dt, "umax": max(h[2] for h in hist), "end": hist[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--only-viscous", action="store_true")
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(case="capwave", interface="vof"), dict(case="capwave", interface="vof", DT=1.2e-4),
             dict(case="capwave", interface="vof", scheme="sbdf1"),
             dict(case="drop", interface="vof-gfm"), dict(case="drop", interface="vof-gfm", DT=1.2e-4),
             dict(case="drop", interface="vof-gfm", scheme="sbdf1")]
    items += [dict(case="capwave", interface="vof", viscous="fd2"), dict(case="drop", interface="vof-gfm", viscous="fd2")]
    if args.only_viscous:
        items = items[-2:]
    pmap(run, items, args.jobs, on_result=lambda k, it, r: print(it, r, flush=True))


if __name__ == "__main__":
    main()

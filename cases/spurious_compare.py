"""Basilisk's parasitic-current test (``src/test/spurious.c``, Popinet 2009): a quarter drop of diameter 0.8 at rest
in [0, 1]^2 (symmetry), constant density 1, viscosity from the Laplace number ``La = sigma rho D / mu^2``, Stokes flow
(no advection), run to the viscous time ``D^2/mu`` or until the volume fraction changes by less than ``DC`` per step.
Outputs, as Basilisk's ``log``: LEVEL, La, max|u| sqrt(D), the shape error (avg, rms, max) and the largest curvature
error. Against Basilisk's C build on a uniform grid (``tools/basilisk/spurious_uniform.c``).

    python -m cases.spurious_compare [--basilisk <log of spurious_uniform>] [--jobs N]
"""

from __future__ import annotations

import argparse
import math

D = 0.8


def run(level: int, laplace: float, dc: float = 0.0, interface: str = "vof", curvature: str = "basilisk"):
    """``interface``: "vof" (Basilisk's CenteredNS + VOF + tension.h), "imex-vof" / "imex-vof-gfm" (TwoPhaseIMEX)."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.curvature import curvature as hf_curvature
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    n = 1 << level
    mu = math.sqrt(D / laplace)
    tmax = D * D / mu
    grid = CartesianGrid((n, n))
    circle = lambda x, y: (D / 2) ** 2 - x * x - y * y  # noqa: E731
    if interface == "vof":
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, stokes=True,
                        tolerance=1e-6, DT=1e30)
        c = CellField.from_interior(grid, fraction(grid, circle), BoundaryConditions())
        tp = TwoPhase(ns, c, rho1=1.0, rho2=1.0, mu1=mu, mu2=mu)
        tp.add_surface_tension(1.0, curvature_method=curvature)
    else:
        from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

        ns = TwoPhaseIMEX(grid, circle, rho1=1.0, rho2=1.0, mu1=mu, mu2=mu, sigma=1.0, tolerance=1e-6,
                          interface=interface.removeprefix("imex-"), vof_curvature=curvature, stokes=True)
        c = ns.vof
    ns.start(events=[tmax])
    cn = np.asarray(c.interior).copy()
    hist = []
    while ns.clock.t < tmax - 1e-12:
        ns.step()
        cc = np.asarray(c.interior)
        change = float(np.abs(cc - cn).max())
        cn = cc.copy()
        un = np.sqrt(sum(np.asarray(u.interior) ** 2 for u in ns.u))
        hist.append((mu * float(ns.clock.t) / D ** 2, float(un.max()) * math.sqrt(D), change))
        if ns.clock.i > 1 and change < dc:
            break
    vol = float(cc.sum()) / n / n
    radius = math.sqrt(4.0 * vol / math.pi)
    cref = np.asarray(fraction(grid, circle))
    ec = cc - cref
    k = np.asarray(hf_curvature(c.interior, c.bc, grid, 1.0))
    ek = np.abs(k[k != NODATA] - 1.0 / radius)
    return {"level": level, "La": laplace, "umax": hist[-1][1], "ec_avg": float(np.abs(ec).mean()),
            "ec_rms": float(np.sqrt((ec ** 2).mean())), "ec_max": float(np.abs(ec).max()),
            "ekmax": float(ek.max()), "steps": ns.clock.i, "tau": hist[-1][0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basilisk", default=None, help="the log (stderr) of tools/basilisk/spurious_uniform.c")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--interfaces", nargs="*", default=["vof"])
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(level=5, laplace=la) for la in (120.0, 1200.0, 12000.0)]
    items += [dict(level=lv, laplace=12000.0, dc=1e-10) for lv in (3, 4, 6, 7)]
    items = [dict(it, interface=i) for i in args.interfaces for it in items]
    ref = {}
    if args.basilisk:
        import numpy as np

        for row in np.atleast_2d(np.loadtxt(args.basilisk)):
            ref[(int(row[0]), float(row[1]))] = row[2:]

    def show(k, it, r):
        line = (f"{it['interface']:12s} L {r['level']} La {r['La']:6g}: max|u|sqrt(D) {r['umax']:.3e} ec avg "
                f"{r['ec_avg']:.3e} rms {r['ec_rms']:.3e} max {r['ec_max']:.3e} ekmax {r['ekmax']:.3e} "
                f"({r['steps']} steps, tau {r['tau']:.3f})")
        b = ref.get((r["level"], r["La"]))
        if b is not None:
            line += f" | C: {b[0]:.3e} {b[1]:.3e} {b[2]:.3e} {b[3]:.3e} {b[4]:.3e}"
        print(line, flush=True)

    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

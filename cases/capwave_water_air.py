"""Stability of the paper's capillary wave (water--air, 32^2, density ratio 829) against the step and the method
(docs/notes/57). The step is ``f`` times tension.h's bound ``sqrt(rho_m h^3 / (pi sigma))``, ``rho_m`` the mean
density; runs to 5 periods (or blow-up), recording max |u| per period.

    python -m cases.capwave_water_air [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS

RHO1, RHO2, MU1, MU2, SIGMA = 998.2, 1.204, 1.002e-3, 1.825e-5, 0.0728
T = 0.046742983863
L, A0, Y0 = 0.02, 2e-4, 0.01
K = 2 * math.pi / 0.01


def bound(n: int) -> float:
    h = L / n
    return math.sqrt(0.5 * (RHO1 + RHO2) * h ** 3 / (math.pi * SIGMA))


def build(method: str, n: int, dt: float, ratio: float | None = None, variant: str = ""):
    grid = CartesianGrid((n, n), length=(L, L))
    x, y = grid.meshgrid()
    eta = Y0 + A0 * np.cos(K * x)
    phi = (eta - y) / np.sqrt(1.0 + (A0 * K * np.sin(K * x)) ** 2)
    walls = {"bottom": 0.0, "top": 0.0}
    props = dict(rho1=RHO1, rho2=RHO2 if ratio is None else RHO1 / ratio, mu1=MU1, mu2=MU2)
    if method.startswith("imex") or method.startswith("rk"):
        from cases.paper_benchmarks import _stepper

        kw = dict(props, sigma=SIGMA, periodic=(0,), fixed_dt=dt, capillary_dt_factor=1e6)
        if method == "imex-ab2":
            kw["transport"] = "ab2"
        if method.startswith("rk"):
            kw["tableau"] = method.split("-")[1]
        return _stepper(grid, phi, kw)
    ns = CenteredNS(grid, periodic=(0,), uf_normal=walls, tolerance=1e-8, DT=dt,
                    viscous=None if method == "vof" else CCDViscosity(),
                    **({"coupled_iterations": 30, "coupled_rtol": 1e-6, "coupled_on_failure": "ignore"}
                       if method == "centered-coupled" else {}))
    if method == "vof":
        bc = BoundaryConditions()
        from basilisk_cupy.boundary import Periodic

        bc.set((0, 0), Periodic()).set((0, 1), Periodic())
        f = CellField.from_interior(grid, fraction(grid, lambda x, y: Y0 + A0 * np.cos(K * x) - y), bc)
        tp = TwoPhase(ns, f, **props)
    else:
        from basilisk_cupy.boundary import Neumann, Periodic

        bc = BoundaryConditions()
        bc.set((0, 0), Periodic()).set((0, 1), Periodic()).set((1, 0), Neumann(0.0)).set((1, 1), Neumann(0.0))
        cls = CLSInterface.from_level_set(grid, phi, bc, divergence="fccd")
        extra = {}
        if "heights" in variant:
            extra["curvature_kw"] = {"method": "heights", "heights_method": "fd2"}
        import re as _re

        m = _re.search(r"reinit(\d*)-?(xi_sdf|compression)?", variant)
        if m:
            extra.update(reinit_every=int(m.group(1) or 10), reinit_method=m.group(2) or "compression")
        tp = TwoPhaseCLS(ns, cls, pressure_jump="csf" not in variant, capillary_dt_factor=1e6, **props, **extra)
    tp.add_surface_tension(SIGMA)
    ns.cfl = 0.5
    return ns


def run(method: str, f: float, n: int = 32, periods: int = 5, ratio: float | None = None, variant: str = ""):
    warnings.filterwarnings("ignore")
    dt = f * bound(n)
    ns = build(method, n, dt, ratio, variant)
    times = [T * (k + 1) for k in range(periods)]
    ns.start(events=times)
    per_period, umax, k, t0 = [], 0.0, 0, time.time()
    blew = None
    while k < periods:
        ns.step()
        u = max(float(np.abs(c.interior).max()) for c in ns.u)
        umax = max(umax, u)
        if not math.isfinite(u) or u > 1.0:
            blew = ns.clock.t / T
            break
        if abs(ns.clock.t - times[k]) <= 1e-9:
            per_period.append(umax)
            umax, k = 0.0, k + 1
        if time.time() - t0 > 3000:
            blew = -1.0  # timed out
            break
    return {"per_period": per_period, "blew_up_at_period": blew, "steps": ns.clock.i, "dt": dt,
            "wall": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--isolate", action="store_true", help="CenteredNS + CLS variants at f = sqrt(5/12)")
    ap.add_argument("--reinit-scan", action="store_true", help="with --isolate: re-initialisation method and period")
    args = ap.parse_args()
    if args.isolate:
        from cases.parallel import pmap

        f = math.sqrt(5 / 12)
        items = [dict(method="centered", f=f, ratio=r) for r in (1.0, 10.0, 100.0)]
        items += [dict(method="centered", f=f, variant=v) for v in ("heights", "reinit", "csf", "heights+reinit")]
        items += [dict(method="vof", f=f, ratio=r) for r in (1.0, 10.0, 100.0)]
        if args.reinit_scan:
            items = [dict(method="centered", f=f, variant=f"reinit{k}-{m}") for m in ("compression", "xi_sdf")
                     for k in (1, 5, 10, 50)]
        for it, r in zip(items, pmap(run, items, args.jobs)):
            pp = " ".join(f"{u:.2e}" for u in r["per_period"])
            state = "stable" if r["blew_up_at_period"] is None else f"BLEW UP at {r['blew_up_at_period']:.2f} T"
            print(f"{it['method']:9s} ratio {it.get('ratio') or 829.1:6.1f} {it.get('variant', ''):15s}: {state:20s} "
                  f"max|u| per period: {pp}")
        return
    from cases.parallel import pmap

    methods = ("imex", "imex-ab2", "rk-ars222", "centered", "centered-coupled", "vof")
    factors = (1.0, math.sqrt(5 / 12), 0.5, 0.35, 0.25)
    items = [dict(method=m, f=f) for m in methods for f in factors]
    print(f"tension.h bound {bound(32):.3e} s; period {T} s")
    for it, r in zip(items, pmap(run, items, args.jobs)):
        pp = " ".join(f"{u:.2e}" for u in r["per_period"])
        state = "stable" if r["blew_up_at_period"] is None else (
            "timeout" if r["blew_up_at_period"] < 0 else f"BLEW UP at {r['blew_up_at_period']:.2f} T")
        print(f"{it['method']:17s} f {it['f']:.3f}: {state:22s} max|u| per period: {pp}  ({r['steps']} steps, "
              f"{r['wall']:.0f}s)")


if __name__ == "__main__":
    main()

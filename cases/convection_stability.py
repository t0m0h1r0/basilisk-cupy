"""Stability of the convection treatments on a convection-dominated two-phase flow (docs/notes/72): Basilisk's
Rayleigh--Taylor test (``src/test/rt.c``: rho 1.225 / 0.1694, [-2, 2]^2, walls, DT = 5e-3) with reduced gravity, to
t = 1, at rt.c's viscosity and a tenth of it, N = 64 and 128:

* ``centered-vof``: Basilisk's CenteredNS + VOF (BCG advection);
* ``centered-conserving``: the same with momentum-conserving VOF advection (``conserving.h``); ``-ccd``: the momentum
  concentrations' gradient by CCD along each phase's runs of cells (docs/notes/72);
* ``imex-central`` / ``imex-uccd6`` / ``imex-fuccd``: TwoPhaseIMEX's VOF with the non-dissipative central flux, UCCD6
  or FUCCD4 (upwind sigma |a| h^7 d^8 dissipation); ``@sbdf3`` etc. selects the multistep scheme (SBDF2 by default).

Kinetic energy, largest velocity, mass and the range of f every 0.05; a run stops when |u| exceeds 50.

    python -m cases.convection_stability [--jobs N] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import math

RHO1, RHO2, MU = 1.225, 0.1694, 0.00313


def conserving_options(variant: str) -> dict:
    """TwoPhase's options from a variant name: "conserving" (conserving.h), then "-ccd" / "-ccd2" (the gradient of the
    momentum concentrations) and "-c<NN>" (``conserving_ccd_cmin = NN/100``, e.g. "-c99")."""
    import re

    m = re.search(r"-c(\d\d)$", variant)
    v = variant[: m.start()] if m else variant
    grad = "ccd2" if v.endswith("-ccd2") else "ccd" if v.endswith("-ccd") else None
    return dict(conserving="conserving" in v, conserving_gradient=grad,
                conserving_ccd_cmin=int(m.group(1)) / 100 if m else None)


def shear(variant: str, n: int, ratio: float, t_end: float = 2.0, delta: float = 1.0 / 30):
    """Two-phase double shear layer without forcing (periodic [0, 1]^2, inviscid, no gravity or tension): a heavy band
    0.25 < y < 0.75 (density ``ratio`` times the light fluid's) carried by u = tanh((y - 1/4)/delta) below y = 1/2 and
    tanh((3/4 - y)/delta) above, v = 0.05 sin(2 pi x). The kinetic energy cannot grow physically."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    band = lambda x, y: np.minimum(y - 0.25, 0.75 - y)  # noqa: E731
    rho1, rho2 = 1.0, 1.0 / ratio
    if variant.startswith("centered"):
        from basilisk_cupy.interface.two_phase import TwoPhase

        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-6)
        from basilisk_cupy.boundary import Periodic

        bc = BoundaryConditions()
        for ax in range(2):
            for side in (0, 1):
                bc.set((ax, side), Periodic())
        f = CellField.from_interior(grid, fraction(grid, band), bc)
        tp = TwoPhase(ns, f, rho1=rho1, rho2=rho2, **conserving_options(variant))
        rho = lambda: tp.rho(f.interior)  # noqa: E731
    else:
        from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

        conv, _, scheme = variant.removeprefix("imex-").partition("@")  # "imex-central@sbdf3": the multistep scheme
        ns = TwoPhaseIMEX(grid, band, rho1=rho1, rho2=rho2, sigma=0.0, periodic=(0, 1), interface="vof",
                          convection=conv, transport="ab2", tolerance=1e-6, scheme=scheme or "sbdf2")
        f = ns.vof
        rho = lambda: ns.tp.rho(f.interior)  # noqa: E731
    x, y = (to_numpy(a) for a in grid.meshgrid())
    u0 = np.where(y <= 0.5, np.tanh((y - 0.25) / delta), np.tanh((0.75 - y) / delta))
    ns.u[0].interior = grid.xp.asarray(u0)
    ns.u[1].interior = grid.xp.asarray(0.05 * np.sin(2.0 * math.pi * x))
    times = [0.1 * (k + 1) for k in range(int(round(t_end / 0.1)))]
    ns.start(events=times)
    dv = float(grid.h[0] * grid.h[1])
    series, k, failed = [], 0, False
    while k < len(times):
        ns.step()
        umax = max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u)
        if not math.isfinite(umax) or umax > 20.0:
            failed = True
            series.append(dict(t=float(ns.clock.t), umax=umax, failed=True))
            break
        if abs(ns.clock.t - times[k]) < 1e-9:
            ke = 0.5 * float(np.sum(to_numpy(rho()) * sum(to_numpy(c.interior) ** 2 for c in ns.u))) * dv
            series.append(dict(t=times[k], ke=ke, umax=umax))
            k += 1
    ke0 = 0.5 * float(np.sum(np.where(band(x, y) > 0, rho1, rho2) * (u0 ** 2 + (0.05 * np.sin(2 * math.pi * x)) ** 2))) * dv
    return {"variant": variant, "n": n, "ratio": ratio, "failed": failed, "steps": ns.clock.i, "ke0": ke0,
            "series": series}


def translate(variant: str, n: int, ratio: float, t_end: float = 1.0, sigma: float = 0.01, mu: float = 1e-4,
              speed: float = 1.0, kappa_every: int = 0):
    """A drop (R = 0.15, density 1, the outer fluid 1/ratio) carried with the uniform velocity (``speed``, 0) across
    the periodic [0, 1]^2 for one period: exactly a translation, so the velocity error max|u - (speed, 0)| and the
    shape error |f - f0| after t = 1 measure the momentum advection (Galilean invariance); ``speed = 0``: the same drop
    at rest (the static spurious currents). Variant suffix "+kccd": ``curvature_method="ccd"``. ``kappa_every``: every
    so many steps, the relative curvature error |kappa R| - 1 in the interfacial cells (max and RMS, means over the
    samples in ``kappa_max`` / ``kappa_rms``)."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    disc = lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2  # noqa: E731
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-8)
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    f0 = fraction(grid, disc, refine=8)
    f = CellField.from_interior(grid, f0, bc)
    base, _, kmethod = variant.partition("+")
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, mu1=mu, mu2=mu, **conserving_options(base))
    tp.add_surface_tension(sigma, curvature_method="ccd" if kmethod == "kccd" else "basilisk")
    ns.u[0].interior = speed + 0.0 * ns.u[0].interior
    ns.start(events=[t_end])
    failed = False
    errs = []  # max|u - U| after every step (the spurious currents fluctuate: the final value alone is noisy)
    kmax, krms = [], []
    from basilisk_cupy.interface.heights import NODATA

    while ns.clock.t < t_end - 1e-12:
        ns.step()
        if not max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u) < 10.0:
            failed = True
            break
        errs.append(max(float(np.abs(to_numpy(ns.u[0].interior) - speed).max()),
                        float(np.abs(to_numpy(ns.u[1].interior)).max())))
        if kappa_every and ns.clock.i % kappa_every == 0:
            phi = to_numpy(tp.potentials[0](tp))
            F = to_numpy(f.interior)
            sel = (phi != NODATA) & (F > 1e-6) & (F < 1 - 1e-6)
            e = np.abs(phi[sel]) / sigma * 0.15 - 1.0
            kmax.append(float(np.abs(e).max()))
            krms.append(float(np.sqrt(np.mean(e ** 2))))
    du = errs[-1] if errs else math.nan
    return {"variant": variant, "n": n, "ratio": ratio, "failed": failed, "du": du,
            "du_max": max(errs, default=math.nan), "du_mean": float(np.mean(errs)) if errs else math.nan,
            "kappa_max": float(np.mean(kmax)) if kmax else math.nan,
            "kappa_rms": float(np.mean(krms)) if krms else math.nan,
            "shape": float(np.abs(to_numpy(f.interior) - to_numpy(f0)).sum()) / n / n, "steps": ns.clock.i}


def run(variant: str, n: int, mu: float, t_end: float = 1.0):
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n), length=(4.0, 4.0), origin=(-2.0, -2.0))
    level = lambda x, y: 0.05 * np.cos(2.0 * math.pi * x) + y  # noqa: E731
    G = ((0.0, -9.81), (0.0, 0.0))
    if variant.startswith("centered"):
        from basilisk_cupy.interface.two_phase import TwoPhase

        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-6, DT=5e-3)
        f = CellField.from_interior(grid, fraction(grid, level), BoundaryConditions())
        tp = TwoPhase(ns, f, rho1=RHO1, rho2=RHO2, mu1=mu, mu2=mu, **conserving_options(variant))
        tp.add_reduced_gravity(*G)
        rho = lambda: tp.rho(f.interior)  # noqa: E731
    else:
        from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

        ns = TwoPhaseIMEX(grid, level, rho1=RHO1, rho2=RHO2, mu1=mu, mu2=mu, sigma=0.0, gravity=G, interface="vof",
                          convection=variant.removeprefix("imex-"), transport="ab2", tolerance=1e-6, DT=5e-3)
        f = ns.vof
        rho = lambda: ns.tp.rho(f.interior)  # noqa: E731
    times = [0.05 * (k + 1) for k in range(int(round(t_end / 0.05)))]
    ns.start(events=times)
    dv = float(grid.h[0] * grid.h[1])
    m0 = float(np.sum(to_numpy(f.interior))) * dv
    series, k, failed = [], 0, False
    while k < len(times):
        ns.step()
        umax = max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u)
        if not math.isfinite(umax) or umax > 50.0:
            failed = True
            series.append(dict(t=float(ns.clock.t), umax=umax, failed=True))
            break
        if abs(ns.clock.t - times[k]) < 1e-9:
            fi = to_numpy(f.interior)
            ke = 0.5 * float(np.sum(to_numpy(rho()) * sum(to_numpy(c.interior) ** 2 for c in ns.u))) * dv
            series.append(dict(t=times[k], ke=ke, umax=umax, mass=float(fi.sum()) * dv / m0 - 1.0,
                               fmin=float(fi.min()), fmax=float(fi.max())))
            k += 1
    return {"variant": variant, "n": n, "mu": mu, "failed": failed, "steps": ns.clock.i, "series": series}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, nargs="*", default=[64, 128])
    ap.add_argument("--shear", action="store_true", help="the forcing-free two-phase double shear layer")
    ap.add_argument("--translate", action="store_true", help="a drop carried by a uniform flow (Galilean invariance)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="e.g. imex-central@sbdf3 imex-uccd6@cnlf (the multistep scheme after @)")
    args = ap.parse_args()
    from cases.parallel import pmap

    if args.translate:
        variants = args.variants or ("centered-vof", "centered-conserving", "centered-conserving-ccd",
                                     "centered-conserving-ccd2", "centered-conserving-ccd-c99",
                                     "centered-conserving-ccd2-c99")
        items = [dict(variant=v, n=n, ratio=r) for n in (128, 64, 32) for r in (10.0, 1000.0) for v in variants]
        pmap(translate, items, args.jobs, on_result=lambda k, it, r: print(
            f"translate {it['variant']:26s} N {it['n']:3d} ratio {it['ratio']:6g}: "
            + ("FAILED " if r["failed"] else "") + f"max|u - U| end {r['du']:.3e} over t {r['du_max']:.3e}"
            f" mean {r['du_mean']:.3e} shape L1 {r['shape']:.3e}"
            f" ({r['steps']} steps)", flush=True))
        return
    if args.shear:
        variants = args.variants or ("centered-vof", "centered-conserving", "imex-central", "imex-uccd6", "imex-fuccd")
        items = [dict(variant=v, n=n, ratio=r) for n in args.n for r in (10.0, 100.0) for v in variants]

        def show_shear(k, it, r):
            s = r["series"]
            ke = [d["ke"] for d in s if "ke" in d]
            print(f"shear {it['variant']:20s} N {it['n']:3d} ratio {it['ratio']:5g}: "
                  + (f"FAILED at t {s[-1]['t']:.2f} " if r["failed"] else "")
                  + (f"ke(t)/ke0 max {max(ke) / r['ke0']:.4f} end {ke[-1] / r['ke0']:.4f} (t {s[-1]['t']:.1f})"
                     if ke else "") + f" ({r['steps']} steps)", flush=True)

        results = pmap(shear, items, args.jobs, on_result=show_shear)
        if args.out:
            with open(args.out, "w") as fp:
                json.dump(results, fp)
        return

    variants = ("centered-vof", "centered-conserving", "imex-central", "imex-uccd6", "imex-fuccd")
    items = [dict(variant=v, n=n, mu=m) for n in args.n for m in (MU, MU / 10) for v in variants]

    def show(k, it, r):
        s = r["series"]
        last = s[-1]
        ke = [d["ke"] for d in s if "ke" in d]
        print(f"{it['variant']:20s} N {it['n']:3d} mu {it['mu']:.2e}: "
              + (f"FAILED at t {last['t']:.3f}" if r["failed"] else
                 f"t {last['t']:.2f} ke {last['ke']:.4f} (max {max(ke):.4f}) max|u| {last['umax']:.3f} "
                 f"mass {last['mass']:+.1e} f [{last['fmin']:.1e}, {last['fmax'] - 1:+.1e}]")
              + f" ({r['steps']} steps)", flush=True)

    results = pmap(run, items, args.jobs, on_result=show)
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(results, fp)


if __name__ == "__main__":
    main()

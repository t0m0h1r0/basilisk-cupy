"""docs/notes/122 (A-3 stage 1, condition 9 of #166): surface tension at the density ratio 1000, filtered=False, the
full configuration (cases/r0_combined.py, with and without E3) against the library as it is (same filtered=False).
Pass (fixed before the results): no blow-up; the parasitic currents, the frequency and the damping not worse than
the library's.

  static_drop()  R = 0.2 at the centre of the periodic unit square, sigma = 1, rho 1 / 1e-3, mu 1e-2 / (1e-2 / 55),
                 the curvature exact (sigma / R in the interfacial cells) or numerical (tension.h's heights):
                 max |u| over time (the parasitic currents) and at the end.
  capillary_wave()  y = 0.01 cos 2 pi x between slip walls at y = +-0.5 (periodic in x), heavy below, the same
                 properties: the amplitude (largest height-function position) in time; frequency from the zero
                 crossings, damping from the extrema (log decrement).

    python -m cases.r0_combined_tension     # docs/figures/r0_combined_tension.json
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import numpy as np

RHO = (1.0, 1e-3)
MU = (1e-2, 1e-2 / 55.0)


def _ctx(conf):
    from cases.r0_combined import configuration

    return configuration(conf) if conf != "base" else contextlib.nullcontext()


def static_drop(conf, n=32, curvature="numerical", t_end=1.0, R=0.2):
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    with _ctx(conf):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=MU[0])
        f0 = to_numpy(fraction(grid, lambda x, y: R * R - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=RHO[0], rho2=RHO[1],
                      mu1=MU[0], mu2=MU[1], conserving=True)
        tp.add_surface_tension(1.0)
        if curvature == "exact":
            numeric = tp.potentials[0]
            sign = np.sign(np.mean([v for v in to_numpy(numeric(tp)).ravel() if v < NODATA / 2]))

            def exact(t):
                fi = to_numpy(t.f.interior)
                return grid.xp.asarray(np.where((fi > 0) & (fi < 1), sign / R, NODATA))

            tp.potentials[0] = exact
        ns.start(events=[t_end])
        umax, hist, failed = 0.0, [], None
        try:
            while ns.clock.t < t_end - 1e-12:
                ns.step()
                u = float(np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)).max())
                umax = max(umax, u)
                hist.append((float(ns.clock.t), u))
                if not math.isfinite(u) or u > 1e2:
                    failed = "blow-up at t = %.4f" % ns.clock.t
                    break
        except Exception as exc:  # noqa: BLE001
            failed = repr(exc)[:200]
    return {"case": "static_drop", "conf": conf, "n": n, "curvature": curvature, "umax": umax,
            "u_end": hist[-1][1] if hist else None, "steps": len(hist), "failed": failed,
            "history": hist[:: max(1, len(hist) // 40)]}


def capillary_wave(conf, n=32, t_end=1.2, a0=0.01):
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Neumann, Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA, position
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    with _ctx(conf):
        grid = CartesianGrid((n, n), origin=(0.0, -0.5))
        ns = CenteredNS(grid, periodic=(0,), tolerance=1e-10, mu=MU[0])
        bc = BoundaryConditions()
        for side in (0, 1):
            bc.set((0, side), Periodic())
            bc.set((1, side), Neumann(0.0))
        f0 = to_numpy(fraction(grid, lambda x, y: a0 * math.cos(0.0) * np.cos(2 * math.pi * x) - y, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=RHO[0], rho2=RHO[1], mu1=MU[0], mu2=MU[1],
                      conserving=True)
        tp.add_surface_tension(1.0)
        samples = list(np.arange(1, int(t_end / 0.004) + 1) * 0.004)
        ns.start(events=samples)
        amp, failed = [], None
        try:
            for te in samples:
                while ns.clock.t < te - 1e-12:
                    ns.step()
                p = to_numpy(position(tp.f.interior, tp.f.bc, grid, (0.0, 1.0), (0.0, 0.0)))
                x = (np.arange(n) + 0.5) / n
                # the amplitude: the interface's height at the column x = 0 (cos = 1), from the cells there
                col = p[0, :]
                ok = col < NODATA / 2
                amp.append((te, float(col[ok].mean()) if ok.any() else float("nan")))
                del x
                if not math.isfinite(amp[-1][1]) or abs(amp[-1][1]) > 0.5:
                    failed = "lost at t = %.3f" % te
                    break
        except Exception as exc:  # noqa: BLE001
            failed = repr(exc)[:200]
    t = np.array([a[0] for a in amp])
    a = np.array([a[1] for a in amp])
    out = {"case": "capillary_wave", "conf": conf, "n": n, "failed": failed, "amplitude": amp[::5]}
    if len(a) > 20 and failed is None:
        zc = [t[k] - a[k] * (t[k + 1] - t[k]) / (a[k + 1] - a[k]) for k in range(len(a) - 1) if a[k] * a[k + 1] < 0]
        out["period"] = float(2 * np.mean(np.diff(zc))) if len(zc) > 2 else None
        ext = [k for k in range(1, len(a) - 1) if abs(a[k]) >= abs(a[k - 1]) and abs(a[k]) >= abs(a[k + 1])]
        if len(ext) > 2:
            ta, aa = t[ext], np.abs(a[ext])
            out["damping"] = float(-np.polyfit(ta, np.log(aa), 1)[0])
    return out


def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, kw = a
    return {"static_drop": static_drop, "capillary_wave": capillary_wave}[kind](**kw)


def main() -> None:
    import multiprocessing as mp

    J = []
    for conf in ("base", "full", "full-noE3"):
        for n in (32, 64):
            for cv in ("exact", "numerical"):
                J.append(("static_drop", dict(conf=conf, n=n, curvature=cv)))
            J.append(("capillary_wave", dict(conf=conf, n=n)))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "12"))) as pool:
        res = pool.map(_job, J, chunksize=1)
    with open(os.path.join("docs", "figures", "r0_combined_tension.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for r in res:
        print({k: v for k, v in r.items() if k not in ("history", "amplitude")})


if __name__ == "__main__":
    main()

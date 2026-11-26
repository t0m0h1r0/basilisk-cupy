"""P0 (docs/notes/77): the VOF concentrations carried by a prescribed, discretely divergence-free velocity, with no
surface tension and no flow solve, to measure the flux error of the concentration gradient alone.

A drop (R = 0.15) on the periodic [0, 1]^2 holds ``t1 = c phi1`` and the outer fluid ``t2 = (1 - c) phi2`` (vof.h's
concentrations, ``VOFTracer`` / ``inverse=True``) with smooth ``phi1 = 2 + sin(2 pi x) cos(2 pi y)``,
``phi2 = 1 + 0.5 cos(2 pi x) sin(4 pi y)``. Flows:

* ``uniform``: (1, 0.5) for t = 2 (the drop and both fields return to the start; direction splitting is exact);
* ``vortex``: psi = sin(2 pi x) sin(2 pi y) / (2 pi), reversed by cos(pi t / T), T = 0.5, back to the start at T.

The concentration ``t/c`` (``t/(1 - c)``) is compared with the initial cell averages in the full cells of each phase
(c > 1 - 1e-6, resp. c < 1e-6), in bands of the physical distance d from the interface: ``near`` d <= 3h, ``mid``
3h < d <= 0.05, ``far`` d > 0.05: max and mean (L1) errors. (The interfacial cells start with c times the mean of
phi over the phase's part of the cell.) Gradients: vof.h's centred difference (Basilisk's default for conserving.h's momentum), the
phase-wise CCD (``"ccd"``) and CCD + the swept parabola (``"ccd2"``), each with the CCD runs starting at c >= 0.5 or
c >= 0.99; "-geo": the slope term at the centroids of the phase's part of the swept strip and of the cell
(``VOFTracer(geometric=True)``, P1).

    python -m cases.pure_advection [--jobs 28] [--n 32 64 128 256] [--gradients ...]
    python -m cases.pure_advection --oracle | --coupled | --cfl [--n ...]    # sections 2, 3 and 3.1 of the note
    python -m cases.pure_advection --free-stream                             # section 4
    python -m cases.pure_advection --plot
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

JSON = "docs/figures/pure_advection.json"
GRADIENTS = {"centred": (None, 0.5), "ccd": ("ccd", 0.5), "ccd-c99": ("ccd", 0.99), "ccd2": ("ccd2", 0.5),
             "ccd2-c99": ("ccd2", 0.99), "centred-geo": (None, 0.5), "ccd-geo": ("ccd", 0.5),
             "ccd-c99-geo": ("ccd", 0.99)}  # "-geo": VOFTracer(geometric=True) (P1)


def phi1(x, y):
    return 2.0 + np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def phi2(x, y):
    return 1.0 + 0.5 * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)


def cell_average(f, n, order: int = 6):
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(order)
    c = (np.arange(n) + 0.5) * h
    out = np.zeros((n, n))
    for a, wa in zip(xg, wg):
        for b, wb in zip(xg, wg):
            X, Y = np.meshgrid(c + 0.5 * h * a, c + 0.5 * h * b, indexing="ij")
            out += 0.25 * wa * wb * f(X, Y)
    return out


def phase_means(n, xc, yc, k: int = 16):
    """The means of phi1 over the drop's part and of phi2 over the rest of every cell (k x k sub-points; NaN where a
    phase has no sub-point)."""
    h = 1.0 / n
    s = (np.arange(k) + 0.5) / k * h
    S1, S2, W1, W2 = (np.zeros((n, n)) for _ in range(4))
    c = np.arange(n) * h
    for a in s:
        for b in s:
            X, Y = np.meshgrid(c + a, c + b, indexing="ij")
            inside = (X - xc) ** 2 + (Y - yc) ** 2 < 0.15 ** 2
            S1 += np.where(inside, phi1(X, Y), 0.0)
            W1 += inside
            S2 += np.where(inside, 0.0, phi2(X, Y))
            W2 += ~inside
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(W1 > 0, S1 / W1, np.nan), np.where(W2 > 0, S2 / W2, np.nan)


def face_velocity(flow: str, n: int, t: float, T: float):
    """(u on x-faces (n + 1, n), v on y-faces (n, n + 1)), exactly divergence-free (from psi at the vertices)."""
    if flow == "uniform":
        return np.ones((n + 1, n)), np.full((n, n + 1), 0.5)
    h = 1.0 / n
    xv = np.arange(n + 1) * h
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    P = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) / (2 * np.pi) * math.cos(math.pi * t / T)
    u = (P[:, 1:] - P[:, :-1]) / h     # x-face (i h, (j + 1/2) h): psi_y
    v = -(P[1:, :] - P[:-1, :]) / h    # y-face ((i + 1/2) h, j h): -psi_x
    return u, v


def _exact_gradient(n: int, clock: dict):
    """vof.h's concentration slope (times h) replaced by the exact one of the translated phi1 / phi2 at the cell
    centres (the oracle of P1, uniform flow only); ``clock["t"]``: the time of the current step's start."""
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    def gradient(xp, cp, tp, axis, inverse, limiter):
        x, y = X - clock["t"], Y - 0.5 * clock["t"]
        if not inverse:
            d = (2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) if axis == 0
                 else -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
        else:
            d = (-np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y) if axis == 0
                 else 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y))
        return xp.asarray(d * h)

    return gradient


def run(flow: str, gradient: str, n: int, cfl: float = 0.5, xc: float = 0.5, yc: float = 0.5,
        oracle: bool = False) -> dict:
    """``oracle``: vof.h's slope of the concentrations replaced by the exact one (uniform flow, ``gradient``
    "centred" or "centred-geo")."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection

    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    if flow == "vortex":
        yc = 0.3
    c0 = np.asarray(fraction(grid, lambda x, y: 0.15 ** 2 - (x - xc) ** 2 - (y - yc) ** 2, refine=8))
    p1, p2 = cell_average(phi1, n), cell_average(phi2, n)
    # the interfacial cells hold c times the mean of phi over the phase's part of the cell (16 x 16 sub-points)
    m1, m2 = phase_means(n, xc, yc)
    c = CellField.from_interior(grid, c0.copy(), bc)
    t1 = CellField.from_interior(grid, c0 * np.where(np.isnan(m1), p1, m1), bc)
    t2 = CellField.from_interior(grid, (1.0 - c0) * np.where(np.isnan(m2), p2, m2), bc)
    g, cmin = GRADIENTS[gradient]
    geo = gradient.endswith("-geo")
    itf = VOFInterface(c, [VOFTracer(t1, gradient=g, ccd_cmin=cmin, geometric=geo),
                           VOFTracer(t2, inverse=True, gradient=g, ccd_cmin=cmin, geometric=geo)])
    h = 1.0 / n
    t_end, T = (2.0, 0.0) if flow == "uniform" else (0.5, 0.5)
    umax = 1.0 if flow == "uniform" else 1.0  # max |psi_y| = 1
    steps = int(math.ceil(t_end / (cfl * h / umax) - 1e-9))
    dt = t_end / steps
    uf = FaceField(grid, name="u")
    import basilisk_cupy.interface.vof as vofmod

    saved, clock = vofmod._concentration_gradient, {"t": 0.0}
    if oracle:
        if flow != "uniform":
            raise ValueError("the oracle slope is the translated phi's (uniform flow)")
        vofmod._concentration_gradient = _exact_gradient(n, clock)
    try:
        for i in range(steps):
            clock["t"] = i * dt
            u, v = face_velocity(flow, n, (i + 0.5) * dt, T)
            uf.components[0][...] = u
            uf.components[1][...] = v
            vof_advection([itf], uf, dt, i)
    finally:
        vofmod._concentration_gradient = saved
    cf = np.asarray(c.interior)
    out = {"flow": flow, "gradient": gradient + ("-oracle" if oracle else ""), "n": n, "steps": steps, "shape_l1": float(np.abs(cf - c0).mean())}
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    d = np.abs(np.hypot(X - xc, Y - yc) - 0.15)  # distance from the (returned) interface
    bands = {"near": d <= 3 * h, "mid": (d > 3 * h) & (d <= 0.05), "far": d > 0.05}
    for k, (tf, full, ref) in enumerate(((t1, cf > 1 - 1e-6, p1), (t2, cf < 1e-6, p2)), start=1):
        conc = np.asarray(tf.interior) / np.where(k == 1, np.maximum(cf, 1e-300), np.maximum(1 - cf, 1e-300))
        err = np.abs(conc - ref)
        for name, band in bands.items():
            sel = full & band
            out[f"phase{k}_{name}"] = float(err[sel].max()) if sel.any() else math.nan
            out[f"phase{k}_{name}_l1"] = float(err[sel].mean()) if sel.any() else math.nan
    return out


def coupled(variant: str, n: int, ratio: float = 1.0, t_end: float = 0.5) -> dict:
    """The coupled half of P0: inviscid Taylor-Green ``u = (sin 2 pi x cos 2 pi y, -cos 2 pi x sin 2 pi y)`` (a steady
    Euler flow) on the periodic [0, 1]^2 through Basilisk's centred solver, with a passive drop (R = 0.15 at (0.5,
    0.3), density ratio ``ratio``, no surface tension) carried by it. At density ratio 1 the exact velocity stays the
    initial one, so max|u - u0| at ``t_end`` measures the momentum transport with an interface present. ``variant``:
    "single" (no drop: the solver's own error), or a TwoPhase variant of ``cases.convection_stability`` (e.g.
    "centered-conserving-ccd-c99-geo")."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.convection_stability import conserving_options

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-8)
    if variant != "single":
        from basilisk_cupy.interface.two_phase import TwoPhase

        bc = BoundaryConditions()
        for ax in range(2):
            for side in (0, 1):
                bc.set((ax, side), Periodic())
        f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2,
                                                   refine=8), bc)
        TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, **conserving_options(variant))
    x, y = (to_numpy(a) for a in grid.meshgrid())
    u0 = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    v0 = -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    ns.u[0].interior = grid.xp.asarray(u0)
    ns.u[1].interior = grid.xp.asarray(v0)
    ns.start(events=[t_end])
    failed = False
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        if not max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u) < 10.0:
            failed = True
            break
    du = np.hypot(to_numpy(ns.u[0].interior) - u0, to_numpy(ns.u[1].interior) - v0)
    out = {"variant": variant, "n": n, "ratio": ratio, "failed": failed, "steps": ns.clock.i,
           "du_max": float(du.max()), "du_l1": float(du.mean())}
    if variant != "single":
        F = to_numpy(f.interior)
        # where: near the drop's interface (cells with 0 < f < 1 and their 3-cell neighbourhood) or not
        near = (F > 1e-6) & (F < 1 - 1e-6)
        for _ in range(3):
            near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
        out["du_near"] = float(du[near].max())
        out["du_away"] = float(du[~near].max())
    return out


def tg_cfl(variant: str, n: int, cfl: float, t_end: float = 0.5) -> dict:
    """Inviscid Taylor-Green with no interface (``variant`` "plain": centred.h's BCG advection; otherwise a
    conserving TwoPhase variant with f = 0 everywhere), the CFL number set after TwoPhase's: max|u - u0| at t_end
    separates the time error (C, the C build: tools/basilisk/tg_conserving.c) from the space error."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.convection_stability import conserving_options

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-8)
    if variant != "plain":
        from basilisk_cupy.interface.two_phase import TwoPhase

        bc = BoundaryConditions()
        for ax in range(2):
            for side in (0, 1):
                bc.set((ax, side), Periodic())
        TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), bc), rho1=1.0, rho2=1.0,
                 **conserving_options(variant))
    ns.cfl = cfl
    x, y = (to_numpy(a) for a in grid.meshgrid())
    u0 = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    v0 = -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    ns.u[0].interior = grid.xp.asarray(u0)
    ns.u[1].interior = grid.xp.asarray(v0)
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()
    du = np.hypot(to_numpy(ns.u[0].interior) - u0, to_numpy(ns.u[1].interior) - v0)
    return {"variant": variant, "n": n, "cfl": cfl, "steps": ns.clock.i, "du_max": float(du.max())}


def free_stream(variant: str, n: int = 32, ratio: float = 1000.0, steps: int = 60) -> dict:
    """P0-1: a drop (R = 0.15, density ratio ``ratio``, no surface tension) carried by the uniform flow (1, 0.37):
    max over ``steps`` steps of |u - U| (round-off when the momentum transport preserves a free stream)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.convection_stability import conserving_options

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2,
                                               refine=8), bc)
    TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, **conserving_options(variant))
    U = (1.0, 0.37)
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
    ns.start()
    e = 0.0
    for _ in range(steps):
        ns.step()
        e = max([e] + [float(np.abs(to_numpy(ns.u[ax].interior) - U[ax]).max()) for ax in range(2)])
    return {"variant": variant, "n": n, "ratio": ratio, "du_max": e}


def _merge(path: str, rows: list, keys: tuple) -> None:
    """Add ``rows`` to the JSON list at ``path``, replacing the rows with the same ``keys``."""
    try:
        with open(path) as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        old = []
    new = {tuple(r[k] for k in keys) for r in rows}
    with open(path, "w") as fh:
        json.dump([r for r in old if tuple(r[k] for k in keys) not in new] + rows, fh, indent=1)


def plot() -> None:
    """Left: the uniform translation's phase-1 error near the interface (d <= 3h) for vof.h's slope / the exact one
    with vof.h's / the consistent geometry; right: the coupled Taylor-Green's max|u - u0|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    with open(JSON) as fh:
        rows = {(r["flow"], r["gradient"], r["n"]): r for r in json.load(fh)}
    with open(JSON.replace(".json", "_coupled.json")) as fh:
        crow = {(r["variant"], r["n"]): r for r in json.load(fh)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = [32, 64, 128, 256]
    for g, label, st in (("centred", "vof.h slope, vof.h geometry", "k-o"),
                         ("centred-geo", "vof.h slope, consistent geometry", "k--o"),
                         ("ccd", "CCD slope (c >= 0.5), vof.h geometry", "C3-^"),
                         ("ccd-geo", "CCD slope (c >= 0.5), consistent geometry", "C3--^"),
                         ("centred-oracle", "exact slope, vof.h geometry", "C0-s"),
                         ("centred-geo-oracle", "exact slope, consistent geometry", "C0--s")):
        x = [n for n in ns if ("uniform", g, n) in rows]
        a.loglog(x, [min(rows[("uniform", g, n)]["phase1_near"], 1.0) for n in x], st, label=label)
    a.set_title("pure advection (uniform), phase 1 near the interface")
    a.set_ylabel("max error, d <= 3h (clipped at 1)")
    a.legend(fontsize=7)
    nc = [32, 64, 128]
    for v, label, st in (("single", "no drop (BCG)", "k-o"), ("centered-vof", "Basilisk VOF", "C2-o"),
                         ("centered-conserving", "conserving.h", "C3-^"),
                         ("centered-conserving-ccd-c99-geo", "conserving + CCD c>=0.99 + geometry", "C1-v"),
                         ("centered-conserving-src", "conserving + source term", "C0-s")):
        b.loglog(nc, [crow[(v, n)]["du_max"] for n in nc], st, label=label)
    b.set_title("Taylor-Green with a passive drop (ratio 1), t = 0.5")
    b.set_ylabel("max|u - u0|")
    b.legend(fontsize=7)
    for ax, xs in ((a, ns), (b, nc)):
        ax.set_xlabel("N")
        ax.set_xticks(xs, [str(n) for n in xs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/figures/pure_advection.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--flows", nargs="*", default=["uniform", "vortex"])
    ap.add_argument("--gradients", nargs="*", default=list(GRADIENTS))
    ap.add_argument("--coupled", action="store_true", help="the coupled test (Taylor-Green with a passive drop)")
    ap.add_argument("--oracle", action="store_true", help="the exact slope (uniform flow, centred / centred-geo)")
    ap.add_argument("--cfl", action="store_true", help="Taylor-Green with no interface against the CFL number")
    ap.add_argument("--plot", action="store_true", help="docs/figures/pure_advection.png from the JSON files")
    ap.add_argument("--free-stream", action="store_true", help="P0-1: a drop in a uniform flow, no tension")
    args = ap.parse_args()
    if args.plot:
        plot()
        return
    if args.free_stream:
        for v in ("centered-vof", "centered-conserving", "centered-conserving-ccd", "centered-conserving-ccd2",
                  "centered-conserving-geo", "centered-conserving-src", "centered-conserving-ccd-geo-src"):
            r = free_stream(v)
            print(f"free stream {v:34s}: max over 60 steps |u - U| {r['du_max']:.2e}", flush=True)
        return
    from cases.parallel import pmap

    if args.cfl:
        variants = ["plain", "centered-conserving", "centered-conserving-pred", "centered-conserving-src",
                    "centered-conserving-src-tr", "centered-conserving-fresh", "centered-conserving-src-fresh"]
        items = [dict(variant=v, n=n, cfl=c) for n in sorted(args.n, reverse=True) for c in (0.5, 0.25, 0.125)
                 for v in variants]
        rows = []

        def fshow(k, it, r):
            rows.append(r)
            print(f"tg {r['variant']:32s} N {r['n']:3d} CFL {r['cfl']:5.3f}: max|u - u0| {r['du_max']:.3e}"
                  f" ({r['steps']} steps)", flush=True)

        pmap(tg_cfl, items, args.jobs, on_result=fshow)
        _merge(JSON.replace(".json", "_cfl.json"), rows, ("variant", "n", "cfl"))
        return
    if args.oracle:
        items = [dict(flow="uniform", gradient=g, n=n, oracle=True) for n in sorted(args.n, reverse=True)
                 for g in ("centred", "centred-geo")]
        rows = []

        def oshow(k, it, r):
            rows.append(r)
            print(f"oracle {r['gradient']:20s} N {r['n']:3d}: " + " | ".join(
                f"phase {p} {b} {r[f'phase{p}_{b}']:.2e}" for p in (1, 2) for b in ("near", "mid", "far")), flush=True)

        pmap(run, items, args.jobs, on_result=oshow)
        _merge(JSON, rows, ("flow", "gradient", "n"))
        return

    if args.coupled:
        variants = ["single", "centered-vof", "centered-conserving", "centered-conserving-geo",
                    "centered-conserving-ccd", "centered-conserving-ccd-c99", "centered-conserving-ccd-geo",
                    "centered-conserving-ccd-c99-geo", "centered-conserving-ccd2-c99", "centered-conserving-src",
                    "centered-conserving-geo-src", "centered-conserving-ccd-c99-geo-src"]
        items = [dict(variant=v, n=n) for n in sorted(args.n, reverse=True) for v in variants]
        crow = []

        def cshow(k, it, r):
            crow.append(r)
            print(f"coupled {r['variant']:32s} N {r['n']:3d}: " + ("FAILED " if r["failed"] else "")
                  + f"max|u - u0| {r['du_max']:.3e} (L1 {r['du_l1']:.3e})"
                  + (f" near the drop {r['du_near']:.3e} away {r['du_away']:.3e}" if "du_near" in r else ""),
                  flush=True)

        pmap(coupled, items, args.jobs, on_result=cshow)
        _merge(JSON.replace(".json", "_coupled.json"), crow, ("variant", "n", "ratio"))
        return

    items = [dict(flow=f, gradient=g, n=n) for n in sorted(args.n, reverse=True) for f in args.flows for g in args.gradients]
    rows = []

    def show(k, it, r):
        rows.append(r)
        print(f"{r['flow']:8s} {r['gradient']:9s} N {r['n']:3d}: " + " | ".join(
            f"phase {p} {b} {r[f'phase{p}_{b}']:.2e} ({r[f'phase{p}_{b}_l1']:.2e})" for p in (1, 2)
            for b in ("near", "mid", "far")) + f" | shape L1 {r['shape_l1']:.2e}", flush=True)

    pmap(run, items, args.jobs, on_result=show)
    _merge(JSON, rows, ("flow", "gradient", "n"))


if __name__ == "__main__":
    main()

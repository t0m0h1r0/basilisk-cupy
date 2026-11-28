"""#194: the missing parts of R0's tests (the inventory of #183), as functions that take the configuration and the
test's definition as arguments. No verdict here (the thresholds, grids and times come from design/168 / #168).

Configuration ``conf`` (a dict; every key optional):
    conserving (True), filtered (False), face_density ("smooth": two-phase.h; "mean": TwoPhaseVOFJump with the
    arithmetic-mean cell coefficient, docs/notes/115), viscosity_mean ("arithmetic": two-phase.h; "harmonic": the face
    viscosity 1 / (c / mu1 + (1 - c) / mu2), docs/notes/121), fixes (False; True: PR #192's
    conserving_consistent=True, conserving_slope_limit=3.0), project_start (False: Basilisk's start; True: a
    divergence-free first face velocity), tolerance (1e-10).

Common outputs: ``divergence`` (h max|div u_f| / U_ref over the run, the start included) and ``split`` (L1 / L2 /
Linf of an error field in all cells, in the interface band -- cell centres within 2.5 h of the reference interface
along its normal -- and in the pure phases).

  couette_poiseuille()  two layers between walls (exact steady profiles), the interface on a face or inside a cell
  bubble()              Hysing's rising bubble (cases 1 and 2), with ``HYSING`` (literature values, see the note)
  capwave_reference()   the exact linear capillary wave over 10 periods (the audit's capwave_exact.py as the yardstick)
  static_drop()         the drop with its centre moved by a fraction of a cell
  drop_translation()    a drop carried by a uniform flow for a given number of periods: velocity, f and mass errors
  shear_layer()         the two-phase double shear layer: kinetic energy and total momentum in time
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

RHO = (1.0, 1e-3)


# --- the configuration -----------------------------------------------------------------------------------------
def two_phase(ns, f, rho, mu, conf=None):
    """TwoPhase (or TwoPhaseVOFJump) for ``conf``; the viscosity mean as a hook."""
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.interface.vof_jump import TwoPhaseVOFJump

    conf = dict(conf or {})
    kw = dict(rho1=rho[0], rho2=rho[1], mu1=mu[0], mu2=mu[1], conserving=conf.get("conserving", True),
              filtered=conf.get("filtered", False))
    if conf.get("fixes") and kw["conserving"]:
        kw.update(conserving_consistent=True, conserving_slope_limit=3.0)
    if conf.get("face_density", "smooth") == "mean":
        tp = TwoPhaseVOFJump(ns, f, cell_coefficient="mean", **kw)
    else:
        tp = TwoPhase(ns, f, **kw)
    if conf.get("viscosity_mean", "arithmetic") == "harmonic" and (mu[0] or mu[1]):
        from cases.r0_shear import harmonic_viscosity

        harmonic_viscosity(ns, tp)
    ns.project_start = bool(conf.get("project_start", False))
    return tp


def tolerance(conf):
    return float((conf or {}).get("tolerance", 1e-10))


# --- common outputs ----------------------------------------------------------------------------------------------
def divergence(ns, u_ref=1.0):
    """h max |div u_f| / U_ref (h div u_f is the face-velocity difference of a cell)."""
    from basilisk_cupy.backend import to_numpy

    fx, fy = (to_numpy(c) for c in ns.uf.components)
    nx, ny = ns.grid.n
    d = (fx[1:nx + 1, :] - fx[:nx, :]) + (fy[:, 1:ny + 1] - fy[:, :ny])
    return float(np.abs(d).max()) / u_ref


def split(err, band):
    """L1 / L2 / Linf of ``err`` (cell array, absolute values taken) in all cells, in ``band`` and outside it."""
    err = np.abs(np.asarray(err))
    out = {}
    for name, sel in (("all", np.ones_like(band, bool)), ("band", band), ("pure", ~band)):
        x = err[sel]
        out[name] = ({"L1": float(x.mean()), "L2": float(np.sqrt((x * x).mean())), "Linf": float(x.max()),
                      "cells": int(x.size)} if x.size else {"L1": 0.0, "L2": 0.0, "Linf": 0.0, "cells": 0})
    return out


def band_from_distance(dist, h, width=2.5):
    """Cells whose centre is within ``width`` cells of the reference interface (``dist``: signed or unsigned distance
    of the cell centres to it)."""
    return np.abs(dist) < width * h


# --- two-layer Couette / Poiseuille --------------------------------------------------------------------------------
def _layers_exact(kind, yi, mu, U=1.0, F=1.0):
    """The steady profile u(y) of two layers between walls y = 0 (u = 0) and y = 1 (u = U for Couette, 0 for
    Poiseuille), phase 1 (mu1) below y = yi. Couette: (mu u')' = 0; Poiseuille: (mu u')' = -F. Continuity of u and
    mu u' at yi. Returns a function of y (vectorised)."""
    m1, m2 = mu
    if kind == "couette":
        tau = U / (yi / m1 + (1 - yi) / m2)
        return lambda y: np.where(y < yi, tau * y / m1, tau * yi / m1 + tau * (y - yi) / m2)
    # mu u' = C - F y in both layers; u(0) = 0; u(1) = 0
    # u1 = (C y - F y^2 / 2) / m1 ; u2 = u1(yi) + (C (y - yi) - F (y^2 - yi^2) / 2) / m2
    a = yi / m1 + (1 - yi) / m2
    b = (yi * yi / 2) / m1 + ((1 - yi * yi) / 2) / m2
    C = F * b / a
    u1 = lambda y: (C * y - F * y * y / 2) / m1  # noqa: E731
    return lambda y: np.where(y < yi, u1(y), u1(yi) + (C * (y - yi) - F * (y * y - yi * yi) / 2) / m2)


def _cell_averages_1d(fun, n, yi):
    """Cell averages over [j h, (j + 1) h] of a piecewise quadratic with a kink at yi (3-point Gauss on each piece:
    exact)."""
    g, w = np.polynomial.legendre.leggauss(3)
    h = 1.0 / n
    out = np.zeros(n)
    for j in range(n):
        lo, hi = j * h, (j + 1) * h
        cuts = [lo] + ([yi] if lo < yi < hi else []) + [hi]
        for a, b in zip(cuts[:-1], cuts[1:]):
            y = a + (b - a) * (g + 1) / 2
            out[j] += np.sum(w * fun(y)) * (b - a) / 2
        out[j] /= h
    return out


def couette_poiseuille(kind="couette", n=32, rho=RHO, mu=(1.0, 1.0 / 55.0), yi=0.5, offset=0.0, t_end=2.0, conf=None,
                       U=1.0, F=1.0, dt_max=None):
    """Two layers in a channel [0, 1] x [0, 1], periodic in x, no-slip walls at y = 0 and 1 (the top one moving at U
    for Couette), phase 1 below y = yi + offset h (offset 0: the interface on a face; 0.3: inside a cell). Poiseuille:
    a uniform force per volume F along x (the face acceleration F / rho_face). Started at rest; the error of u_x
    (cell averages) against the steady profile at t_end, split into the band and the pure phases, the divergence,
    and how steady the field is (max |u(t_end) - u(t_end - 0.1 t_end)|)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Dirichlet, Neumann, Periodic
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    h = 1.0 / n
    y_i = yi + offset * h
    grid = CartesianGrid((n, n))
    walls = {w: 0.0 for w in ("bottom", "top")}
    ns = CenteredNS(grid, periodic=(0,), uf_normal=walls,
                    tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(U if kind == "couette" else 0.0)},
                    tolerance=tolerance(conf), mu=max(mu))
    ns.DT = dt_max or 0.5 / n
    bc = BoundaryConditions()
    for side in (0, 1):
        bc.set((0, side), Periodic())
        bc.set((1, side), Neumann(0.0))
    yc = (np.arange(n) + 0.5) * h
    f1 = np.clip((y_i - np.arange(n) * h) / h, 0.0, 1.0)            # phase 1 below y_i (exact cell fractions)
    f0 = np.tile(f1, (n, 1))
    tp = two_phase(ns, CellField.from_interior(grid, grid.xp.asarray(f0), bc), rho, mu, conf)
    if kind == "poiseuille":
        ns.a_is_constant = False

        def force(s):
            s.a[0] = s.a[0] + F * s.alpha[0]

        ns.on("acceleration", force)
    exact = _layers_exact(kind, y_i, mu, U, F)
    ue = np.tile(_cell_averages_1d(exact, n, y_i), (n, 1))
    ns.start(events=sorted({0.9 * t_end, t_end}))
    div, prev = divergence(ns, max(U, 1e-300)), None
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        div = max(div, divergence(ns, max(U, 1e-300)))
        if abs(ns.clock.t - 0.9 * t_end) < 1e-12:
            prev = to_numpy(ns.u[0].interior).copy()
    u = to_numpy(ns.u[0].interior)
    band = band_from_distance(np.tile(yc - y_i, (n, 1)), h)
    return {"kind": kind, "n": n, "rho": list(rho), "mu": list(mu), "yi": y_i, "t": float(ns.clock.t),
            "steps": ns.clock.i, "conf": conf or {}, "error": split(u - ue, band),
            "uy_max": float(np.abs(to_numpy(ns.u[1].interior)).max()), "divergence": div,
            "unsteady": float(np.abs(u - prev).max()) if prev is not None else None,
            "f_change": float(np.abs(to_numpy(tp.f.interior) - f0).max())}


# --- Hysing's rising bubble ---------------------------------------------------------------------------------------
#: Hysing et al. (2009), "Quantitative benchmark computations of two-dimensional bubble dynamics", the reference
#: groups' values (TP2D) at the finest grid, as recalled here: NOT checked against the paper in this session (see
#: docs/notes/124): rise velocity maximum (and its time), centroid at t = 3.
# Hysing et al., MOX Report 23/2008 (the preprint of IJNMF 2009), group 1 (TP2D): case 1 table V, case 2 table XVI.
# Checked against the preprint by the audit (PR #197); the published version's tables not seen. Case 2's groups
# disagree (y_c(3): TP2D 1.1380, FreeLIFE 1.1249, MooNMD 1.1376): record the difference (design/168).
HYSING = {1: {"v_max": 0.2417, "t_v_max": 0.9213, "yc_t3": 1.0813, "source": "TP2D, table V", "checked": True},
          2: {"v_max": 0.2524, "t_v_max": 0.7332, "v_max2": 0.2434, "t_v_max2": 2.0705, "yc_t3": 1.1380,
              "source": "TP2D, table XVI", "checked": True}}


def bubble(case=2, n=32, T=3.0, conf=None, every=0.01):
    """Hysing's rising bubble (the configuration of the audit's PR #177 and of cases/r0_trials.bubble): [0, 1] x
    [0, 2], slip side walls, no-slip top and bottom, R = 0.25 at (0.5, 0.5), reduced gravity 0.98. Rise velocity and
    centroid in time, the phase mass, the divergence."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Dirichlet
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    p = {1: dict(rho=(1000.0, 100.0), mu=(10.0, 1.0), sigma=24.5),
         2: dict(rho=(1000.0, 1.0), mu=(10.0, 0.1), sigma=1.96)}[case]
    grid = CartesianGrid((n, 2 * n), length=(1.0, 2.0))
    walls = {w: 0.0 for w in ("left", "right", "bottom", "top")}
    ns = CenteredNS(grid, uf_normal=walls, tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)},
                    tolerance=float((conf or {}).get("tolerance", 1e-6)), mu=p["mu"][0])
    f0 = to_numpy(fraction(grid, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.25 ** 2, refine=8))
    tp = two_phase(ns, CellField.from_interior(grid, f0, BoundaryConditions()), p["rho"], p["mu"], conf)
    tp.add_surface_tension(p["sigma"])
    tp.add_reduced_gravity(G=(0.0, -0.98), Z=(0.5, 0.5))
    y = (np.arange(2 * n) + 0.5) / n
    ns.DT = 0.5 / n
    times = list(np.arange(1, int(round(T / every)) + 1) * every)
    ns.start(events=times)
    m0 = float((1.0 - np.clip(f0, 0, 1)).sum())
    hist, failed, div = [], None, divergence(ns)
    for te in times:
        while ns.clock.t < te - 1e-12:
            try:
                ns.step()
            except Exception as exc:  # noqa: BLE001
                failed = repr(exc)[:100]
                break
            div = max(div, divergence(ns))
        if failed:
            break
        b = 1.0 - np.clip(to_numpy(tp.f.interior), 0, 1)
        vb = float((b * to_numpy(ns.u[1].interior)).sum() / b.sum())
        hist.append((float(ns.clock.t), vb, float((b * y[None, :]).sum() / b.sum())))
        if not math.isfinite(vb) or abs(vb) > 20:
            failed = "blow-up at t = %.3f" % ns.clock.t
            break
    h = np.array(hist) if hist else np.zeros((0, 3))
    k = int(np.argmax(h[:, 1])) if len(h) else 0
    out = {"case": case, "n": n, "conf": conf or {}, "t": float(ns.clock.t), "steps": ns.clock.i, "failed": failed,
           "v_max": float(h[k, 1]) if len(h) else None, "t_v_max": float(h[k, 0]) if len(h) else None,
           "yc_end": float(h[-1, 2]) if len(h) else None, "divergence": div,
           "mass": abs(float((1.0 - np.clip(to_numpy(tp.f.interior), 0, 1)).sum()) - m0) / m0,
           "history": h[:: max(1, len(h) // 100)].tolist(), "reference": HYSING[case]}
    return out


# --- the capillary wave's exact linear reference over 10 periods --------------------------------------------------
def capwave_reference(t_end=4.1, dt=0.004, path=None, out=None, procs=16):
    """The exact linear initial-value solution of the capillary wave of cases/r0_combined_tension.capillary_wave
    (two layers of depth 1/2, slip walls, rho 1 / 1e-3, mu 1e-2 / (1e-2 / 55), sigma 1, a0 = 0.01, k = 2 pi) from the
    audit's ``audit/pr177/capwave_exact.py`` (Laplace transform; inverted with Talbot's method here: the audit's de Hoog of degree 24 fails beyond t ~ 3), evaluated to t_end
    (10 periods = 4.03). ``path``: that file (env CAPWAVE_EXACT; it is on the audit's branch, not on main).
    The table it writes, docs/figures/capwave_exact_10periods.json, is the reference to use (checked by the audit
    against de Hoog of degree 48 to 2.3e-13, PR #197); this function only regenerates it."""
    import importlib.util
    import multiprocessing as mp

    path = path or os.environ.get("CAPWAVE_EXACT", "audit/pr177/capwave_exact.py")
    out = out or os.path.join("docs", "figures", "capwave_exact_10periods.json")
    t = list(np.arange(1, int(round(t_end / dt)) + 1) * dt)
    with mp.get_context("spawn").Pool(procs) as pool:
        a = pool.starmap(_capwave_point, [(path, ti) for ti in t])
    spec = importlib.util.spec_from_file_location("capwave_exact", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    period, damping, _ = mod.measure(np.array(t), np.array(a))
    res = {"source": "audit C-2, audit/pr177/capwave_exact.py (branch audit/20261128-pr177), evaluated by "
                     "cases.r0_parts.capwave_reference", "t": t, "a": a, "period": period, "damping": damping}
    with open(out, "w") as fh:
        json.dump(res, fh)
    return res


def _capwave_point(path, t):
    import importlib.util

    spec = importlib.util.spec_from_file_location("capwave_exact", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Talbot's contour (the audit's de Hoog of degree 24 loses the solution from t ~ 3: 1e-7 instead of 1e-3; Talbot and
    # de Hoog of degree 48 agree to 1e-14 there)
    return float(mod.mp.invertlaplace(mod.eta_hat, mod.mp.mpf(float(t)), method="talbot").real)


# --- the static drop, moved off the grid ----------------------------------------------------------------------------
def static_drop(n=32, curvature="numerical", offset=(0.0, 0.0), t_end=1.0, R=0.2, conf=None, mu=(1e-2, 1e-2 / 55)):
    """The drop of cases/r0_combined_tension.static_drop (periodic unit square, sigma = 1, rho 1 / 1e-3) with its
    centre at (0.5, 0.5) + offset h. Parasitic currents in time (max |u|), the velocity split into the band and the
    pure phases at t_end, the divergence."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    h = 1.0 / n
    cx, cy = 0.5 + offset[0] * h, 0.5 + offset[1] * h
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance(conf), mu=mu[0])
    f0 = to_numpy(fraction(grid, lambda x, y: R * R - (x - cx) ** 2 - (y - cy) ** 2, refine=8))
    tp = two_phase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), RHO, mu, conf)
    tp.add_surface_tension(1.0)
    if curvature == "exact":
        numeric = tp.potentials[0]
        sign = np.sign(np.mean([v for v in to_numpy(numeric(tp)).ravel() if v < NODATA / 2]))

        def exact(t):
            fi = to_numpy(t.f.interior)
            return grid.xp.asarray(np.where((fi > 0) & (fi < 1), sign / R, NODATA))

        tp.potentials[0] = exact
    ns.start(events=[t_end])
    umax, div, failed = 0.0, divergence(ns), None
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        u = float(np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)).max())
        umax, div = max(umax, u), max(div, divergence(ns))
        if not math.isfinite(u) or u > 1e2:
            failed = "blow-up at t = %.4f" % ns.clock.t
            break
    c = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(c, c, indexing="ij")
    band = band_from_distance(np.hypot(X - cx, Y - cy) - R, h)
    speed = np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior))
    return {"n": n, "curvature": curvature, "offset": list(offset), "conf": conf or {}, "umax": umax,
            "speed_end": split(speed, band), "divergence": div, "steps": ns.clock.i, "failed": failed}


# --- the drop carried by a uniform flow ----------------------------------------------------------------------------
def drop_translation(n=32, ratio=1e3, periods=10, U=(1.0, 0.0), R=0.15, conf=None):
    """A drop (R at (0.5, 0.5)) carried by U through the periodic unit square for ``periods`` crossings along x
    (t_end = periods / |U_x|): max |u - U| over the run, the f error against the exactly translated drop at t_end
    (cell averages, L1), the phase mass's relative change, the divergence."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance(conf))

    def drop_at(cx, cy):
        def phi(x, y):
            dx = (x - cx + 0.5) % 1.0 - 0.5
            dy = (y - cy + 0.5) % 1.0 - 0.5
            return R * R - dx * dx - dy * dy
        return to_numpy(fraction(grid, phi, refine=8))

    f0 = drop_at(0.5, 0.5)
    tp = two_phase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), (1.0, 1.0 / ratio),
                   (0.0, 0.0), conf)
    ns.cfl = 0.5
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    t_end = periods / abs(U[0])
    ns.start(events=[t_end])
    m0 = float(f0.sum())
    worst, div, failed = 0.0, divergence(ns, math.hypot(*U)), None
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
        worst, div = max(worst, e), max(div, divergence(ns, math.hypot(*U)))
        if not math.isfinite(e) or e > 10:
            failed = "blow-up at t = %.4f" % ns.clock.t
            break
    t = float(ns.clock.t)
    fe = drop_at(0.5 + U[0] * t, 0.5 + U[1] * t)
    f = to_numpy(tp.f.interior)
    return {"n": n, "ratio": ratio, "periods": periods, "U": list(U), "conf": conf or {}, "t": t,
            "steps": ns.clock.i, "velocity_error": worst, "f_L1": float(np.abs(f - fe).mean()),
            "mass": abs(float(f.sum()) - m0) / m0, "divergence": div, "failed": failed}


# --- the two-phase double shear layer ------------------------------------------------------------------------------
def shear_layer(n=64, ratio=1e3, t_end=2.0, delta=1.0 / 30, conf=None, every=0.1):
    """cases/convection_stability.shear's layer (inviscid, periodic, heavy band 1/4 < y < 3/4, tanh profiles,
    v = 0.05 sin 2 pi x) through ``two_phase(conf)``: the kinetic energy (density-weighted, the clamped density) and
    the total momentum (the density the configuration recovers the velocity with) in time, the divergence."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    band = lambda x, y: np.minimum(y - 0.25, 0.75 - y)  # noqa: E731
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=float((conf or {}).get("tolerance", 1e-6)))
    f0 = to_numpy(fraction(grid, band))
    tp = two_phase(ns, CellField.from_interior(grid, grid.xp.asarray(f0), BoundaryConditions.periodic(2)),
                   (1.0, 1.0 / ratio), (0.0, 0.0), conf)
    x, y = (to_numpy(a) for a in grid.meshgrid())
    u0 = np.where(y <= 0.5, np.tanh((y - 0.25) / delta), np.tanh((0.75 - y) / delta))
    ns.u[0].interior = grid.xp.asarray(u0)
    ns.u[1].interior = grid.xp.asarray(0.05 * np.sin(2.0 * math.pi * x))
    lin = bool((conf or {}).get("fixes"))

    def budget():
        f = to_numpy(tp.f.interior)
        c = f if lin else np.clip(f, 0, 1)
        rho = c * (tp.rho1 - tp.rho2) + tp.rho2
        rc = np.clip(f, 0, 1) * (tp.rho1 - tp.rho2) + tp.rho2
        u, v = to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)
        return (0.5 * float((rc * (u * u + v * v)).sum()) / n ** 2,
                np.array([float((rho * u).sum()), float((rho * v).sum())]) / n ** 2)

    ke0, p0 = budget()
    pref = max(float(np.hypot(*p0)), 0.5 * (tp.rho1 + tp.rho2))     # the total momentum is ~0: a mass scale
    times = [every * (k + 1) for k in range(int(round(t_end / every)))]
    ns.start(events=times)
    series, failed, div = [], None, divergence(ns)
    for te in times:
        while ns.clock.t < te - 1e-12:
            ns.step()
            div = max(div, divergence(ns))
        ke, p = budget()
        series.append({"t": te, "ke_rel": ke / ke0 - 1.0, "momentum": float(np.hypot(*(p - p0))) / pref})
        if not math.isfinite(ke) or ke > 1e3 * ke0:
            failed = "blow-up at t = %.3f" % ns.clock.t
            break
    return {"n": n, "ratio": ratio, "conf": conf or {}, "steps": ns.clock.i, "failed": failed,
            "ke_increase_max": max(s["ke_rel"] for s in series) if series else None,
            "momentum_max": max(s["momentum"] for s in series) if series else None, "divergence": div,
            "series": series}


# --- the values before the R0 changes (#194: recorded, no verdict) -------------------------------------------------
def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, kw = a
    f = {"layers": couette_poiseuille, "bubble": bubble, "static": static_drop, "translate": drop_translation,
         "shear": shear_layer}[kind]
    r = f(**kw)
    r["kind"] = kind
    return r


def jobs():
    J = []
    for conf in ({}, {"fixes": True}):
        for kind in ("couette", "poiseuille"):
            for n in (32, 64):
                for off in (0.0, 0.3):
                    J.append(("layers", dict(kind=kind, n=n, offset=off, conf=conf)))
        for n in (32, 64):
            J.append(("translate", dict(n=n, conf=conf)))
            for cv in ("exact", "numerical"):
                for off in ((0.0, 0.0), (0.3, 0.1)):
                    J.append(("static", dict(n=n, curvature=cv, offset=off, conf=conf)))
        J.append(("shear", dict(n=64, conf=conf)))
        for case in (1, 2):
            J.append(("bubble", dict(case=case, n=32, conf=conf)))
    return J


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_parts.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    key = lambda a: json.dumps([a[0], a[1]], sort_keys=True, default=str)  # noqa: E731
    J = [a for a in jobs() if key(a) not in done]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "16")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job, J, chunksize=1)):
            r["job"] = key(a)
            with open(part, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            done[r["job"]] = r
    with open(out, "w") as fh:
        json.dump(list(done.values()), fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()

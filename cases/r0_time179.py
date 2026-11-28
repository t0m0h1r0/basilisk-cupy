"""#179 (design/179_time_integration.md, PR #186 at 59d8817; B's approval of 2026-11-28): the time-integration
candidates built from hooks on the library (unchanged), and the tests T1 / T2 of the design's section 6.

Configurations (``scheme(name)``; every TwoPhase(conserving=True) built inside gets them):
  c-lib  the library's defaults
  c      the current scheme of the design's 4.1: D1 (conserving_consistent), D2 limited (theta = 3), project_start
  d0     c + conserving_predict's carrying velocity (BCG prediction with the saved g, projection over dt/2, force
         from f^{n+1}), with the walls' normal face velocity of the interpolation (``d0-lib``: the library's option
         as it is, which leaves O(dt) normal velocity on the walls)
  d1     c + the velocity-Verlet form of the design's 4.2: the carrying face velocity
           uf^{n+1/2} = Pi_{dt/2}[alpha^n](B(u^n; 0) + dt/2 a^n),  a^n = a_f[phi^n, f^n; alpha^n]
         (no saved g), g_P = C(a^n - alpha^n grad pf); the face value's source g_P outside the interface band
         (1 cell, note 117) and 0 inside; the viscous step's dt g in / out with g_P (V-BE); the interfacial force
         from (phi^n + phi^{n+1}) / 2 (phi^n the one stored at the previous step's force); the effective CFL number
         of uf^{n+1/2} checked (> 0.5: dt x 0.9 x 0.5 / CFL and the prediction redone, at most twice)
  b2     c + the half-step stage of the design's 4.3 on copies: (f~, u~) = T_{dt/2}[uf^n](f^n, u^n; 0), P(f~),
         V_{dt/2} (with the saved g, the current step's form), a~ = a_f[phi(f~), f~; alpha~],
         uf^{n+1/2} = Pi_{dt/2}[alpha~](I_f u~ + dt/2 a~), g~ = C(a~ - alpha~ grad pi~); then as d1 with g~

T1 (capillary wave, note 122's condition 9, N = 32 / 64, t <= 1.2) and T2 (mu / 100, 2000 steps; static drop with
the numerical curvature, N = 32, 2000 steps). The interface positions of B's reading beta (fixed before the runs):
OFFSETS, fractions of a cell (x phase, mean height). Spread of the solver tolerance and of a small perturbation:
SPREAD.

    python -m cases.r0_time179 t1      # docs/figures/r0_time179_t1.json
    python -m cases.r0_time179 t2      # docs/figures/r0_time179_t2.json
    python -m cases.r0_time179 checks  # the checks after the results (docs/figures/r0_time179_checks.json)
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time

import numpy as np

RHO = (1.0, 1e-3)
MU = (1e-2, 1e-2 / 55.0)
EXACT_PERIOD, EXACT_DAMPING = 0.4025716727004185, 0.7146625675736967   # audit C-2 (docs/figures/capwave_exact_audit.json)
SCHEMES = ("c-lib", "c", "d0", "d0-lib", "d1", "b2")
# B's reading beta (#179): interface positions fixed before the measurement, (x phase, mean height) in cells
OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.25), (0.5, 0.5), (0.37, 0.13))
# the spread of the base position: the solver tolerance and a small perturbation of the phase (1e-3 cell)
SPREAD = (dict(tolerance=1e-9), dict(tolerance=1e-11), dict(dx=1e-3), dict(dx=-1e-3))
BAND = 1
CFL_REDO = 2


# --- the configurations -----------------------------------------------------------------------------------------------
def _combine(pots, tp):
    """The potentials summed as TwoPhase._interfacial_force does (NODATA outside the interface)."""
    from basilisk_cupy.interface.heights import NODATA

    xp = tp.f.grid.xp
    phi = None
    for pot in pots:
        p = pot(tp)
        phi = p if phi is None else xp.where((phi == NODATA) | (p == NODATA), xp.where(phi == NODATA, p, phi), phi + p)
    return phi


def _force(tp, phi, alpha, f):
    """a_f[phi, f; alpha] as TwoPhase._interfacial_force (f clamped), face arrays."""
    from basilisk_cupy.interface.heights import NODATA, pad_bc

    xp = tp.f.grid.xp
    fc = xp.minimum(xp.maximum(f, 0.0), 1.0)
    fp = pad_bc(fc, tp.f.bc, 1)
    php = pad_bc(phi, tp.f.bc, 1)
    n = phi.shape
    h = float(tp.f.grid.h[0])
    out = []
    for ax in range(2):
        lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
        lo[ax] = slice(0, n[ax] + 1)
        hi = list(lo)
        hi[ax] = slice(1, n[ax] + 2)
        fl, fr = fp[tuple(lo)], fp[tuple(hi)]
        pl, pr = php[tuple(lo)], php[tuple(hi)]
        okl, okr = pl < NODATA, pr < NODATA
        phif = xp.where(okl & okr, (pr + pl) / 2.0, xp.where(okr, pr, xp.where(okl, pl, 0.0)))
        out.append(xp.where(fr != fl, alpha[ax] * phif * (fr - fl) / h, 0.0))
    return out


def _cell_accel(ns, a_faces, p):
    """C(a - alpha grad p) through the library's centered_gradient (the walls' rule included), without touching
    ns.g / ns.a / ns.p."""
    from basilisk_cupy.fields import FaceField

    g_save = [c.interior.copy() for c in ns.g.components]
    a_save, p_save, const = ns.a, ns.p, ns.a_is_constant
    ns.a = FaceField(ns.grid, [x for x in a_faces], "a_pred")
    ns.p = p
    try:
        ns.centered_gradient()
        out = [c.interior.copy() for c in ns.g.components]
    finally:
        ns.a, ns.p, ns.a_is_constant = a_save, p_save, const
        for c, g0 in zip(ns.g.components, g_save):
            c.interior = g0
        ns.g.fill_ghosts()
    return out


def _band_mask(tp):
    """1 away from the interface band (cells with 0 < f < 1 and their neighbours within BAND), 0 in it."""
    from cases.r0_combined import _band_cells

    return 1.0 - _band_cells(tp.f.interior, tp, BAND)


def _setup(ns, tp, name, record):
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.heights import NODATA
    from basilisk_cupy.linear.viscosity import viscosity
    from basilisk_cupy.projection.approximate import project
    from basilisk_cupy.terms.convection.bcg import prediction

    grid = tp.f.grid
    xp = grid.xp
    h = float(grid.h[0])
    st = {"phi_n": None, "phi_next": None, "orig": None}
    zero_g = ns.g.copy()
    for c in zero_g.components:
        c.interior = 0.0 * c.interior
    zero_g.fill_ghosts()

    def mid(t):                                                   # d-4: the force from (phi^n + phi^{n+1}) / 2
        cur = _combine(st["orig"], t)
        st["phi_next"] = cur
        prev = st["phi_n"]
        if prev is None:
            return cur
        okc, okp = cur < NODATA / 2, prev < NODATA / 2
        return xp.where(okc & okp, 0.5 * (prev + cur), xp.where(okc, cur, xp.where(okp, prev, NODATA)))

    def capture():
        if name == "d0":
            st["orig"] = list(tp.potentials)
            return
        if not (len(tp.potentials) == 1 and tp.potentials[0] is mid):
            st["orig"] = list(tp.potentials)
            tp.potentials[:] = [mid] if st["orig"] else []

    def a_now():
        """a^n = a_f[phi^n, f^n; alpha^n] (+ no constant force in these tests)."""
        if not st["orig"]:
            return [xp.zeros_like(ns.alpha[ax]) for ax in range(2)]
        if st["phi_n"] is None:
            st["phi_n"] = _combine(st["orig"], tp)
        return _force(tp, st["phi_n"], ns.alpha, tp.f.interior)

    def walls(uf):
        """The normal face velocity on the walls from u's boundary conditions (the interpolation's value: 0 on a
        slip or no-slip wall). BCG's prediction leaves O(dt) there, which the Neumann projection cannot remove
        (the library's conserving_predict does the same: the multigrid then runs to its iteration limit)."""
        ns._apply_uf_bc(uf)
        fv = None
        for ax in range(2):
            if ax in ns.periodic:
                continue
            fv = fv or ns.face_velocity()
            idx = [slice(None)] * 2
            for k in (0, -1):
                idx[ax] = k
                uf[ax][tuple(idx)] = fv[ax][tuple(idx)]
        return uf

    def predict_d1(dt):
        an = a_now()
        uf = walls(prediction(ns.u, zero_g, dt))
        for ax in range(2):
            uf[ax] = uf[ax] + 0.5 * dt * an[ax]
        walls(uf)
        ns.mgpf = project(uf, ns.pf, ns.alpha, dt / 2.0, ops=ns.ops, nrelax=ns.mgpf.nrelax or 4,
                          tolerance=ns.tolerance, basilisk_tolerance=True)
        walls(uf)
        record["mgpf_max"] = max(record.get("mgpf_max", 0), int(getattr(ns.mgpf, "i", 0) or 0))
        return uf, _cell_accel(ns, an, ns.pf)

    def predict_d0(dt):
        """conserving_predict's prediction (the saved g, the projection over dt / 2), with the walls' normal face
        velocity as above."""
        uf = walls(prediction(ns.u, ns.g, dt))
        ns.mgpf = project(uf, ns.pf, ns.alpha, dt / 2.0, ops=ns.ops, nrelax=ns.mgpf.nrelax or 4,
                          tolerance=ns.tolerance, basilisk_tolerance=True)
        walls(uf)
        return uf, None

    def predict_b2(dt):
        tau = 0.5 * dt
        f0 = tp.f.interior.copy()
        u0 = [c.interior.copy() for c in ns.u.components]
        al0 = [ns.alpha[ax].copy() for ax in range(2)]
        mu0 = [ns.mu[ax].copy() for ax in range(2)] if ns.mu is not None else None
        rho0 = ns.rho.copy() if hasattr(ns.rho, "copy") else ns.rho
        dt0, src0 = ns.clock.dt, tp.conserving_source
        ptil = ns.p.copy()
        try:
            ns.clock.dt = tau
            tp.conserving_source = False
            tp._vof(ns)                                           # T_{dt/2}[uf^n](f^n, u^n; 0)
            tp._properties(ns)                                    # P(f~)
            if ns.mu is not None:                                 # V_{dt/2}, the current step's form (saved g)
                ns.correction(tau)
                viscosity(ns.u, ns.mu, ns.rho, tau, tolerance=ns.tolerance)
                ns.correction(-tau)
            if st["orig"]:
                at = _force(tp, _combine(st["orig"], tp), ns.alpha, tp.f.interior)
            else:
                at = [xp.zeros_like(ns.alpha[ax]) for ax in range(2)]
            ns.u.fill_ghosts()
            uf = FaceField(grid, [ns.ops.interp_f(ns.u[ax], ax) + tau * at[ax] for ax in range(2)], "uf_half")
            ns._apply_uf_bc(uf)
            project(uf, ptil, ns.alpha, tau, ops=ns.ops, nrelax=4, tolerance=ns.tolerance, basilisk_tolerance=True)
            ns._apply_uf_bc(uf)
            gt = _cell_accel(ns, at, ptil)
        finally:
            ns.clock.dt, tp.conserving_source = dt0, src0
            tp.f.interior = f0
            tp.f.fill_ghosts()
            for c, v in zip(ns.u.components, u0):
                c.interior = v
            ns.u.fill_ghosts()
            for ax in range(2):
                ns.alpha[ax] = al0[ax]
                if mu0 is not None:
                    ns.mu[ax] = mu0[ax]
            ns.rho = rho0
        return uf, gt

    orig_stability = ns._stability

    def stability():
        capture()
        prev = ns.clock.previous
        dtmax = ns.dtmax
        redo = 0
        while True:
            orig_stability()
            dt = ns.clock.dt
            uf, g = {"d1": predict_d1, "b2": predict_b2, "d0": predict_d0}[name](dt)
            cfl = max(float(xp.abs(uf[ax]).max()) for ax in range(2)) * dt / h
            record["cfl_max"] = max(record.get("cfl_max", 0.0), cfl)
            if name == "d0" or cfl <= ns.cfl * (1 + 1e-12) or redo >= CFL_REDO:
                break
            redo += 1
            record["redo"] = record.get("redo", 0) + 1
            ns.dtmax = min(dtmax, 0.9 * ns.cfl / cfl * dt)
            ns.clock.previous = prev
        if cfl > ns.cfl * (1 + 1e-12) and name != "d0":
            record["cfl_over"] = record.get("cfl_over", 0) + 1
        st["uf_half"], st["g"] = uf, g

    ns._stability = stability

    def before(solver):                                           # "vof", before TwoPhase's transport
        solver.uf = st["uf_half"]
        if name == "d0":
            return
        tp.conserving_source = True
        w = _band_mask(tp)
        for c, g in zip(solver.g.components, st["g"]):
            c.interior = g * w
        solver.g.fill_ghosts()

    def after(solver):                                            # "tracer_advection": V-BE's dt g is g_P (g~)
        if name == "d0":
            return
        for c, g in zip(solver.g.components, st["g"]):
            c.interior = g
        solver.g.fill_ghosts()

    def end(solver):
        if st["orig"] and name != "d0":
            st["phi_n"] = st["phi_next"]

    ns.on("vof", before)
    ns.on("tracer_advection", after)
    ns.on("end_timestep", end, after=True)


@contextlib.contextmanager
def scheme(name, record=None):
    from basilisk_cupy.interface import two_phase

    if name not in SCHEMES:
        raise ValueError(name)
    rec = {} if record is None else record
    init = two_phase.TwoPhase.__init__

    def patched(self, ns, f, **k):
        if k.get("conserving") and name != "c-lib":
            k.update(conserving_consistent=True, conserving_slope_limit=3.0)
            if name == "d0-lib":
                k["conserving_predict"] = True
        init(self, ns, f, **k)
        if k.get("conserving") and name != "c-lib":
            ns.project_start = True
            if name in ("d0", "d1", "b2"):
                _setup(ns, self, name, rec)

    two_phase.TwoPhase.__init__ = patched
    try:
        yield rec
    finally:
        two_phase.TwoPhase.__init__ = init


# --- G: the gain of the face -> cell -> face average of the net face acceleration (T1's record) ------------------------
def _gain(ns, tp, acc):
    """<I_f C G_f, G_f> / <G_f, G_f> per component on the faces within 2 cells of the interface,
    G_f = a - alpha grad p of the last projection (ns.g = C(G_f))."""
    from cases.r0_combined import _band_cells

    xp = ns.grid.xp
    cells = _band_cells(tp.f.interior, tp, 2)
    per = tp.f.bc.periodic_axes(2)
    ns.p.fill_ghosts()
    for ax in range(2):
        gf = ns.a[ax] - ns.alpha[ax] * ns.ops.grad_f(ns.p, ax)
        igf = ns.ops.interp_f(ns.g[ax], ax)
        pad = [(0, 0), (0, 0)]
        pad[ax] = (1, 1)
        cp = xp.pad(cells, pad, mode="wrap" if per[ax] else "edge")
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        m = xp.maximum(cp[tuple(lo)], cp[tuple(hi)])
        if not per[ax]:                                           # leave out the wall faces (zero by the wall rule)
            m = m.copy()
            idx = [slice(None)] * 2
            idx[ax] = 0
            m[tuple(idx)] = 0.0
            idx[ax] = -1
            m[tuple(idx)] = 0.0
        acc[ax][0] += float((m * igf * gf).sum())
        acc[ax][1] += float((m * gf * gf).sum())


# --- T1 / T2: the capillary wave ---------------------------------------------------------------------------------------
def capillary_wave(name, n=32, t_end=1.2, a0=0.01, dx=0.0, dy=0.0, tolerance=1e-10, mu_scale=1.0, steps=None,
                   gain_until=EXACT_PERIOD):
    """y = dy h + a0 cos 2 pi (x - dx h) between slip walls at y = +-0.5 (periodic in x), heavy below, sigma = 1,
    rho 1 / 1e-3, mu (1e-2, 1e-2 / 55) x mu_scale. The amplitude: the first Fourier mode of the column heights
    (Basilisk's heights) along cos 2 pi (x - dx h); and (dx = dy = 0) the column x = h/2 as note 122.
    ``steps``: run that many steps instead (T2: max |u| of every step)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Neumann, Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA, position
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    h = 1.0 / n
    x0, y0 = dx * h, dy * h
    rec = {}
    t0 = time.time()
    with scheme(name, rec):
        grid = CartesianGrid((n, n), origin=(0.0, -0.5))
        ns = CenteredNS(grid, periodic=(0,), tolerance=tolerance, mu=MU[0] * mu_scale)
        bc = BoundaryConditions()
        for side in (0, 1):
            bc.set((0, side), Periodic())
            bc.set((1, side), Neumann(0.0))
        f0 = to_numpy(fraction(grid, lambda x, y: y0 + a0 * np.cos(2 * math.pi * (x - x0)) - y, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=RHO[0], rho2=RHO[1], mu1=MU[0] * mu_scale,
                      mu2=MU[1] * mu_scale, conserving=True)
        tp.add_surface_tension(1.0)
        xc = (np.arange(n) + 0.5) * h
        mode = np.cos(2 * math.pi * (xc - x0))

        def amplitude():
            p = to_numpy(position(tp.f.interior, tp.f.bc, grid, (0.0, 1.0), (0.0, 0.0)))
            ok = p < NODATA / 2
            col = np.array([p[i, ok[i]].mean() if ok[i].any() else np.nan for i in range(n)])
            return float(2.0 / n * np.sum((col - y0) * mode)), float(col[0] - y0)

        gacc = [[0.0, 0.0], [0.0, 0.0]]
        amp, umax, failed = [], [], None
        if steps is None:
            samples = list(np.arange(1, int(round(t_end / 0.004)) + 1) * 0.004)
            ns.start(events=samples)
        else:
            samples = None
            ns.start()
        try:
            if samples is not None:
                for te in samples:
                    while ns.clock.t < te - 1e-12:
                        ns.step()
                        if ns.clock.t <= gain_until + 1e-12:
                            _gain(ns, tp, gacc)
                    a_f, a_c = amplitude()
                    amp.append((te, a_f, a_c))
                    if not (math.isfinite(a_f) and abs(a_f) < 0.5):
                        failed = "lost at t = %.3f" % te
                        break
            else:
                for _ in range(steps):
                    ns.step()
                    u = float(max(np.abs(to_numpy(c.interior)).max() for c in ns.u.components))
                    umax.append(u)
                    if not math.isfinite(u) or u > 1e2:
                        failed = "blow-up at step %d" % ns.clock.i
                        break
        except Exception as exc:  # noqa: BLE001
            failed = repr(exc)[:200]
    out = {"case": "capillary_wave", "scheme": name, "n": n, "dx": dx, "dy": dy, "tolerance": tolerance,
           "mu_scale": mu_scale, "failed": failed, "steps": ns.clock.i, "t": float(ns.clock.t),
           "seconds": time.time() - t0, "record": rec, "rho_floor_hits": tp.rho_floor_hits}
    if samples is not None:
        out["gain"] = [gacc[ax][0] / gacc[ax][1] if gacc[ax][1] else None for ax in range(2)]
        for key, k in (("", 1), ("_column", 2)):
            if key and (dx or dy):
                continue
            t = np.array([a[0] for a in amp])
            a = np.array([a[k] for a in amp])
            if len(a) > 20 and failed is None:
                zc = [t[j] - a[j] * (t[j + 1] - t[j]) / (a[j + 1] - a[j]) for j in range(len(a) - 1)
                      if a[j] * a[j + 1] < 0]
                out["period" + key] = float(2 * np.mean(np.diff(zc))) if len(zc) > 2 else None
                ext = [j for j in range(1, len(a) - 1) if abs(a[j]) >= abs(a[j - 1]) and abs(a[j]) >= abs(a[j + 1])]
                if len(ext) > 2:
                    out["damping" + key] = float(-np.polyfit(t[ext], np.log(np.abs(a[ext])), 1)[0])
        out["amplitude"] = amp
    else:
        out.update(_growth(umax))
    return out


def _growth(umax):
    """T2's reading (B, #179): the largest max|u| of the last 1000 steps over that of the first 1000."""
    u = np.array(umax)
    if len(u) < 2000:
        return {"umax_first": float(u.max()) if len(u) else None, "umax_last": None, "growth": None,
                "umax_trace": u[:: 20].tolist()}
    first, last = float(u[:1000].max()), float(u[-1000:].max())
    return {"umax_first": first, "umax_last": last, "growth": last / first, "umax_trace": u[:: 20].tolist()}


def static_drop(name, n=32, steps=2000, mu_scale=0.01, R=0.2, tolerance=1e-10, dt_factor=1.0):
    """T2: R = 0.2 at the centre of the periodic unit square, numerical curvature, the capillary wave's properties
    with mu x mu_scale; max |u| of every step."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    rec = {}
    t0 = time.time()
    umax, failed = [], None
    with scheme(name, rec):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance, mu=MU[0] * mu_scale)
        f0 = to_numpy(fraction(grid, lambda x, y: R * R - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=RHO[0], rho2=RHO[1],
                      mu1=MU[0] * mu_scale, mu2=MU[1] * mu_scale, conserving=True)
        tp.add_surface_tension(1.0)
        tp._curvature_dt_factor = dt_factor                       # the capillary step x dt_factor (a check)
        ns.start()
        try:
            for _ in range(steps):
                ns.step()
                u = float(max(np.abs(to_numpy(c.interior)).max() for c in ns.u.components))
                umax.append(u)
                if not math.isfinite(u) or u > 1e2:
                    failed = "blow-up at step %d" % ns.clock.i
                    break
        except Exception as exc:  # noqa: BLE001
            failed = repr(exc)[:200]
    out = {"case": "static_drop", "scheme": name, "n": n, "mu_scale": mu_scale, "dt_factor": dt_factor, "failed": failed, "steps": ns.clock.i,
           "t": float(ns.clock.t), "seconds": time.time() - t0, "record": rec}
    out.update(_growth(umax))
    return out


# --- drivers ------------------------------------------------------------------------------------------------------------
def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, kw = a
    return {"capillary_wave": capillary_wave, "static_drop": static_drop}[kind](**kw)


def t1_jobs(schemes=("c", "d0", "d1", "b2", "c-lib")):
    J = []
    for n in (64, 32):
        for name in schemes:
            for dx, dy in OFFSETS:
                J.append(("capillary_wave", dict(name=name, n=n, dx=dx, dy=dy)))
            if name in ("c", "d1", "b2"):
                for s in SPREAD:
                    J.append(("capillary_wave", dict(name=name, n=n, **s)))
        J.append(("capillary_wave", dict(name="d0-lib", n=n)))
    return J


def t2_jobs(schemes=("c", "d1", "b2")):
    J = []
    for name in schemes:
        for n in (64, 32):
            J.append(("capillary_wave", dict(name=name, n=n, mu_scale=0.01, steps=2000)))
        J.append(("static_drop", dict(name=name, n=32)))
    return J


def run(J, path):
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "26"))) as pool:
        res = []
        for r in pool.imap_unordered(_job, J):
            res.append(r)
            print({k: v for k, v in r.items() if k not in ("amplitude", "umax_trace")}, flush=True)
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1, default=float)
    return res


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "t1"
    if which == "t1":
        run(t1_jobs(), os.path.join("docs", "figures", "r0_time179_t1.json"))
    elif which == "t2":
        run(t2_jobs(), os.path.join("docs", "figures", "r0_time179_t2.json"))
    elif which == "checks":                                       # after the results: note 125's checks
        J = [("static_drop", dict(name=nm, n=32, steps=4000, dt_factor=0.5)) for nm in ("c", "d1")]
        J += [("static_drop", dict(name="c", n=32, steps=2000, mu_scale=1.0))]
        J += [("capillary_wave", dict(name="d0", n=64, dx=0.5, dy=0.0)), ("capillary_wave", dict(name="d0", n=64)),
              ("capillary_wave", dict(name="d0-lib", n=32))]
        run(J, os.path.join("docs", "figures", "r0_time179_checks_rerun.json"))
    else:
        raise SystemExit("t1 | t2 | checks")


if __name__ == "__main__":
    main()


# --- reading (fixed before the results: design/179 section 6 and B's reading beta) ------------------------------------
D0_PRED = {32: 0.875, 64: 0.777}                     # design/179 T1: d0's damping, +- 0.02
D_PRED = {32: 0.005, 64: 0.001}                      # design/179 T1: |d1 - c|, |b2 - c| at most this


def summary_t1(path=os.path.join("docs", "figures", "r0_time179_t1.json")):
    res = json.load(open(path))
    base = {(r["scheme"], r["n"], r["dx"], r["dy"]): r for r in res
            if r["tolerance"] == 1e-10 and (r["dx"], r["dy"]) in [tuple(o) for o in OFFSETS]}
    out = {}
    for n in (32, 64):
        for name in ("c", "c-lib", "d0", "d1", "b2"):
            rows = [base.get((name, n) + tuple(o)) for o in OFFSETS]
            if any(r is None or r.get("damping") is None for r in rows):
                out[(name, n)] = None
                continue
            dmp = [r["damping"] for r in rows]
            per = [r["period"] for r in rows]
            out[(name, n)] = {
                "damping": dmp, "period": per,
                "worst_damping_error": max(abs(d - EXACT_DAMPING) / EXACT_DAMPING for d in dmp),
                "worst_period_error": max(abs(p - EXACT_PERIOD) / EXACT_PERIOD for p in per),
                "gain": [r["gain"] for r in rows], "record": [r["record"] for r in rows],
                "seconds": [r["seconds"] for r in rows], "steps": [r["steps"] for r in rows]}
        # the spread of c, d1, b2 at the base position (tolerance 1e-9 / 1e-11, phase +-1e-3 cell)
        for name in ("c", "d1", "b2"):
            b = base.get((name, n, 0.0, 0.0))
            sp = [r for r in res if r["scheme"] == name and r["n"] == n and r not in base.values()]
            if b and b.get("damping") and sp and all(r.get("damping") for r in sp):
                out[(name, n, "spread")] = {
                    "damping": max(abs(r["damping"] - b["damping"]) for r in sp),
                    "period": max(abs(r["period"] - b["period"]) for r in sp)}
    return out


def summary_t2(path=os.path.join("docs", "figures", "r0_time179_t2.json")):
    res = json.load(open(path))
    return [{k: r[k] for k in ("case", "scheme", "n", "failed", "steps", "t", "umax_first", "umax_last", "growth",
                               "record", "seconds")} for r in res]

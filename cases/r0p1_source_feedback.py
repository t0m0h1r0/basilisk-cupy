"""docs/notes/114 (R0-P1, issue #141, B's #145 P0): why the half-step pressure source in the face value of the
conserving momentum transport (``conserving_source``, docs/notes/77 and 112) is unstable at high density ratios.

The mechanism (section B of the note): the fluxes carry the leaving mass with the velocity u + dt/2 g, so a cell
keeps the momentum  M* = M^n - sum (mass out) (u + dt/2 g) + ...; for a uniform flow U and a uniform g

    u* - U = (dt / 2) g (1 - m^n / m^{n+1})                                   (K)

exactly, for one sweep: the half impulse is taken away with the OLD mass while the velocity is formed with the NEW
one. A cell the heavy phase has just left has m^n / m^{n+1} up to the density ratio.

  kick()         (K) against one transport step (one sweep: exact; two sweeps: per phase).
  one_sweep()    the same with a random g: the exact formula (F) of the face source (mass OUT over the new mass)
                 and the bounded one (A) of "accel" (mass IN over the new mass).
  one_step_gain()  the spectral radius of the one-step map g^n -> g^{n+1} with the geometry frozen (power iteration).
  growth()       the growth per step of a round-off perturbation of a drop carried by a uniform flow, for four ways
                 of putting the same half-step acceleration into the transport (interventions, all from hooks):
                   "face"    the library's source in the face value;
                   "face-upwind", "face-massw"  the same with the donor cell's g / the density-weighted mean;
                   "face+m"  the same, and (K) taken back from every cell after the transport;
                   "force"   M += dt/2 rho^n g before the transport, M -= dt/2 rho^n g after (the old mass);
                   "accel"   u += dt/2 g before the transport, u -= dt/2 g after (the new mass);
                   "trapezoid"  u += dt/2 G before the transport and nothing taken back: the projection then
                             supplies only the other half, dt g~ = dt/2 G^{n+1} (G = 2 g~; a first look only).
  single_phase() the accuracy of "face" and "accel" on the steady Taylor-Green flow (docs/notes/112).
  vortex()       a drop in a Taylor-Green vortex, density ratio 1000: stability and the momentum budget.

    python -m cases.r0p1_source_feedback          # docs/figures/r0p1_source_feedback.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

TWO_PI = 2 * math.pi
U0 = (1.0, 0.37)
MODES = ("none", "face", "face+m", "force", "accel", "trapezoid")


def _pbc():
    from basilisk_cupy import BoundaryConditions
    from basilisk_cupy.boundary import Periodic

    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    return bc


def _drop(grid, r=0.15, c=(0.5, 0.5)):
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction

    return to_numpy(fraction(grid, lambda x, y: r * r - (x - c[0]) ** 2 - (y - c[1]) ** 2, refine=8))


def _bubble(grid):
    """The light phase inside (a bubble of radius 0.15 in the heavy phase)."""
    return 1.0 - _drop(grid)


def _band(grid):
    """An oblique periodic band of the heavy phase: 0.2 < frac(x + 2 y) < 0.6 (the interfaces along x + 2 y)."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction

    def phi(x, y):
        s = (x + 2.0 * y) % 1.0
        return 0.2 - abs(s - 0.4)

    return to_numpy(fraction(grid, phi, refine=8))


SHAPES = {"drop": _drop, "bubble": _bubble, "band": _band}


class face_average:
    """Context manager (the library is not changed): the face value's source as another average of the two cells'
    g than the library's arithmetic mean, by rewriting that one line of ``vof._sweep``.
      "upwind"  the donor cell's g;   "massw"  the density-weighted mean (rho_L g_L + rho_R g_R) / (rho_L + rho_R)."""

    LINE = "ff = ff + 0.25 * dt * (_faces(sp, axis, 0) + _faces(sp, axis, 1))"
    NEW = {"upwind": "ff = ff + 0.5 * dt * upwind(sp)",
           "massw": ("rk_ = _RHO[1] if tr.inverse else _RHO[0]\n"
                     "            cq_ = xp.minimum(xp.maximum(cp, 0.0), 1.0)\n"
                     "            rc_ = _RHO[0] * cq_ + _RHO[1] * (1.0 - cq_)\n"
                     "            fo_ = rc_ * sp / rk_\n"
                     "            ff = ff + 0.5 * dt * rk_ * (_faces(fo_, axis, 0) + _faces(fo_, axis, 1))"
                     " / (_faces(rc_, axis, 0) + _faces(rc_, axis, 1))")}

    def __init__(self, kind, ratio):
        self.kind, self.ratio = kind, ratio

    def __enter__(self):
        import inspect

        import basilisk_cupy.interface.vof as vof

        self.vof, self.saved = vof, vof._sweep
        if self.kind in self.NEW:
            src = inspect.getsource(vof._sweep)
            assert src.count(self.LINE) == 1
            vof._RHO = (1.0, 1.0 / self.ratio)
            ns_ = {}
            exec(src.replace(self.LINE, self.NEW[self.kind]), vof.__dict__, ns_)
            vof._sweep = ns_["_sweep"]
        return self

    def __exit__(self, *exc):
        self.vof._sweep = self.saved
        return False


def solver(n, ratio, mode, f0=None, tolerance=1e-10):
    """CenteredNS + TwoPhase(conserving=True) with the half-step acceleration put in as ``mode`` (hooks only)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance)
    f = CellField.from_interior(grid, np.zeros((n, n)) if f0 is None else f0(grid), _pbc())
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, conserving=True,
                  conserving_source=mode in ("face", "face+m", "face-upwind", "face-massw"))
    st = {}

    def before(s):                                               # "vof", before TwoPhase's hook
        dt = s.clock.dt
        st["rho"] = tp.rho(tp.f.interior).copy()
        st["g"] = [g.interior.copy() for g in s.g.components]
        if mode in ("force", "accel"):
            for u, g in zip(s.u.components, st["g"]):
                u.interior = u.interior + 0.5 * dt * g
            s.u.fill_ghosts()
        if mode == "trapezoid" and "G" in st:                    # the half impulse stays in: nothing is taken back
            for u, g in zip(s.u.components, st["G"]):
                u.interior = u.interior + 0.5 * dt * g
            s.u.fill_ghosts()
        if mode == "trapezoid-p" and "phi" in st:
            # docs/notes/116: the half impulse from the MEAN of the last two pressures, with the face coefficients
            # and the cell average of the present densities (a force that sums to zero: conserving)
            phi = st["phi"] if "phi_prev" not in st else 0.5 * (st["phi"] + st["phi_prev"])
            keep = s.p.data.copy()
            s.p.data[...] = phi
            s.p.fill_ghosts()
            for ax, u in enumerate(s.u.components):
                gf = -s.alpha[ax] * s.ops.grad_f(s.p, ax)
                u.interior = u.interior + 0.5 * dt * s.ops.face_to_cell(gf, ax, periodic=True)
            s.p.data[...] = keep
            s.u.fill_ghosts()

    def after(s):                                                # "tracer_advection": just after the transport
        dt = s.clock.dt
        rho = tp.rho(tp.f.interior)
        w = {"force": -st["rho"] / rho, "accel": -1.0, "face+m": st["rho"] / rho - 1.0}.get(mode)
        if w is not None:
            for u, g in zip(s.u.components, st["g"]):
                u.interior = u.interior + 0.5 * dt * w * g
            s.u.fill_ghosts()

    def end(s):                                                  # "trapezoid": the projection gave dt g~ = dt/2 G^{n+1}
        first = "G" not in st                                    # (the first step is the library's: G^1 = g~)
        st["G"] = [(1.0 if first else 2.0) * g.interior.copy() for g in s.g.components]

    if mode not in ("none", "trapezoid-p"):
        ns.on("vof", before)
        ns.on("tracer_advection", after)
    def end_p(s):                                                # "trapezoid-p": the potential of dt/2 G^{n+1}
        if "phi" in st:
            st["phi_prev"] = st["phi"]
        st["phi"] = (2.0 if "phi" in st else 1.0) * s.p.data.copy()   # (the first step is the library's)

    if mode == "trapezoid":
        ns.on("end_timestep", end, after=True)
    if mode == "trapezoid-p":
        ns.on("vof", before)
        ns.on("end_timestep", end_p, after=True)
    return grid, ns, tp


# ----------------------------------------------------------------------------------------------- (K), one step
def kick(ratio, U=(1.0, 0.0), n=32, cfl=0.5, g0=(0.7, -0.3)):
    """One transport step (``TwoPhase._vof``) of a drop in the uniform flow U with a uniform g0 as the face
    source: u* - U against (K). U along x: only the x sweep moves mass and (K) is exact."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.time.stepping import Clock

    grid, ns, tp = solver(n, ratio, "face", f0=_drop)
    xp = grid.xp
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
        ns.g[ax].interior = g0[ax] + 0.0 * ns.g[ax].interior
    ns.u.fill_ghosts()
    ns.g.fill_ghosts()
    ns._hooks["vof"] = [h for h in ns._hooks["vof"] if getattr(h, "__self__", None) is tp]   # TwoPhase's only
    ns.uf.components[0][...] = xp.full((n + 1, n), U[0])
    ns.uf.components[1][...] = xp.full((n, n + 1), U[1])
    dt = cfl / n / max(abs(U[0]), abs(U[1]))
    ns.clock = Clock(dt=dt, i=0)
    m0 = to_numpy(tp.rho(tp.f.interior)).copy()
    tp._vof(ns)
    m1 = to_numpy(tp.rho(tp.f.interior))
    out = {"ratio": ratio, "U": U, "n": n, "cfl": cfl, "mass_ratio_max": float((m0 / m1).max()),
           "half_step": 0.5 * dt * max(abs(v) for v in g0)}
    du = [to_numpy(ns.u[ax].interior) - U[ax] for ax in range(2)]
    pred = [0.5 * dt * g0[ax] * (1.0 - m0 / m1) for ax in range(2)]
    out["kick_max"] = float(max(np.abs(d).max() for d in du))
    out["formula_max"] = float(max(np.abs(p).max() for p in pred))
    out["difference_max"] = float(max(np.abs(d - p).max() for d, p in zip(du, pred)))
    big = (m0 / m1) > 2.0                                        # cells that lost more than half of their mass
    tot = sum(float((d * d).sum()) for d in du)
    out["share_in_emptied_cells"] = float(sum(float((d[big] ** 2).sum()) for d in du) / tot) if tot > 0 else 0.0
    out["emptied_cells"] = int(big.sum())
    return out


def one_sweep(ratio, n=32, cfl=0.5, seed=0):
    """One transport step in the uniform flow (1, 0) (only the x sweep moves mass) with a RANDOM cell field g
    (|g| <= 1) as the half-step acceleration, put in as the face source and as "accel". With the mass fluxes
    rebuilt from the densities before and after (out_i - out_{i-1} = m_i - m'_i, out = rho2 C in the light phase):

        face:   u*_i - U = (dt/2) [m_in (g_{i-1} + g_i)/2 - m_out (g_i + g_{i+1})/2] / m'_i          (F)  exact
        accel:  u*_i - U ~ (dt/2) (m_in / m'_i) (g_{i-1} - g_i)                                      (A)  donor values
    (F) has m_out / m' (up to the density ratio), (A) only m_in / m' <= 1."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.time.stepping import Clock

    rng = np.random.default_rng(seed)
    g = rng.uniform(-1.0, 1.0, (n, n))
    dt = cfl / n
    out = {"ratio": ratio, "n": n, "cfl": cfl, "half_step": 0.5 * dt}
    for mode in ("face", "accel"):
        grid, ns, tp = solver(n, ratio, "face" if mode == "face" else "none", f0=_drop)
        xp = grid.xp
        ns.u[0].interior = 1.0 + (0.5 * dt * g if mode == "accel" else 0.0) + 0.0 * ns.u[0].interior
        ns.u[1].interior = 0.0 * ns.u[1].interior
        ns.g[0].interior = xp.asarray(g)
        ns.g[1].interior = 0.0 * ns.g[1].interior
        ns.u.fill_ghosts()
        ns.g.fill_ghosts()
        ns.uf.components[0][...] = xp.full((n + 1, n), 1.0)
        ns.uf.components[1][...] = xp.zeros((n, n + 1))
        ns.clock = Clock(dt=dt, i=0)
        m0 = to_numpy(tp.rho(tp.f.interior)).copy()
        tp._vof(ns)
        m1 = to_numpy(tp.rho(tp.f.interior))
        du = to_numpy(ns.u[0].interior) - 1.0 - (0.5 * dt * g if mode == "accel" else 0.0)
        mout = np.cumsum(m0 - m1, axis=0) + cfl / ratio           # column 0 is in the light phase
        mout = mout + (cfl / ratio - mout[-1])                    # (closes the periodic row to round-off)
        min_ = np.roll(mout, 1, axis=0)
        gl, gr = np.roll(g, 1, axis=0), np.roll(g, -1, axis=0)
        if mode == "face":
            pred = 0.5 * dt * (min_ * (gl + g) / 2 - mout * (g + gr) / 2) / m1
        else:
            pred = 0.5 * dt * (min_ / m1) * (gl - g)
        out[mode] = {"max": float(np.abs(du).max()), "formula_max": float(np.abs(pred).max()),
                     "difference_max": float(np.abs(du - pred).max()),
                     "max_over_half_step": float(np.abs(du).max() / (0.5 * dt)),
                     "mout_over_m_max": float((mout / m1).max()), "min_over_m_max": float((min_ / m1).max())}
    return out


# ------------------------------------------------------------------------------------------- growth of round-off
def growth(ratio, mode, n=32, cfl=0.5, steps=200, U=U0, shape="drop"):
    """A drop (R = 0.15) carried by the uniform flow U, no surface tension (docs/notes/77, P0-1): max |u - U| per
    step, and the growth per step between 1e-9 and 1e-3 (nan if it never gets there)."""
    from basilisk_cupy.backend import to_numpy

    if mode in ("face-upwind", "face-massw"):                    # the same source, averaged differently
        with face_average(mode[5:], ratio):
            return dict(growth(ratio, "face", n, cfl, steps, U, shape), mode=mode)
    grid, ns, tp = solver(n, ratio, mode, f0=SHAPES[shape])
    ns.cfl = cfl
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
    ns.start()
    es, where = [], []
    for _ in range(steps):
        rho0 = to_numpy(tp.rho(tp.f.interior)).copy()
        ns.step()
        d = np.maximum(*[np.abs(to_numpy(ns.u[ax].interior) - U[ax]) for ax in range(2)])
        es.append(float(d.max()))
        k = np.unravel_index(d.argmax(), d.shape)
        where.append(float(rho0[k] / to_numpy(tp.rho(tp.f.interior))[k]))
        if not math.isfinite(es[-1]) or es[-1] > 1e2:
            break
    k1 = next((i for i, e in enumerate(es) if e > 1e-9), None)
    k2 = next((i for i, e in enumerate(es) if e > 1e-3), None)
    rate = (es[k2] / es[k1]) ** (1.0 / (k2 - k1)) if k1 is not None and k2 is not None and k2 > k1 else float("nan")
    grow = [w for e, w in zip(es, where) if 1e-9 < e < 1e-3]
    return {"ratio": ratio, "mode": mode, "n": n, "cfl": cfl, "shape": shape, "steps": len(es), "max": max(es), "last": es[-1],
            "rate": rate, "unstable": bool(max(es) > 1e-3),
            "max_cell_emptied_share": float(np.mean([w > 2.0 for w in grow])) if grow else float("nan")}


# ------------------------------------------------------------- the one-step map g^n -> g^{n+1}, frozen geometry
def one_step_gain(ratio, n=32, cfl=0.5, steps=(4, 8, 12, 16, 20, 24), mode="face", iters=60, U=U0):
    """The linear map g^n -> g^{n+1} of ONE step around the exact state (u = U, g = 0) of the drop carried by U,
    at the steps ``steps`` of the run (the base run has no source: it stays at round-off). The step's geometry
    (f, uf = U) does not depend on g, so the map is linear; its spectral radius is found by power iteration, the
    same step repeated from the saved state. Returned per step: the radius, and the largest mass ratios of the step
    (m^n / m^{n+1} - 1 is the factor of the formulas (K), (F))."""
    from basilisk_cupy.backend import to_numpy

    grid, ns, tp = solver(n, ratio, mode, f0=_drop)
    xp = grid.xp
    ns.cfl = cfl
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
    src = tp.conserving_source
    hooks = {k: list(v) for k, v in ns._hooks.items()}

    def plain(on):                                               # the base run: no half-step acceleration at all
        tp.conserving_source = src and not on
        for k in ("vof", "tracer_advection"):
            ns._hooks[k] = [h for h in hooks[k] if getattr(h, "__self__", None) is not None] if on else list(hooks[k])

    plain(True)
    ns.start()
    rng = np.random.default_rng(0)
    out = []
    for k in range(max(steps) + 1):
        if k in steps:
            plain(False)
            save = {"f": tp.f.data.copy(), "p": ns.p.data.copy(), "pf": ns.pf.data.copy(),
                    "uf": [c.copy() for c in ns.uf.components], "clock": (ns.clock.t, ns.clock.i, ns.clock.dt,
                                                                          ns.clock.tnext, ns.clock.previous),
                    "dtmax": ns.dtmax}
            v = [rng.standard_normal((n, n)) for _ in range(2)]
            norms, ratio_m = [], None
            for _ in range(iters):
                nv = math.sqrt(sum(float((a * a).sum()) for a in v))
                tp.f.data[...] = save["f"]
                ns.p.data[...] = save["p"]
                ns.pf.data[...] = save["pf"]
                for c, a in zip(ns.uf.components, save["uf"]):
                    c[...] = a
                ns.clock.t, ns.clock.i, ns.clock.dt, ns.clock.tnext, ns.clock.previous = save["clock"]
                ns.dtmax = save["dtmax"]
                eps = 1e-6 / nv
                for ax in range(2):
                    ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
                    ns.g[ax].interior = xp.asarray(eps * v[ax])
                ns.u.fill_ghosts()
                ns.g.fill_ghosts()
                m0 = to_numpy(tp.rho(tp.f.interior)).copy()
                ns.step()
                if ratio_m is None:
                    ratio_m = float((m0 / to_numpy(tp.rho(tp.f.interior))).max())
                v = [to_numpy(ns.g[ax].interior) / eps for ax in range(2)]
                norms.append(math.sqrt(sum(float((a * a).sum()) for a in v)) / nv)
            out.append({"step": k, "radius": float(np.exp(np.mean(np.log(norms[-20:])))), "first": norms[0],
                        "mass_ratio_max": ratio_m})
            tp.f.data[...] = save["f"]                           # back to the base run
            ns.p.data[...] = save["p"]
            ns.pf.data[...] = save["pf"]
            for c, a in zip(ns.uf.components, save["uf"]):
                c[...] = a
            ns.clock.t, ns.clock.i, ns.clock.dt, ns.clock.tnext, ns.clock.previous = save["clock"]
            ns.dtmax = save["dtmax"]
            for ax in range(2):
                ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
                ns.g[ax].interior = 0.0 * ns.g[ax].interior
            ns.u.fill_ghosts()
            ns.g.fill_ghosts()
            plain(True)
        ns.step()
    r = [o["radius"] for o in out]
    return {"ratio": ratio, "n": n, "cfl": cfl, "mode": mode, "steps": out,
            "radius_geometric_mean": float(np.exp(np.mean(np.log(r)))), "radius_max": max(r), "radius_min": min(r)}


# -------------------------------------------------------------------------------------------- single-phase accuracy
def single_phase(n, mode, cfl=0.5, t_end=0.5):
    """The steady inviscid Taylor-Green flow (docs/notes/112, section 4): max |u - u0| at t_end."""
    from basilisk_cupy.backend import to_numpy

    grid, ns, tp = solver(n, 1.0, mode)
    xp = grid.xp
    ns.cfl = cfl
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0, v0 = np.sin(TWO_PI * X) * np.cos(TWO_PI * Y), -np.cos(TWO_PI * X) * np.sin(TWO_PI * Y)
    ns.u[0].interior = xp.asarray(u0)
    ns.u[1].interior = xp.asarray(v0)
    ns.start(events=[t_end])
    gs = []
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        gs = (gs + [to_numpy(ns.g[0].interior).copy()])[-3:]
    e = np.hypot(to_numpy(ns.u[0].interior) - u0, to_numpy(ns.u[1].interior) - v0)
    # the flow is steady: the projection's g~ of consecutive steps should agree (a 2 dt oscillation shows here)
    osc = float(np.abs(gs[-1] - gs[-2]).max() / np.abs(gs[-1]).max())
    return {"n": n, "mode": mode, "cfl": cfl, "max": float(e.max()), "g_step_to_step": osc}


# ---------------------------------------------------------------------------- a drop in a vortex, density ratio 1000
def vortex(mode, n=64, ratio=1000.0, t_end=0.25, cfl=0.5):
    """A drop (R = 0.15 at (0.5, 0.3)) in the Taylor-Green vortex plus the uniform flow (0.5, 0.2), density ratio
    ``ratio``, no viscosity, no surface tension, periodic. No exact solution: the run is for the stability and the
    budgets (total momentum, mass-weighted kinetic energy, max |u|)."""
    from basilisk_cupy.backend import to_numpy

    grid, ns, tp = solver(n, ratio, mode, f0=lambda g: _drop(g, c=(0.5, 0.3)))
    xp = grid.xp
    ns.cfl = cfl
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = xp.asarray(0.5 + np.sin(TWO_PI * X) * np.cos(TWO_PI * Y))
    ns.u[1].interior = xp.asarray(0.2 - np.cos(TWO_PI * X) * np.sin(TWO_PI * Y))

    def budgets():
        rho = to_numpy(tp.rho(tp.f.interior))
        u, v = to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)
        return (float((rho * u).sum() / n ** 2), float((rho * v).sum() / n ** 2),
                float(0.5 * (rho * (u * u + v * v)).sum() / n ** 2), float(np.hypot(u, v).max()))

    b0 = budgets()
    ns.start(events=[t_end])
    umax, failed = b0[3], None
    try:
        while ns.clock.t < t_end - 1e-12:
            ns.step()
            umax = max(umax, float(np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)).max()))
            if not math.isfinite(umax) or umax > 1e3:
                failed = "blow-up at t = %.4f" % ns.clock.t
                break
    except Exception as exc:  # noqa: BLE001
        failed = repr(exc)
    b1 = budgets()
    return {"mode": mode, "n": n, "ratio": ratio, "steps": ns.clock.i, "t": ns.clock.t, "failed": failed,
            "umax": umax, "momentum0": b0[:2], "momentum1": b1[:2], "energy0": b0[2], "energy1": b1[2],
            "momentum_change": float(math.hypot(b1[0] - b0[0], b1[1] - b0[1]) / math.hypot(b0[0], b0[1]))}


def _job(a):
    kind, args = a
    return kind, {"gain": one_step_gain, "sweep": one_sweep, "kick": kick, "growth": growth, "single": single_phase, "vortex": vortex}[kind](*args)


def jobs():
    out = [("kick", (r, U)) for r in (10.0, 1e3, 1e5) for U in ((1.0, 0.0), (0.0, 1.0), U0)]
    out += [("sweep", (r,)) for r in (1.0, 10.0, 1e3, 1e5)]
    out += [("gain", (r, n, 0.5, (4, 8, 12, 16, 20, 24), m)) for m in ("face", "accel")
            for r, n in ((30.0, 32), (100.0, 32), (300.0, 32), (1e3, 32), (3e3, 32), (1e4, 32), (1e3, 64), (100.0, 64))]
    out += [("gain", (1e3, 32, c)) for c in (0.25, 0.125)]
    out += [("growth", (r, m)) for m in MODES for r in (10.0, 100.0, 300.0, 1e3, 3e3, 1e4, 1e6)]
    out += [("growth", (r, "face", 32, c)) for r in (1e3, 1e4) for c in (0.25, 0.125)]
    out += [("growth", (r, m, 64)) for m in ("face", "accel", "face+m") for r in (1e3, 1e6)]
    out += [("growth", (r, m, n)) for m in ("face-upwind", "face-massw") for r, n in ((300.0, 32), (1e3, 32), (1e4, 32), (1e3, 64))]
    out += [("growth", (1e3, m, 32, 0.5, 200, U)) for m in ("face", "accel") for U in ((1.0, 0.0), (0.3, 1.0))]
    out += [("single", (n, m)) for m in MODES for n in (32, 64, 128)]
    out += [("vortex", (m, n)) for m in MODES for n in (32, 64)]
    return out


def main() -> None:
    import multiprocessing as mp

    res = {"kick": [], "sweep": [], "gain": [], "growth": [], "single": [], "vortex": []}
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P1_PROCS", "16"))) as pool:
        for kind, o in pool.imap_unordered(_job, jobs()):
            res[kind].append(o)
            print(kind, {k: (float("%.4g" % v) if isinstance(v, float) else v) for k, v in o.items() if k != "steps"},
                  flush=True)
    for k in res:
        res[k].sort(key=lambda o: json.dumps(o, sort_keys=True, default=str))
    json.dump(res, open("docs/figures/r0p1_source_feedback.json", "w"), indent=1, default=float)
    print("wrote docs/figures/r0p1_source_feedback.json")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.exit(main())


def plot() -> None:
    """python -c "from cases.r0p1_source_feedback import plot; plot()"  ->  docs/figures/r0p1_source_feedback.png"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open("docs/figures/r0p1_source_feedback.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    for mode, c in (("face", "C3"), ("accel", "C0")):
        for n, mk in ((32, "o"), (64, "s")):
            pts = [(s["mass_ratio_max"], s["radius"]) for o in R["gain"] if o["mode"] == mode and o["n"] == n
                   and o["cfl"] == 0.5 for s in o["steps"]]
            a.loglog(*zip(*pts), mk, color=c, ms=4, alpha=0.8, label="%s, N = %d" % (mode, n))
    x = np.array([2.0, 3000.0])
    a.loglog(x, 0.107 * x, "k:", lw=0.8, label="0.107 x")
    a.axhline(1.0, color="k", lw=0.6)
    a.set_xlabel("max over the cells of m^n / m^(n+1) in the step")
    a.set_ylabel("spectral radius of the one-step map g^n -> g^(n+1)")
    a.set_title("one step, frozen geometry (CFL 0.5, density ratios 30 ... 1e4)", fontsize=9)
    a.legend(fontsize=7)
    a.grid(True, which="both", alpha=0.3)
    for mode, c in (("face", "C3"), ("force", "C1"), ("face+m", "C4"), ("accel", "C0"), ("trapezoid", "C2"), ("none", "C7")):
        rows = sorted((o["ratio"], o["rate"] if o["unstable"] else 1.0) for o in R["growth"]
                      if o["mode"] == mode and o["n"] == 32 and o["cfl"] == 0.5 and tuple(o.get("U", U0)) == U0)
        b.loglog(*zip(*rows), "-o", color=c, ms=4, label=mode)
    b.set_xlabel("density ratio")
    b.set_ylabel("growth per step of |u - U| (1: no growth)")
    b.set_title("a drop carried by a uniform flow, N = 32, CFL 0.5", fontsize=9)
    b.legend(fontsize=7)
    b.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/figures/r0p1_source_feedback.png", dpi=130)

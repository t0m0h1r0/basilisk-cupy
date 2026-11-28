"""docs/notes/118 (audit #162, R0 #141): a free stream carrying an interface that lies on grid faces grows in the
conserving momentum transport (TwoPhase(conserving=True), no source term; Basilisk's conserving.h does the same) at
density ratios 1e3 and 1e6. Two independent defects of the transport, each a mismatch between the momentum and the
mass a cell keeps, which the velocity u = q / rho(f) then divides by a light cell's mass:

  D1  where a phase is "absent" (vof.h: c <= 1e-10 in the donor cell) the phase's momentum flux is set to zero
      while its volume flux cf u is not (c slightly negative or of the order of round-off); and the momentum is
      built from clamp(f) while f itself (unclamped) is transported and rho(f) clamps again.
  D2  the slope term of the face value, (1 - C)/2 g with vof.h's one-sided gradient g, in a donor cell the phase
      fills only in part: when nearly all of the phase leaves in one sweep (c just above CMIN = 1/2 with the CFL
      number C just below 1/2: an aligned interface at the CFL bound), the slope's share of the momentum stays
      behind with the small mass that remains.

The interventions (hooks, the library is not changed; ``scheme(d1, d2)`` patches two functions for the duration):

  D1 fix  the momentum from the unclamped f, u = q / rho_lin(f) with rho_lin(f) = f rho1 + (1 - f) rho2 (bounded
          below by rho_min / 2), and where the phase is absent the face value is the mixture's rho_k u (the current
          (q1 + q2) / rho_lin) instead of a zero flux: the momentum flux is then rho_k u times the same volume flux.
  D2 fix  no slope term (g = 0) in a donor cell the phase does not fill (0 < c_k < 1).

  matrix()     audit #162's configurations (C's audit/aligned/aligned_base.py: CFL ramped to its bound, 600 steps)
               and a fixed-dt CFL scan, for the four schemes.
  kick_d2()    the one-sweep formula of D2 against the transport (a cell, a linear velocity in the heavy phase).
  ablation     (in matrix()) D1's two parts alone ("d1-absent": the face value where the phase is absent, "d1-lin":
               the unclamped f and density) against both.
  shapes()     C's shapes of break_stability.py (drop, bubble, small drop, film, oblique band, two drops) and the
               aligned band and square, 2000 steps, density ratios 1e3 and 1e6.
  identical()  with no interface the fields of the four schemes are the same (Taylor-Green, 50 steps).
  vortex()     a drop in a Taylor-Green vortex (density ratio 1000): stability, total momentum, and the difference
               of the fields between the schemes against the grid-convergence difference (what D2 costs).

    python -m cases.r0_aligned_free_stream         # docs/figures/r0_aligned_free_stream.json (about 15 min, NP=22 processes)
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import sys
import textwrap

import numpy as np

SCHEMES = {"base": (False, False), "d1": (True, False), "d2": (False, True), "d1+d2": (True, True),
           "d1-absent": ("absent", False), "d1-lin": ("lin", False),
           "d1+d2lim": (True, "limited"), "d1+geo": (True, "geometric")}
MAIN = ("base", "d1", "d2", "d1+d2")

# --- the hooks ---------------------------------------------------------------------------------------------------
_SWEEP_EDITS = [
    ("        else:\n            g = _concentration_gradient(xp, cp, tp, axis, tr.inverse, tr.gradient)\n",
     "        else:\n            g = _concentration_gradient(xp, cp, tp, axis, tr.inverse, tr.gradient)\n"
     "        if getattr(tr, '_d2', False) in (True, 'full'):  # D2 fix: no slope where the phase fills the cell only in part\n"
     "            ck = (1.0 - c.interior) if tr.inverse else c.interior\n"
     "            g = xp.where(ck >= 1.0, g, 0.0)\n"),
    ("        ti, gi = upwind(tp), upwind(gf)\n",
     "        ti, gi = upwind(tp), upwind(gf)\n"
     "        if getattr(tr, '_d2', False) == 'limited':  # D2, limited: no slope where (D2)'s gain would exceed 1\n"
     "            rk, ro = tr._rhos\n"
     "            a_ = xp.abs(un)\n"
     "            out_ = xp.minimum(cf1 * a_, xp.maximum(ci, 0.0))\n"
     "            rem_ = rk * xp.maximum(ci - out_, 0.0) + ro * xp.maximum(1.0 - ci, 0.0)\n"
     "            gain_ = rk * out_ * (1.0 - a_) / 2.0\n"
     "            cut_ = (gain_ > rem_) & (gi != 0.0)\n"
     "            _COUNT[0] += int(cut_.sum())\n"
     "            _COUNT[1] += int(((ci > 1e-10) & (ci < 1.0 - 1e-10) & (a_ > 0.0)).sum())\n"
     "            gi = xp.where(gain_ > rem_, 0.0, gi)\n"),
    ("        tfluxes.append(xp.where(ci > 1e-10, ff * cf1 * u, 0.0))\n",
     "        absent = getattr(tr, '_absent', None)\n"
     "        if absent is not None:  # D1 fix: the mixture's value where the phase is absent (after every term)\n"
     "            ff = xp.where(ci > 1e-10, ff, upwind(_pad(absent(), periodic, xp)))\n"
     "        tfluxes.append(ff * cf1 * u if absent is not None else xp.where(ci > 1e-10, ff * cf1 * u, 0.0))\n"),
]
_SWEEP_EDITS += [  # A-8 (docs/notes/122): capture the sweep's fluxes for the local-budget verifier (off: _CAP[0] None)
    ("    tfluxes = []\n", "    tfluxes = []\n    _ffs = []\n"),
    ("        tfluxes.append(ff * cf1 * u if absent is not None else xp.where(ci > 1e-10, ff * cf1 * u, 0.0))\n",
     "        tfluxes.append(ff * cf1 * u if absent is not None else xp.where(ci > 1e-10, ff * cf1 * u, 0.0))\n"
     "        _ffs.append((ff, cf1, ci))\n"),
    ("    n = grid.n[axis]\n",
     "    if _CAP[0] is not None:\n"
     "        _CAP[0].append({'axis': axis, 'dt': dt, 'h': h, 'u': u.copy(), 'flux': flux.copy(), 'cc': cc,\n"
     "                        'tfluxes': [t.copy() for t in tfluxes], 'ffs': _ffs, 'tcs': tcs,\n"
     "                        'c0': c.interior.copy(), 't0': [tr.field.interior.copy() for tr in interface.tracers],\n"
     "                        'inverse': [tr.inverse for tr in interface.tracers], 'periodic': periodic})\n"
     "    n = grid.n[axis]\n"),
    ("    return cfl\n",
     "    if _CAP[0] is not None:\n"
     "        _CAP[0][-1]['c1'] = c.interior.copy()\n"
     "        _CAP[0][-1]['t1'] = [tr.field.interior.copy() for tr in interface.tracers]\n"
     "    return cfl\n"),
]
CAPTURE = [None]  # a list to append the sweeps to (cases/r0_local_budget.py), None: off
_VOF_EDITS = [
    ("        fc = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)\n        tracers = []\n",
     "        fc = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)\n        if _D[0] in (True, 'lin'):\n"
     "            fc = self.f.interior  # D1 fix: the momentum of the transported (unclamped) f\n"
     "        tracers = []\n"),
    ("            qs.append((q1, q2))\n",
     "            if _D[0] in (True, 'absent'):\n"
     "                for tr, r in zip(tracers[-2:], (self.rho1, self.rho2)):\n"
     "                    tr._absent = (lambda q1=q1, q2=q2, r=r: r * (q1.interior + q2.interior) / _rho_lin(self))\n"
     "            for tr, rr in zip(tracers[-2:], ((self.rho1, self.rho2), (self.rho2, self.rho1))):\n"
     "                tr._d2, tr._rhos = _D[1], rr\n"
     "                if _D[1] == 'geometric':\n"
     "                    tr.geometric = True  # docs/notes/77's centroid form of the slope terms\n"
     "            qs.append((q1, q2))\n"),
    ("        rho = self.rho(self.f.interior)\n        for u, (q1, q2) in zip(ns.u.components, qs):\n",
     "        rho = _rho_lin(self) if _D[0] in (True, 'lin') else self.rho(self.f.interior)\n"
     "        for u, (q1, q2) in zip(ns.u.components, qs):\n"),
]
_GHOST_EDITS = [
    ("            c = xp.minimum(xp.maximum(f, 0.0), 1.0)\n",
     "            c = f if _D[0] in (True, 'lin') else xp.minimum(xp.maximum(f, 0.0), 1.0)\n"),
    ("                                  (q1.interior + q2.interior) / density(f))\n",
     "                                  (q1.interior + q2.interior) / (_rho_lin_f(density, f) if _D[0] in (True, 'lin') else density(f)))\n"),
    ("        fc = xp.minimum(xp.maximum(self._f.data, 0.0), 1.0)\n",
     "        fc = self._f.data if _D[0] in (True, 'lin') else xp.minimum(xp.maximum(self._f.data, 0.0), 1.0)\n"),
]
_D = [False, False]
COUNT = [0, 0]  # "limited": faces whose slope was dropped, faces with a partial donor (both phases, all sweeps)


FLOOR = [0]  # cells (and calls) where rho_lin's floor was reached (docs/notes/119: count it, B on PR #164)


def _rho_lin(tp, f=None):  # the absent-phase value uses it in every D1 variant
    """f rho1 + (1 - f) rho2 of the unclamped f, bounded below by half the smaller density (counted in FLOOR)."""
    xp = tp.f.grid.xp
    f = tp.f.interior if f is None else f
    r = f * (tp.rho1 - tp.rho2) + tp.rho2
    lo = 0.5 * min(tp.rho1, tp.rho2)
    FLOOR[0] += int((r < lo).sum())
    return xp.maximum(r, lo)


def _rho_lin_f(density, f):
    return _rho_lin(density.__self__, f)


def _patched(func, edits, extra):
    raw = inspect.getsource(func)
    cut = len(raw) - len(raw.lstrip(" "))
    src = textwrap.dedent(raw)
    undent = lambda t: "".join(l[cut:] if l.startswith(" " * cut) else l for l in t.splitlines(True))  # noqa: E731
    for old, new in ((undent(o), undent(n)) for o, n in edits):
        if src.count(old) != 1:
            raise RuntimeError("the library changed: cannot patch %s (%r)" % (func.__qualname__, old[:60]))
        src = src.replace(old, new)
    module = sys.modules[func.__module__]
    scope = dict(module.__dict__, **extra)
    exec(compile(src, "<r0_aligned_free_stream:%s>" % func.__qualname__, "exec"), scope)  # noqa: S102
    return scope[func.__name__]


@contextlib.contextmanager
def scheme(d1: bool, d2: bool):
    """Patch vof._sweep, TwoPhase._vof and the momentum's ghost cells with the D1 / D2 fixes (both off: the
    library's code unchanged in behaviour)."""
    from basilisk_cupy.interface import two_phase, vof

    extra = {"_D": _D, "_rho_lin": _rho_lin, "_rho_lin_f": _rho_lin_f, "_COUNT": COUNT, "_CAP": CAPTURE}
    saved = (vof._sweep, two_phase.TwoPhase._vof, two_phase._MomentumField.fill_ghosts)
    sweep = _patched(saved[0], _SWEEP_EDITS, extra)
    tvof = _patched(saved[1], _VOF_EDITS, extra)
    ghosts = _patched(saved[2], _GHOST_EDITS, extra)
    _D[0], _D[1] = d1, d2
    vof._sweep, two_phase.TwoPhase._vof, two_phase._MomentumField.fill_ghosts = sweep, tvof, ghosts
    try:
        yield
    finally:
        vof._sweep, two_phase.TwoPhase._vof, two_phase._MomentumField.fill_ghosts = saved
        _D[0], _D[1] = False, False


@contextlib.contextmanager
def fixed_dt(dt):
    """Every step dt (no CFL ramp): the CFL bound of conserving.h's stability event replaced by dt."""
    import basilisk_cupy.time.stepping as st

    saved = st.timestep
    st.timestep = lambda uf, dtmax, cfl, clock: min(dtmax, dt)
    try:
        yield
    finally:
        st.timestep = saved


def in_scheme(fn):
    """Run ``fn`` (its argument ``sch``) inside ``scheme``: TwoPhase binds its hooks when it is built, so the
    solver must be built inside the patch."""
    import functools

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def run(*args, **kwargs):
        sch = sig.bind(*args, **kwargs).arguments["sch"]
        with scheme(*SCHEMES[sch]):
            return fn(*args, **kwargs)

    return run


# --- the configurations -------------------------------------------------------------------------------------------
def shape(name, grid):
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction

    n = grid.n[0]
    i, j = np.meshgrid(np.arange(n), np.arange(grid.n[1]), indexing="ij")
    disc = lambda r, cx, cy: (lambda x, y: r * r - (x - cx) ** 2 - (y - cy) ** 2)  # noqa: E731
    fr = lambda phi: to_numpy(fraction(grid, phi, refine=8))  # noqa: E731
    if name == "column":                                   # audit #162: x = 1/4 .. 1/2, on faces
        return ((i >= n // 4) & (i < n // 2)).astype(float)
    if name == "column+0.3":                               # the same, 0.3 cells off the faces
        return np.clip(np.minimum(i + 1, n // 2 + 0.3) - np.maximum(i, n // 4 + 0.3), 0, 1)
    if name == "square":
        return (((i >= 3 * n // 8) & (i < 5 * n // 8)) & ((j >= 3 * n // 8) & (j < 5 * n // 8))).astype(float)
    if name == "square-bubble":
        return 1.0 - shape("square", grid)
    if name == "half":                                     # the band with c = 1/2 exactly in its edge cells (#156)
        f = ((i >= n // 4) & (i < n // 2)).astype(float)
        f[n // 4 - 1, :] = 0.5
        f[n // 2, :] = 0.5
        return f
    if name == "band2":                                    # docs/notes/117's oblique band (x + 2 y)
        return fr(lambda x, y: 0.2 - abs(((x + 2.0 * y) % 1.0) - 0.4))
    if name == "drop":
        return fr(disc(0.15, 0.5, 0.5))
    if name == "bubble":
        return 1.0 - fr(disc(0.15, 0.5, 0.5))
    if name == "small":
        return fr(disc(0.05, 0.47, 0.52))
    if name == "film":
        return fr(lambda x, y: 0.02 - np.abs(x - 0.513))
    if name == "slant":
        return fr(lambda x, y: 0.13 - np.abs(((x + y) % 1.0) - 0.5))
    if name == "two":
        return np.clip(fr(disc(0.12, 0.3, 0.4)) + fr(disc(0.1, 0.72, 0.65)), 0, 1)
    raise ValueError(name)


@in_scheme
def free_stream(name, sch, ratio, U=(1.0, 0.0), n=32, cfl=0.5, steps=600, dt=None, tolerance=1e-10, every=None,
                project=True, mu=(0.0, 0.0)):
    """max |u - U| over ``steps`` steps of a uniform flow carrying ``shape(name)``, periodic. ``dt``: a fixed step
    (otherwise the CFL-ramped step of the solver, as audit #162)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance, mu=(max(mu) if max(mu) > 0 else None))
    f0 = shape(name, grid)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio,
                  mu1=mu[0], mu2=mu[1], conserving=True)
    ns.cfl = cfl
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    m, fout, hist, s = 0.0, 0.0, [], 0
    with fixed_dt(dt) if dt else contextlib.nullcontext(), contextlib.ExitStack() as stack:
        if not project:  # no projection at all (a tolerance of 1e30 still does one cycle: NITERMIN = 1)
            import basilisk_cupy.time.basilisk_centered as bc

            stats = type("NoProjection", (), {"i": 0, "resa": 0.0, "nrelax": 4})()
            stack.callback(setattr, bc, "project", bc.project)
            bc.project = lambda *a, **k: stats
        ns.start()
        for s in range(steps):
            ns.step()
            e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
            m = max(m, e)
            f = to_numpy(tp.f.interior)
            fout = max(fout, -float(f.min()), float(f.max()) - 1.0)
            if every and (s + 1) % every == 0:
                hist.append(e)
            if not math.isfinite(e) or e > 10:
                break
    out = {"shape": name, "scheme": sch, "ratio": ratio, "U": list(U), "n": n, "cfl": cfl, "dt": dt, "project": project,
           "steps": s + 1, "max_err": m, "f_out": fout}
    if every:
        out["history"] = hist
    return out


# --- the local formulas -------------------------------------------------------------------------------------------
def _sweep_once(f0, u0, ratio, un, d1=False, d2=False, ax=0):
    """One x sweep of the conserving transport of one row (periodic, y-uniform), the face velocity U = 1 and
    dt = un h. Returns the new f and u (x component)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface import vof
    from basilisk_cupy.interface.two_phase import TwoPhase, _MomentumField

    n = f0.size
    grid = CartesianGrid((n, 2), length=(1.0, 2.0 / n))
    rho1, rho2 = 1.0, 1.0 / ratio
    with scheme(d1, d2):
        f = CellField.from_interior(grid, np.repeat(f0[:, None], 2, 1), BoundaryConditions.periodic(2))
        tp = TwoPhase.__new__(TwoPhase)
        tp.f, tp.rho1, tp.rho2 = f, rho1, rho2
        u = CellField.from_interior(grid, np.repeat(u0[:, None], 2, 1), BoundaryConditions.periodic(2))
        fc = f.interior if d1 else np.clip(f.interior, 0, 1)
        q1 = _MomentumField(grid, bc=u.bc, name="q1").bind(f, u, rho1, False)
        q2 = _MomentumField(grid, bc=u.bc, name="q2").bind(f, u, rho2, True)
        q1.interior, q2.interior = fc * rho1 * u.interior, (1 - fc) * rho2 * u.interior
        sc = CellField(grid, bc=u.bc, name="uq")
        dens = (lambda ff: _rho_lin(tp, ff)) if d1 else tp.rho
        q1.current(q1, q2, tp.rho, sc)
        q2.current(q1, q2, tp.rho, sc)
        t1, t2 = vof.VOFTracer(q1), vof.VOFTracer(q2, inverse=True)
        for tr, r, rr in ((t1, rho1, (rho1, rho2)), (t2, rho2, (rho2, rho1))):
            tr._d2, tr._rhos = d2, rr
            if d2 == "geometric":
                tr.geometric = True
            if d1:
                tr._absent = (lambda r=r: r * (q1.interior + q2.interior) / _rho_lin(tp))
        uf = FaceField(grid, [np.ones((n + 1, 2)), np.zeros((n, 3))], "uf")
        itf = vof.VOFInterface(f, [t1, t2])
        cc = (f.interior > 0.5).astype(float)
        tcs = [np.where(f.interior > 0.5, q1.interior / np.where(f.interior > 0.5, f.interior, 1), 0),
               np.where(f.interior < 0.5, q2.interior / np.where(f.interior < 0.5, 1 - f.interior, 1), 0)]
        vof._sweep(itf, uf, un / n, ax, cc, tcs, (True, True))
        unew = (q1.interior + q2.interior) / dens(f.interior)
    return to_numpy(f.interior)[:, 0], to_numpy(unew)[:, 0]


def kick_d2(ratio=1e3, cfl=(0.45, 0.49, 0.499, 0.4999), c=(0.5, 0.5001, 0.501, 0.51, 0.55), g=1e-6, n=16):
    """D2's formula. The heavy phase fills the right part c of cell T (light upstream, heavy downstream), the
    velocity is U = 1 upstream and in T and grows linearly across the heavy cells, u_{T+k} = 1 + g k, so vof.h's
    gradient of rho1 u in T is the one-sided rho1 g (c >= CMIN, light upstream). One sweep with CFL C sends the
    heavy volume c - delta out of T (delta = max(c - C, 0)) with the face value rho1 (u_T + (1 - C) g / 2); the slope's
    momentum leaves while the mass it would belong to does not, and T keeps the mass rho1 delta + rho2 (1 - delta):

        u_T' - u_T = - rho1 (c - delta) (1 - C) g / 2 / (rho1 delta + rho2 (1 - delta))          (D2)

    (the light fluid entering from upstream carries u_T exactly here). The gain (D2)/g reaches
    ratio (1 - C) c / 2 ~ ratio / 8 as delta -> 0, i.e. c -> C from above, which an interface on the faces moved at
    the CFL bound 1/2 meets every other step. The D2 fix (no slope in T) gives zero."""
    rows = []
    iT = n // 2
    for C in cfl:
        for cT in c:
            f0 = np.zeros(n)
            f0[iT] = cT
            f0[iT + 1:iT + 6] = 1.0
            u0 = np.ones(n)
            u0[iT:iT + 6] += g * np.arange(6)          # linear in the heavy cells, T included (u_T = 1)
            u0[iT + 6:] += g * 5
            u0[:iT] = 1.0                              # the light phase upstream: uniform (u_T)
            out = {}
            for sch in ("base", "d2", "d1+d2", "d1+d2lim", "d1+geo"):
                _, u1 = _sweep_once(f0, u0, ratio, C, *SCHEMES[sch])
                out[sch] = float(u1[iT] - u0[iT])
            delta = max(cT - C, 0.0)
            rho1, rho2 = 1.0, 1.0 / ratio
            pred = -rho1 * (cT - delta) * (1 - C) * g / 2 / (rho1 * delta + rho2 * (1 - delta))
            rows.append({"cfl": C, "c": cT, "delta": delta, "predicted": pred, "measured": out,
                         "gain_base": out["base"] / g, "gain_d2": out["d2"] / g})
    return rows


# --- single phase, the vortex -------------------------------------------------------------------------------------
@in_scheme
def identical(sch, n=32, steps=50):
    """No interface (f = 0): a Taylor-Green flow plus (0.5, 0.2); the fields after ``steps`` steps."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), BoundaryConditions.periodic(2)), rho1=1.0,
             rho2=1e-3, conserving=True)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    ns.start()
    for _ in range(steps):
        ns.step()
    return np.stack([to_numpy(ns.u[k].interior) for k in range(2)])


@in_scheme
def vortex(sch, n=64, ratio=1000.0, t_end=0.25, cfl=0.5, save=None):
    """docs/notes/114's drop (R = 0.15 at (0.5, 0.3)) in the Taylor-Green vortex plus (0.5, 0.2), no viscosity:
    stability, the total momentum (with the clamped and with the linear density), the fields at t_end."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    from basilisk_cupy.interface.fractions import fraction

    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio,
                  conserving=True)
    ns.cfl = cfl
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    def budget(lin):
        f = to_numpy(tp.f.interior)
        rho = f * (tp.rho1 - tp.rho2) + tp.rho2 if lin else np.clip(f, 0, 1) * (tp.rho1 - tp.rho2) + tp.rho2
        return np.array([(rho * to_numpy(ns.u[k].interior)).sum() / n ** 2 for k in range(2)])

    b0 = budget(False)
    failed, umax = None, 0.0
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        umax = max(umax, float(np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)).max()))
        if not math.isfinite(umax) or umax > 1e3:
            failed = "blow-up at t = %.4f" % ns.clock.t
            break
    if save:
        np.save(save, np.stack([to_numpy(ns.u[k].interior) for k in range(2)] + [to_numpy(tp.f.interior)]))
    norm = float(np.hypot(*b0))
    return {"scheme": sch, "n": n, "ratio": ratio, "steps": ns.clock.i, "failed": failed, "umax": umax,
            "momentum_change_clamped": float(np.hypot(*(budget(False) - b0)) / norm),
            "momentum_change_linear": float(np.hypot(*(budget(True) - b0)) / norm)}


def d2_cost(directory, ns_=(64, 128, 256)):
    """The vortex fields of base and d1+d2 at t = 0.25 (density ratio 1000) on N = 64, 128, 256: the difference of
    the schemes on one grid against the difference between grids (coarsened by cell averages to N = 64)."""
    import multiprocessing as mp

    J = [("vortex", dict(sch=s, n=n, save=os.path.join(directory, "vortex_%s_%d.npy" % (s, n))))
         for s in ("base", "d1+d2") for n in ns_]
    with mp.get_context("spawn").Pool(len(J)) as pool:
        pool.map(_job, J)

    def load(s, n, to=64):
        a = np.load(os.path.join(directory, "vortex_%s_%d.npy" % (s, n)))
        r = n // to
        return a.reshape(a.shape[0], to, r, to, r).mean(axis=(2, 4))

    out = {}
    for n in ns_:
        d = load("base", n) - load("d1+d2", n)
        out["schemes_N%d" % n] = {"u_max": float(np.abs(d[:2]).max()), "u_rms": float(np.sqrt((d[:2] ** 2).mean())),
                                  "f_max": float(np.abs(d[2]).max())}
    for a, b in zip(ns_[:-1], ns_[1:]):
        for s in ("base", "d1+d2"):
            d = load(s, a) - load(s, b)
            out["grids_%s_N%d-N%d" % (s, a, b)] = {"u_max": float(np.abs(d[:2]).max()),
                                                   "u_rms": float(np.sqrt((d[:2] ** 2).mean()))}
    return out


# --- the driver ---------------------------------------------------------------------------------------------------
def _job(a):
    kind, args = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if kind == "free":
        return {"kind": kind, **free_stream(**args)}
    if kind == "vortex":
        return {"kind": kind, **vortex(**args)}
    raise ValueError(kind)


def jobs():
    J = []
    for sch in SCHEMES:
        for s in ("square", "column"):
            for U in ((1.0, 0.0), (1.0, 0.37)):
                for r in (1.0, 1e3, 1e6):
                    J.append(("free", dict(name=s, sch=sch, ratio=r, U=U)))
            for c in (0.45, 0.25):
                J.append(("free", dict(name=s, sch=sch, ratio=1e3, cfl=c)))
        for s in ("column+0.3", "drop"):
            J.append(("free", dict(name=s, sch=sch, ratio=1e3)))
        J.append(("free", dict(name="column", sch=sch, ratio=1e3, n=64, steps=1200)))
        J.append(("free", dict(name="column", sch=sch, ratio=1e3, tolerance=1e-13)))
    for sch in MAIN:
        for c in (0.25, 0.45, 0.49, 0.499, 0.4999, 0.49999, 0.5):          # fixed dt: no ramp
            J.append(("free", dict(name="column", sch=sch, ratio=1e3, dt=c / 32, every=10)))
    for sch in ("base", "d1+d2"):                                          # C's shapes, 2000 steps
        for s in ("drop", "bubble", "small", "film", "slant", "two", "column", "square", "square-bubble"):
            for r in (1e3, 1e6):
                J.append(("free", dict(name=s, sch=sch, ratio=r, U=(1.0, 0.37), steps=2000)))
                if s in ("column", "square", "square-bubble"):
                    J.append(("free", dict(name=s, sch=sch, ratio=r, U=(1.0, 0.0), steps=2000)))
    for sch in MAIN:
        for n in (64, 128):
            J.append(("vortex", dict(sch=sch, n=n)))
    return J


def _key(d):
    return json.dumps({k: d[k] for k in sorted(d)}, sort_keys=True)


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_aligned_free_stream.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    J = [j for j in jobs() if _key({"kind": j[0], **j[1]}) not in done]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "20")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job, J, chunksize=1)):
            r["job"] = _key({"kind": a[0], **a[1]})
            with open(part, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            done[r["job"]] = r
    res = {"free": [d for d in done.values() if d["kind"] == "free"],
           "vortex": [d for d in done.values() if d["kind"] == "vortex"],
           "kick_d2": kick_d2()}
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res["d2_cost"] = d2_cost(tmp)
    res["no_projection"] = [free_stream("column", s, 1e3, dt=0.4999 / 32, project=False) for s in ("base", "d1+d2")]
    ident = {s: identical(s) for s in SCHEMES}
    res["identical"] = {s: float(np.abs(ident[s] - ident["base"]).max()) for s in SCHEMES}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()


def plot() -> None:
    """docs/figures/r0_aligned_free_stream.png: (a) D2's one-sweep gain against the heavy part delta left in T,
    (b) max |u - U| every 10 steps at fixed dt, CFL 0.4999 and 0.45 (density ratio 1000, aligned band)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open(os.path.join("docs", "figures", "r0_aligned_free_stream.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = kick_d2(cfl=(0.45, 0.49, 0.499, 0.4999, 0.49999), c=(0.5, 0.5001, 0.501, 0.51, 0.55, 0.6))
    d = np.array([r["delta"] for r in rows])
    o = np.argsort(d)
    ax[0].loglog(d[o], [abs(rows[k]["predicted"]) / 1e-6 for k in o], "-", color="0.4", label="formula (D2)")
    ax[0].loglog(d, [abs(r["gain_base"]) for r in rows], "o", ms=5, label="transport, base")
    ax[0].axhline(1000 / 8, ls=":", color="0.6")
    ax[0].set_xlabel("heavy fraction left in T, delta = c - C")
    ax[0].set_ylabel("|u_T' - u_T| / g  (one sweep)")
    ax[0].set_title("(a) D2: the slope's momentum over the mass left (ratio 1000)")
    ax[0].legend()
    for r in res["free"]:
        if r.get("history") and r["dt"] and abs(r["dt"] * 32 - 0.4999) < 1e-9:
            ax[1].semilogy(10 * (1 + np.arange(len(r["history"]))), np.maximum(r["history"], 1e-17), label=r["scheme"])
    for r in res["free"]:
        if r.get("history") and r["dt"] and abs(r["dt"] * 32 - 0.45) < 1e-9 and r["scheme"] in ("base", "d1+d2"):
            ax[1].semilogy(10 * (1 + np.arange(len(r["history"]))), np.maximum(r["history"], 1e-17), "--",
                           label=r["scheme"] + ", CFL 0.45")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("max |u - U|")
    ax[1].set_title("(b) aligned band, ratio 1000, fixed CFL 0.4999 (solid)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0_aligned_free_stream.png"), dpi=130)

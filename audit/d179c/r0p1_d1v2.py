"""docs/notes/113 (R0-P1 continued, issue #141): the time order of the conserving momentum transport for UNSTEADY and
VISCOUS single-phase flows. docs/notes/112 found the two causes for the steady inviscid Taylor-Green flow; three more
candidates only show in unsteady / viscous flows:

  E1  the face velocity that carries the momentum is the projected one of time t^n (conserving.h), not t^{n+1/2};
  E2  the half-step source in the face value has the pressure acceleration only (no viscous acceleration);
  E3  the viscous step is backward Euler (viscosity.h).

Exact solutions (periodic unit square): psi an eigenfunction of the Laplacian (lap psi = -k^2 psi), w = (psi_y, -psi_x),
    u(x, t) = U + exp(-nu k^2 t) w(x - U t)
is an exact Navier-Stokes solution for any constant U (the vorticity is k^2 psi, so w . grad(vorticity) = 0).
  "tg"    psi = sin 2 pi x sin 2 pi y / (2 pi)                                   (k^2 = 8 pi^2, the flow of note 112)
  "asym"  psi = c [sin 2 pi (x + 2y) + 0.6 cos 2 pi (2x - y) + 0.4 sin(2 pi (2x + y) + 1)]   (k^2 = 20 pi^2, no symmetry)
Without a boost (U = 0) the leading terms of E1 and E2 are gradients (removed by the projection): only E3 shows.

The library is not changed: every modification is made from hooks (and ``periodic_momentum_ghosts`` of note 112).

    python -m cases.r0p1_unsteady_viscous            # all tables (docs/figures/r0p1_unsteady_viscous.json)
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

TWO_PI = 2 * math.pi
BOOST = (1.0, 0.37)
T_ENDS = (0.125, 0.25, 0.5)
GRIDS = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256)

_GX, _GW = np.polynomial.legendre.leggauss(4)
_GX, _GW = 0.5 * (_GX + 1.0), 0.5 * _GW


def _psi_raw(flow, x, y):
    if flow == "tg":
        return np.sin(TWO_PI * x) * np.sin(TWO_PI * y) / TWO_PI
    return (np.sin(TWO_PI * (x + 2 * y)) + 0.6 * np.cos(TWO_PI * (2 * x - y))
            + 0.4 * np.sin(TWO_PI * (2 * x + y) + 1.0))


def _w_raw(flow, x, y):
    if flow == "tg":
        return np.sin(TWO_PI * x) * np.cos(TWO_PI * y), -np.cos(TWO_PI * x) * np.sin(TWO_PI * y)
    a, b, c = TWO_PI * (x + 2 * y), TWO_PI * (2 * x - y), TWO_PI * (2 * x + y) + 1.0
    py = TWO_PI * (2 * np.cos(a) + 0.6 * np.sin(b) + 0.4 * np.cos(c))
    px = TWO_PI * (np.cos(a) - 1.2 * np.sin(b) + 0.8 * np.cos(c))
    return py, -px


def _scale(flow):
    """"asym" is normalised to max |w| = 1 (as "tg")."""
    if flow == "tg":
        return 1.0
    if "asym" not in _scale.cache:
        s = (np.arange(1024) + 0.5) / 1024
        X, Y = np.meshgrid(s, s, indexing="ij")
        u, v = _w_raw("asym", X, Y)
        _scale.cache["asym"] = 1.0 / float(np.hypot(u, v).max())
    return _scale.cache["asym"]


_scale.cache = {}


def k2(flow):
    return (8.0 if flow == "tg" else 20.0) * math.pi ** 2


def mix(nu, U):
    """The manufactured flow "mix" (B's request on PR #152): two waves of DIFFERENT Laplacian eigenvalues with
    different time factors, carried by U,
        psi = 0.6 exp(-t) sin 2 pi X sin 2 pi Y / (2 pi) + 0.3 cos(3 t) cos(2 pi (2 X + Y) + 0.5) / (2 pi),
        X = x - U_x t, Y = y - U_y t,   u = U + (psi_y, -psi_x),
    with the body force a = u_t + (u . grad) u - nu lap u (the pressure of the exact solution is zero; a is not a
    gradient-free field, the solver's pressure takes its gradient part). Not an eigenmode: (w . grad) w is not a
    gradient and the viscous term is not proportional to the velocity. Lambdified psi, u, v, a_x, a_y of (x, y, t)."""
    key = (float(nu), tuple(U))
    if key not in mix.cache:
        import sympy as sp

        x, y, t = sp.symbols("x y t")
        X, Y = x - U[0] * t, y - U[1] * t
        psi = (sp.Rational(3, 5) * sp.exp(-t) * sp.sin(2 * sp.pi * X) * sp.sin(2 * sp.pi * Y)
               + sp.Rational(3, 10) * sp.cos(3 * t) * sp.cos(2 * sp.pi * (2 * X + Y) + sp.Rational(1, 2))) / (2 * sp.pi)
        u, v = U[0] + sp.diff(psi, y), U[1] - sp.diff(psi, x)
        lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)  # noqa: E731
        a = [sp.diff(c, t) + u * sp.diff(c, x) + v * sp.diff(c, y) - nu * lap(c) for c in (u, v)]
        mix.cache[key] = [sp.lambdify((x, y, t), e, "numpy") for e in (psi, u, v, a[0], a[1])]
    return mix.cache[key]


mix.cache = {}


def w_cells(flow, n, shift):
    """Cell averages of the normalised w(x - shift)."""
    h = 1.0 / n
    i = np.arange(n)
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for a, wa in zip(_GX, _GW):
        for b, wb in zip(_GX, _GW):
            X, Y = np.meshgrid((i + a) * h - shift[0], (i + b) * h - shift[1], indexing="ij")
            wu, wv = _w_raw(flow, X, Y)
            u += wa * wb * wu
            v += wa * wb * wv
    return _scale(flow) * u, _scale(flow) * v


def exact_cells(flow, n, t, nu, U):
    """Cell averages of the exact velocity at time t."""
    if flow == "mix":
        _, fu, fv, _, _ = mix(nu, U)
        i = np.arange(n)
        u, v = np.zeros((n, n)), np.zeros((n, n))
        for a, wa in zip(_GX, _GW):
            for b, wb in zip(_GX, _GW):
                X, Y = np.meshgrid((i + a) / n, (i + b) / n, indexing="ij")
                u += wa * wb * fu(X, Y, t)
                v += wa * wb * fv(X, Y, t)
        return u, v
    amp = math.exp(-nu * k2(flow) * t)
    u, v = w_cells(flow, n, (U[0] * t, U[1] * t))
    return U[0] + amp * u, U[1] + amp * v


def error_shapes(flow, n, t, nu, U, delta=1e-4):
    """The two fields the first-order time errors are predicted along (docs/notes/113, section B):
    s1 = w~ (the amplitude) and s2 = (U . grad) w~ (a shift of the pattern), w~ = exp(-nu k^2 t) w(x - U t)."""
    if flow == "mix":
        return None, None
    amp = math.exp(-nu * k2(flow) * t)
    s1 = [amp * a for a in w_cells(flow, n, (U[0] * t, U[1] * t))]
    if U[0] == 0.0 and U[1] == 0.0:
        return s1, None
    p = w_cells(flow, n, (U[0] * (t + delta), U[1] * (t + delta)))
    m = w_cells(flow, n, (U[0] * (t - delta), U[1] * (t - delta)))
    return s1, [-amp * (a - b) / (2 * delta) for a, b in zip(p, m)]


def exact_faces(flow, n, t, nu, U):
    """Exact face averages of the velocity at time t (differences of the stream function at the vertices:
    discretely divergence-free)."""
    h = 1.0 / n
    xv = np.arange(n + 1) * h
    if flow == "mix":
        X, Y = np.meshgrid(xv, xv, indexing="ij")
        P = mix(nu, U)[0](X, Y, t)
        return U[0] + (P[:, 1:] - P[:, :-1]) / h, U[1] - (P[1:, :] - P[:-1, :]) / h
    X, Y = np.meshgrid(xv - U[0] * t, xv - U[1] * t, indexing="ij")
    P = _scale(flow) * math.exp(-nu * k2(flow) * t) * _psi_raw(flow, X, Y)
    return U[0] + (P[:, 1:] - P[:, :-1]) / h, U[1] - (P[1:, :] - P[:-1, :]) / h


from basilisk_cupy.fields import VectorField
from basilisk_cupy.boundary import vector_bcs


def run(n, cfl, flow="tg", nu=0.01, boost=True, variant="conserving", level="n", source="g", visc="be",
        ghost_fix=True, tolerance=1e-10, t_ends=T_ENDS, save=None):
    """One run of the library's solver with modifications made from hooks.

    variant  "plain" (centered.h's BCG advection) or "conserving" (TwoPhase(conserving=True), f = 0).
    level    the face velocity that carries the momentum (conserving only):
             "n" the library's (projected, time t^n); "pred" the library's ``conserving_predict`` (BCG prediction +
             projection); "ab" the extrapolation uf^n + dt / (2 dt_prev) (uf^n - uf^{n-1}) (divergence-free without a
             projection); "exact" the exact face averages at t^n + dt / 2 (oracle).
    source   the half-step source in the face value: "none"; "g" the pressure acceleration g^n (the library's
             ``conserving_source`` / BCG's prediction); "g+visc" g^n + (1/rho) L_h u^n; "lag" the lagged total
             acceleration (u^n - u*^{n-1}) / dt_{n-1} (u*: the velocity just after the previous transport).
    visc     "be" backward Euler (the library); "cn" Crank-Nicolson with the explicit half on u^n; "cn-naive"
             Crank-Nicolson applied to the solver's intermediate velocity u* + dt g^n.
    ``flow`` "mix": the manufactured flow of ``mix`` with its body force (face centres, time t^n + dt / 2).
    ``save``: a path for the velocity at the last of ``t_ends`` (.npy), for differences of fields.
    Returns, at ``t_ends``: the max and RMS of |u - u_exact|, the sum of dt^2, and the error's projections c1, c2 on
    ``error_shapes``."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.r0p1_local_error import periodic_momentum_ghosts

    U = BOOST if boost else (0.0, 0.0)
    grid = CartesianGrid((n, n))
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(0, 1), mu=(nu if nu > 0 else None), tolerance=tolerance)
    if variant == "conserving":
        from basilisk_cupy.interface.two_phase import TwoPhase

        bc = BoundaryConditions()
        for ax in range(2):
            for side in (0, 1):
                bc.set((ax, side), Periodic())
        import inspect

        # the library after issue #146 / audit #149 has fresh ghosts by default and this flag for conserving.h's;
        # before it, the ghosts are conserving.h's and ``periodic_momentum_ghosts`` (below) patches them
        kw = ({"conserving_stale_ghosts": not ghost_fix}
              if "conserving_stale_ghosts" in inspect.signature(TwoPhase.__init__).parameters else {})
        TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), bc), rho1=1.0, rho2=1.0, mu1=nu, mu2=nu,
                 conserving=True, conserving_source=(source != "none"), conserving_predict=(level == "pred"), **kw)
    elif level != "n":
        raise ValueError("level applies to the conserving variant only")
    ns.cfl = cfl
    st = {"uf_prev": None, "dt_prev": None, "ustar_prev": None, "g_saved": None, "Lun": None, "sum_dt2": 0.0}

    def viscous_acceleration(solver, dt):
        """(1 / rho) L_h u for the current u (the stencil of the library's viscosity())."""
        solver.u.fill_ghosts()
        prob = ViscosityProblem.reuse(solver.u, solver.mu, solver.rho, dt)
        res = [xp.zeros(grid.n) for _ in range(2)]
        prob.residual(list(solver.u), [xp.zeros(grid.n) for _ in range(2)], res)   # res = -u + dt / rho L u
        return [(r + c.interior) / dt for r, c in zip(res, solver.u.components)]

    def before(solver):                                          # "vof": u = u^n, nothing done yet
        dt = solver.clock.dt
        if nu > 0 and (source == "g+visc" or visc == "cn"):
            st["Lun"] = viscous_acceleration(solver, dt)
        if variant == "conserving":
            if level == "ab":
                cur = [c.copy() for c in solver.uf.components]
                if st["uf_prev"] is not None:
                    r = 0.5 * dt / st["dt_prev"]
                    for c, a, b in zip(solver.uf.components, cur, st["uf_prev"]):
                        c[...] = a + r * (a - b)
                st["uf_prev"] = cur
            elif level == "exact":
                fx, fy = exact_faces(flow, n, solver.clock.t + 0.5 * dt, nu, U)
                solver.uf.components[0][...] = xp.asarray(fx)
                solver.uf.components[1][...] = xp.asarray(fy)
        st["g_saved"] = [c.interior.copy() for c in solver.g.components]
        if level.startswith("d1v2"):
            g_old = [c.interior.copy() for c in solver.g.components]   # 第2版の d1：予測は B(u^n; s_P=L u^n〔V-CN〕) + 外力の半ステップ、半ステップの射影、g_P を作る
            from basilisk_cupy.projection.approximate import project
            from basilisk_cupy.terms.convection.bcg import prediction
            sp_ = VectorField.zeros(grid, vector_bcs(2, (0, 1)), name="sP")
            if nu > 0 and visc == "cn" and "nopredL" not in level:
                for c, a in zip(sp_.components, st["Lun"]):
                    c.interior = a
            sp_.fill_ghosts()
            uf = prediction(solver.u, sp_, dt)
            solver._apply_uf_bc(uf)
            h_ = 1.0 / n
            ax_ = [xp.zeros((n + 1, n)), xp.zeros((n, n + 1))]
            if flow == "mix":
                _, _, _, fx_, fy_ = mix(nu, U)
                xa, ya = np.meshgrid(np.arange(n + 1) * h_, (np.arange(n) + 0.5) * h_, indexing="ij")
                xb, yb = np.meshgrid((np.arange(n) + 0.5) * h_, np.arange(n + 1) * h_, indexing="ij")
                ax_ = [xp.asarray(fx_(xa, ya, solver.clock.t)), xp.asarray(fy_(xb, yb, solver.clock.t))]
                for k in range(2):
                    uf.components[k][...] += 0.5 * dt * ax_[k]
            project(uf, solver.pf, solver.alpha, dt / 2.0, ops=solver.ops, nrelax=4, tolerance=solver.tolerance,
                    basilisk_tolerance=True)
            solver._apply_uf_bc(uf)
            for c, w in zip(solver.uf.components, uf.components):
                c[...] = w
            solver.pf.fill_ghosts()
            for k in range(2):                       # g_P = C(a^n - alpha grad pf)
                gf = ax_[k] - solver.alpha[k] * solver.ops.grad_f(solver.pf, k)
                solver.g[k].interior = 0.5 * (gf.take(range(0, n), k) + gf.take(range(1, n + 1), k))
            solver.g.fill_ghosts()
            st["g_saved"] = [c.interior.copy() for c in solver.g.components]
            if "addold" in level:            # 粘性の段の出し入れは保存した g^n
                st["g_saved"] = g_old
            if "srcold" in level:            # 輸送の源の項は保存した g^n
                for k in range(2):
                    solver.g[k].interior = g_old[k]
                solver.g.fill_ghosts()
        if source == "g+visc" and nu > 0:
            for c, a in zip(solver.g.components, st["Lun"]):
                c.interior = c.interior + a
        elif source == "lag" and st["ustar_prev"] is not None:
            for c, uc, us in zip(solver.g.components, solver.u.components, st["ustar_prev"]):
                c.interior = (uc.interior - us) / st["dt_prev"]
        if source in ("g+visc", "lag"):
            solver.g.fill_ghosts()
        st["dt_now"] = dt
        st["sum_dt2"] += dt * dt

    def after_transport(solver):                                 # "viscous_term": u = the transported velocity
        st["ustar_prev"] = [c.interior.copy() for c in solver.u.components]
        st["dt_prev"] = st["dt_now"]
        for c, s in zip(solver.g.components, st["g_saved"]):
            c.interior = s
        solver.g.fill_ghosts()

    class CrankNicolson:
        def solve(self, u, mu, rho, dt, tolerance=1e-3):
            acc = st["Lun"] if visc == "cn" else viscous_acceleration(ns, dt)
            for c, a in zip(u.components, acc):
                c.interior = c.interior + 0.5 * dt * a
            u.fill_ghosts()
            return viscosity(u, mu, rho, 0.5 * dt, tolerance=tolerance)

    if nu > 0 and visc != "be":
        ns.viscous = CrankNicolson()
    ns.on("vof", before)                                         # registered later: runs before TwoPhase's vof hook
    ns.on("viscous_term", after_transport)
    if flow == "mix":
        _, _, _, fax, fay = mix(nu, U)
        hh = 1.0 / n
        Xx, Yx = np.meshgrid(np.arange(n + 1) * hh, (np.arange(n) + 0.5) * hh, indexing="ij")
        Xy, Yy = np.meshgrid((np.arange(n) + 0.5) * hh, np.arange(n + 1) * hh, indexing="ij")
        ns.a_is_constant = False

        def force(solver):
            tm = solver.clock.t + 0.5 * solver.clock.dt
            solver.a[0] = solver.a[0] + xp.asarray(fax(Xx, Yx, tm))
            solver.a[1] = solver.a[1] + xp.asarray(fay(Xy, Yy, tm))

        ns.on("acceleration", force)
    u0, v0 = exact_cells(flow, n, 0.0, nu, U)
    ns.u[0].interior = xp.asarray(u0)
    ns.u[1].interior = xp.asarray(v0)
    out = {"n": n, "cfl": cfl, "flow": flow, "nu": nu, "boost": boost, "variant": variant, "level": level,
           "source": source, "visc": visc, "ghost_fix": ghost_fix, "errors": []}

    def advance():
        ns.start(events=list(t_ends))
        for te in t_ends:
            while ns.clock.t < te - 1e-12:
                ns.step()
                if ns.clock.i % 10 == 0 and float(np.abs(to_numpy(ns.u[0].interior)).max()) > 30:
                    raise FloatingPointError("blow-up at t=%.3f step %d" % (ns.clock.t, ns.clock.i))
            ue, ve = exact_cells(flow, n, te, nu, U)
            ex, ey = to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve
            e = np.hypot(ex, ey)
            row = {"t": te, "max": float(e.max()), "rms": float(np.sqrt((e * e).mean())), "steps": ns.clock.i,
                   "sum_dt2": st["sum_dt2"]}
            for key, sh in zip(("c1", "c2"), error_shapes(flow, n, te, nu, U)):   # projections on s1 and s2
                if sh is not None:
                    row[key] = float(((ex * sh[0]).sum() + (ey * sh[1]).sum())
                                     / ((sh[0] ** 2).sum() + (sh[1] ** 2).sum()))
            out["errors"].append(row)
        if save:
            np.save(save, np.stack([to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)]))

    try:
        if ghost_fix and variant == "conserving":
            with periodic_momentum_ghosts():
                advance()
        else:
            advance()
    except Exception as exc:  # noqa: BLE001  (a failed run is a result: recorded, not hidden)
        out["failed"] = repr(exc)
    return out


# ------------------------------------------------------------------------------------------------------- the tables
#: name -> options. Fixed before the measurements (docs/notes/113, section B).
CONFIGS = {
    "plain":                 dict(variant="plain", source="g", visc="be"),
    "plain+visc":            dict(variant="plain", source="g+visc", visc="be"),
    "plain+cn":              dict(variant="plain", source="g", visc="cn"),
    "plain+visc+cn":         dict(variant="plain", source="g+visc", visc="cn"),
    "plain+lag+cn":          dict(variant="plain", source="lag", visc="cn"),
    "cons":                  dict(variant="conserving", level="n", source="g", visc="be"),
    "cons/ab":               dict(variant="conserving", level="ab", source="g", visc="be"),
    "cons/pred":             dict(variant="conserving", level="pred", source="g", visc="be"),
    "cons/exact":            dict(variant="conserving", level="exact", source="g", visc="be"),
    "cons+visc":             dict(variant="conserving", level="n", source="g+visc", visc="be"),
    "cons+cn":               dict(variant="conserving", level="n", source="g", visc="cn"),
    "cons/ab+visc":          dict(variant="conserving", level="ab", source="g+visc", visc="be"),
    "cons/ab+cn":            dict(variant="conserving", level="ab", source="g", visc="cn"),
    "cons+visc+cn":          dict(variant="conserving", level="n", source="g+visc", visc="cn"),
    "cons/ab+visc+cn":       dict(variant="conserving", level="ab", source="g+visc", visc="cn"),
    "cons/ab+lag+cn":        dict(variant="conserving", level="ab", source="lag", visc="cn"),
    "cons/pred+visc+cn":     dict(variant="conserving", level="pred", source="g+visc", visc="cn"),
    "cons/exact+visc+cn":    dict(variant="conserving", level="exact", source="g+visc", visc="cn"),
    "cons/ab+visc+cn-naive": dict(variant="conserving", level="ab", source="g+visc", visc="cn-naive"),
    "cons/ab+visc+cn,stale": dict(variant="conserving", level="ab", source="g+visc", visc="cn", ghost_fix=False),
}
INVISCID = ("plain", "cons", "cons/ab", "cons/pred", "cons/exact", "cons/ab+lag+cn")
NU = {"tg": 0.01, "asym": 0.004, "mix": 0.01}                    # nu k^2 = 0.79 for tg and asym


def _job(a):
    name, n, cfl, flow, nu, boost = a
    o = run(n, cfl, flow=flow, nu=nu, boost=boost, **CONFIGS[name])
    o["config"] = name
    return o


def jobs(quick=False):
    out = []
    # (1) the ladder of fixes: fixed CFL 0.5, N = 64 / 128 / 256, every configuration, four problems
    for flow in ("tg", "asym"):
        for boost in (False, True):
            for nu in (0.0, NU[flow]):
                if nu == 0.0 and not boost:
                    continue                                     # steady inviscid flow: note 112
                for name in (INVISCID if nu == 0.0 else CONFIGS):
                    for n in ((32, 64) if quick else (64, 128, 256)):
                        out.append((name, n, 0.5, flow, nu, boost))
    if quick:
        return out
    # (2) the convention: 13 grids, fixed CFL 0.5, the base and the fixed configurations, the hardest problem
    for name in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn", "cons/pred+visc+cn"):
        for flow in ("tg", "asym"):
            for n in GRIDS:
                out.append((name, n, 0.5, flow, NU[flow], True))
    # (3) fixed grids, the time step alone
    for name in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn"):
        for n in (64, 128):
            for cfl in (0.5, 0.25, 0.125, 0.0625):
                out.append((name, n, cfl, "asym", NU["asym"], True))
    # (4) B's request on PR #152: a manufactured flow that is not an eigenmode (13 grids, fixed CFL)
    for name in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn", "cons/exact+visc+cn"):
        for n in GRIDS:
            out.append((name, n, 0.5, "mix", NU["mix"], True))
    return sorted(set(out), key=lambda a: -a[1] ** 3 / a[2])     # the longest first


# ------------------------------------------ the first-order-in-time coefficient from differences of FIELDS (B, PR #152)
TC_CONFIGS = ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn")
TC_GRIDS = (64, 128, 256)
TC_CFLS = (0.5, 0.25, 0.125, 0.0625)


def _field_job(a):
    name, n, cfl, flow, path = a
    if not os.path.exists(path):
        o = run(n, cfl, flow=flow, nu=NU[flow], boost=True, t_ends=(0.5,), save=path + ".tmp.npy", **CONFIGS[name])
        json.dump({"steps": o["errors"][-1]["steps"], "sum_dt2": o["errors"][-1]["sum_dt2"],
                   "max": o["errors"][-1]["max"]}, open(path + ".json", "w"))
        os.replace(path + ".tmp.npy", path)
    return a


def time_coefficient(directory):
    """Same grid, same physical time t = 0.5, time steps halved: D = u(dt) - u(dt / 2) as FIELDS (max and RMS of the
    vector's magnitude). With an error A dt + B dt^2 + (terms without dt), D = A dt / 2 + 3 B dt^2 / 4: the
    coefficient estimate is a1 = |D| / (dt - dt / 2), and the order of the dt-dependent part is
    log2(|D(dt)| / |D(dt / 2)|). docs/figures/r0p1_time_coefficient.json."""
    import multiprocessing as mp

    os.makedirs(directory, exist_ok=True)
    todo = [(name, n, cfl, flow, os.path.join(directory, "%s_%s_%d_%g.npy" % (name.replace("/", "-"), flow, n, cfl)))
            for flow in ("asym", "mix") for name in TC_CONFIGS for n in TC_GRIDS for cfl in TC_CFLS]
    todo.sort(key=lambda a: -a[1] ** 3 / a[2])
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P1_PROCS", "20"))) as pool:
        for a in pool.imap_unordered(_field_job, todo):
            print("field", a[:4], flush=True)
    rows = []
    for flow in ("asym", "mix"):
        for name in TC_CONFIGS:
            for n in TC_GRIDS:
                f, dt = [], []
                for cfl in TC_CFLS:
                    path = os.path.join(directory, "%s_%s_%d_%g.npy" % (name.replace("/", "-"), flow, n, cfl))
                    f.append(np.load(path))
                    dt.append(0.5 / json.load(open(path + ".json"))["steps"])
                pairs = []
                for k in range(len(TC_CFLS) - 1):
                    d = np.hypot(f[k][0] - f[k + 1][0], f[k][1] - f[k + 1][1])
                    pairs.append({"cfl": TC_CFLS[k], "dt": dt[k], "dt_half": dt[k + 1], "max": float(d.max()),
                                  "rms": float(np.sqrt((d * d).mean())),
                                  "a1_max": float(d.max() / (dt[k] - dt[k + 1])),
                                  "a1_rms": float(np.sqrt((d * d).mean()) / (dt[k] - dt[k + 1]))})
                orders = [math.log2(pairs[k]["rms"] / pairs[k + 1]["rms"]) for k in range(len(pairs) - 1)]
                rows.append({"flow": flow, "config": name, "n": n, "pairs": pairs, "orders_rms": orders})
    json.dump({"t": 0.5, "rows": rows}, open("docs/figures/r0p1_time_coefficient.json", "w"), indent=1)
    return rows


def slope(ns, es):
    return float(-np.polyfit(np.log(np.asarray(ns, float)), np.log(np.asarray(es, float)), 1)[0])


def main() -> None:
    """Results are appended to a .jsonl file as they arrive, and a rerun skips the runs already there."""
    import multiprocessing as mp

    if "--time-coefficient" in sys.argv:
        for r in time_coefficient(os.environ.get("R0P1_FIELDS", "r0p1_fields")):
            print(r["flow"], r["config"], r["n"], ["%.3e" % p["a1_rms"] for p in r["pairs"]],
                  ["%.2f" % o for o in r["orders_rms"]], flush=True)
        return
    quick = "--quick" in sys.argv
    path = "docs/figures/r0p1_unsteady_viscous%s.json" % ("_quick" if quick else "")
    part = os.environ.get("R0P1_PARTIAL", path + "l")
    res = []
    if os.path.exists(part):
        with open(part) as fh:
            res = [json.loads(line) for line in fh if line.strip()]
    key = lambda o: (o["config"], o["n"], o["cfl"], o["flow"], o["nu"], o["boost"])  # noqa: E731
    done = {key(o) for o in res}
    todo = [a for a in jobs(quick) if a not in done]
    print(len(todo), "runs to do,", len(done), "done", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P1_PROCS", "26"))) as pool, open(part, "a") as fh:
        for o in pool.imap_unordered(_job, todo):
            res.append(o)
            fh.write(json.dumps(o, default=float) + "\n")
            fh.flush()
            e = o["errors"][-1] if o["errors"] else {"max": float("nan"), "rms": float("nan"), "steps": 0}
            print("%-22s %-4s boost=%-5s nu=%-6g n=%3d cfl=%-6g  max %.3e rms %.3e (%d steps)%s"
                  % (o["config"], o["flow"], o["boost"], o["nu"], o["n"], o["cfl"], e["max"], e["rms"], e["steps"],
                     "  FAILED " + o["failed"] if "failed" in o else ""), flush=True)
    res.sort(key=lambda o: (o["flow"], o["boost"], o["nu"], o["config"], o["cfl"], o["n"]))
    json.dump({"boost": BOOST, "t_ends": T_ENDS, "nu": NU, "runs": res}, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()

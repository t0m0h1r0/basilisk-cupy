"""docs/notes/112 (R0-P1, issue #141): why the conserving momentum transport is first order in time (docs/notes/77).

Single phase (c = 1 everywhere), the library's VOF-tracer transport (basilisk_cupy.interface.vof, unchanged), a known
steady divergence-free velocity (Taylor-Green from the stream function at the vertices: the face values are exact
face averages and discretely divergence-free).

  P0-3  a scalar carried by the flow: one step from exact data with the sweep order x-y and y-x; the error fields are
        split into the even part (e_xy + e_yx) / 2 and the odd part (e_xy - e_yx) / 2 and compared with the user's
        leading term  e_xy = (dt^2 / 2) ([B, A] + d (A + B)) q,  A = -u d/dx, B = -v d/dy, d = du/dx  (odd).
  P0-4  the x-momentum of the steady Taylor-Green flow with the exact pressure acceleration g = -grad p added after
        the transport: without the half-step source in the face value the predicted error is (dt^2 / 2) u . grad g
        (even); with it (VOFTracer.source) that term cancels.
  Global: fixed grid, alternating sweeps, the error at a fixed time against the time step.

    python -m cases.r0p1_local_error
"""

from __future__ import annotations

import json
import math

import numpy as np

TWO_PI = 2 * math.pi


def face_velocity(n):
    h = 1.0 / n
    xv = np.arange(n + 1) * h
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    P = np.sin(TWO_PI * X) * np.sin(TWO_PI * Y) / TWO_PI
    return (P[:, 1:] - P[:, :-1]) / h, -(P[1:, :] - P[:-1, :]) / h


def uv(x, y):
    return np.sin(TWO_PI * x) * np.cos(TWO_PI * y), -np.cos(TWO_PI * x) * np.sin(TWO_PI * y)


def q0(x, y):
    return 2.0 + np.sin(TWO_PI * x) * np.cos(TWO_PI * y) + 0.5 * np.cos(TWO_PI * (x + 2 * y))


_GX, _GW = np.polynomial.legendre.leggauss(6)
_GX, _GW = 0.5 * (_GX + 1.0), 0.5 * _GW


def cell_average(fun, n):
    h = 1.0 / n
    i = np.arange(n)
    out = np.zeros((n, n))
    for a, wa in zip(_GX, _GW):
        for b, wb in zip(_GX, _GW):
            X, Y = np.meshgrid((i + a) * h, (i + b) * h, indexing="ij")
            out += wa * wb * fun(X, Y)
    return out


def exact_scalar(n, t, sub=None):
    """Cell averages of q(x, t) = q0(X(-t; x)): the characteristics traced back with RK4."""
    sub = sub or max(8, int(math.ceil(t * 64)))

    def back(X, Y):
        dt = -t / sub
        for _ in range(sub):
            k1 = uv(X, Y)
            k2 = uv(X + 0.5 * dt * k1[0], Y + 0.5 * dt * k1[1])
            k3 = uv(X + 0.5 * dt * k2[0], Y + 0.5 * dt * k2[1])
            k4 = uv(X + dt * k3[0], Y + dt * k3[1])
            X = X + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            Y = Y + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        return q0(X, Y)
    return cell_average(back, n)


def predicted_terms():
    """Lambdified leading terms (per dt^2): the user's odd term for the scalar, and the missing-source term for the
    x-momentum."""
    import sympy as sp

    x, y = sp.symbols("x y")
    u = sp.sin(2 * sp.pi * x) * sp.cos(2 * sp.pi * y)
    v = -sp.cos(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    q = 2 + sp.sin(2 * sp.pi * x) * sp.cos(2 * sp.pi * y) + sp.Rational(1, 2) * sp.cos(2 * sp.pi * (x + 2 * y))
    A = lambda f: -u * sp.diff(f, x)  # noqa: E731
    B = lambda f: -v * sp.diff(f, y)  # noqa: E731
    d = sp.diff(u, x)
    odd = sp.Rational(1, 2) * (B(A(q)) - A(B(q)) + d * (A(q) + B(q)))
    comm = sp.Rational(1, 2) * (B(A(q)) - A(B(q)))
    gx = sp.pi * sp.sin(4 * sp.pi * x)                          # -dp/dx, p = (cos 4 pi x + cos 4 pi y) / 4
    src = sp.Rational(1, 2) * (u * sp.diff(gx, x) + v * sp.diff(gx, y))
    f = lambda e: sp.lambdify((x, y), e, "numpy")  # noqa: E731
    return {"odd": f(odd), "commutator": f(comm), "source": f(src), "gx": f(gx), "u": f(u)}


def _setup(n, field0, source=None, **opts):
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.interface.vof import VOFInterface, VOFTracer

    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    c = CellField.from_interior(grid, np.ones((n, n)), bc)
    t = CellField.from_interior(grid, field0.copy(), bc)
    tr = VOFTracer(t, source=source, **opts)
    itf = VOFInterface(c, [tr])
    uf = FaceField(grid, name="u")
    u, v = face_velocity(n)
    uf.components[0][...] = grid.xp.asarray(u)
    uf.components[1][...] = grid.xp.asarray(v)
    return itf, uf, t, tr


def one_step(n, dt, order, kind="scalar", source=False):
    """One transport step from exact data with the sweep order "xy" or "yx": the error field against the exact
    solution. kind "momentum": the x-momentum, the exact pressure acceleration added after the transport (and, with
    ``source``, also in the face value)."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.vof import vof_advection

    T = predicted_terms()
    if kind == "scalar":
        f0 = cell_average(q0, n)
        exact = exact_scalar(n, dt)
        src = None
    else:
        f0 = cell_average(T["u"], n)
        exact = f0
        g = cell_average(T["gx"], n)
        src = g if source else None
    itf, uf, t, _ = _setup(n, f0, source=src)
    vof_advection([itf], uf, dt, 0 if order == "xy" else 1)
    out = to_numpy(t.interior).copy()
    if kind == "momentum":
        out = out + dt * g
    return out - exact


def split(n, dt, kind="scalar", source=False):
    """Even / odd parts of the one-step error and their comparison with the predicted leading terms."""
    T = predicted_terms()
    exy, eyx = one_step(n, dt, "xy", kind, source), one_step(n, dt, "yx", kind, source)
    even, odd = 0.5 * (exy + eyx), 0.5 * (exy - eyx)
    nrm = lambda a: float(np.sqrt((a * a).mean()))  # noqa: E731
    out = {"n": n, "dt": dt, "kind": kind, "source": source, "even": nrm(even), "odd": nrm(odd),
           "even_over_dt2": nrm(even) / dt ** 2, "odd_over_dt2": nrm(odd) / dt ** 2}
    if kind == "scalar":
        pred = dt ** 2 * cell_average(T["odd"], n)
        comm = dt ** 2 * cell_average(T["commutator"], n)
        out.update({"odd_pred": nrm(pred), "odd_minus_pred": nrm(odd - pred), "odd_minus_commutator": nrm(odd - comm),
                    "corr_odd_pred": float((odd * pred).sum() / math.sqrt((odd * odd).sum() * (pred * pred).sum()))})
    else:
        pred = dt ** 2 * cell_average(T["source"], n)
        out.update({"even_pred": nrm(pred), "even_minus_pred": nrm(even - pred),
                    "corr_even_pred": float((even * pred).sum() / math.sqrt((even * even).sum() * (pred * pred).sum()))})
    return out


def global_error(n, cfl, t_end=0.25, kind="scalar", source=False, alternate=True):
    """Many steps on a fixed grid: the max error at t_end. ``alternate``: the library's alternating sweep order."""
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.vof import vof_advection

    T = predicted_terms()
    steps = int(math.ceil(t_end / (cfl / n) - 1e-9))
    dt = t_end / steps
    if kind == "scalar":
        f0 = cell_average(q0, n)
        exact = exact_scalar(n, t_end, sub=256)
        g = None
    else:
        f0 = cell_average(T["u"], n)
        exact = f0
        g = cell_average(T["gx"], n)
    itf, uf, t, tr = _setup(n, f0, source=(g if (source and g is not None) else None))
    for i in range(steps):
        vof_advection([itf], uf, dt, i if alternate else 0)
        if g is not None:
            t.interior = t.interior + dt * t.grid.xp.asarray(g)
    e = to_numpy(t.interior) - exact
    return {"n": n, "cfl": cfl, "steps": steps, "kind": kind, "source": source, "alternate": alternate,
            "max": float(np.abs(e).max()), "rms": float(np.sqrt((e * e).mean()))}


def note77(ns=(32, 64, 128)):
    from cases.pure_advection import tg_cfl

    return [tg_cfl(v, n, 0.5) for v in ("plain", "conserving", "conserving-src") for n in ns]


def main() -> None:
    res = {"note77": note77(), "split": [], "global": []}
    for r in res["note77"]:
        print("note 77", r["variant"], r["n"], "%.3e" % r["du_max"], flush=True)
    for kind, source in (("scalar", False), ("momentum", False), ("momentum", True)):
        for n in (64, 128, 256):
            for cfl in (0.4, 0.2, 0.1, 0.05):
                o = split(n, cfl / n, kind, source)
                res["split"].append(o)
                extra = ("odd-pred %.2e (corr %.4f), odd-comm %.2e" % (o["odd_minus_pred"], o["corr_odd_pred"], o["odd_minus_commutator"])
                         if kind == "scalar" else "even-pred %.2e (corr %.4f)" % (o["even_minus_pred"], o["corr_even_pred"]))
                print("split %-8s src=%-5s n=%3d cfl=%.2f  even %.3e (/dt^2 %.3e)  odd %.3e (/dt^2 %.3e)  %s"
                      % (kind, source, n, cfl, o["even"], o["even_over_dt2"], o["odd"], o["odd_over_dt2"], extra), flush=True)
    for kind, source, alt in (("scalar", False, True), ("scalar", False, False), ("momentum", False, True),
                              ("momentum", True, True), ("momentum", True, False)):
        for n in (64, 128):
            for cfl in (0.4, 0.2, 0.1, 0.05):
                o = global_error(n, cfl, 0.25, kind, source, alt)
                res["global"].append(o)
                print("global %-8s src=%-5s alt=%-5s n=%3d cfl=%.2f  max %.3e rms %.3e" % (kind, source, alt, n, cfl, o["max"], o["rms"]), flush=True)
    json.dump(res, open("docs/figures/r0p1_local_error.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------------------- P0-5: the coupled solver
def coupled(n, cfl, variant="conserving-src", exact_uf=False, exact_source=False, exact_g_final=False, t_end=0.5,
            stale_ghosts=False):
    """The library's solver (CenteredNS + TwoPhase with f = 0, inviscid Taylor-Green, docs/notes/77), with oracle
    replacements made from hooks (the library is not changed):
      exact_uf       the face velocity that carries the momentum is the exact steady one (not the projected uf^n);
      exact_source   the half-step source in the face value uses the exact pressure acceleration (not g^n);
      exact_g_final  after the step, u is reset to (transported u) + dt * exact g (the projection's result replaced).
    ``stale_ghosts``: conserving.h's momentum ghost cells (``TwoPhase(conserving_stale_ghosts=True)``: the library as
    it was when docs/notes/112 was written; the default has been the fixed one since audit #149 / issue #146).
    Returns max |u - u0| at t_end."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Periodic
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.convection_stability import conserving_options

    grid = CartesianGrid((n, n))
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    bc = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), bc), rho1=1.0, rho2=1.0,
             conserving_stale_ghosts=stale_ghosts, **conserving_options(variant))
    ns.cfl = cfl
    T = predicted_terms()
    u0 = cell_average(T["u"], n)
    v0 = cell_average(lambda x, y: -np.cos(TWO_PI * x) * np.sin(TWO_PI * y), n)
    gx = cell_average(T["gx"], n)
    gy = cell_average(lambda x, y: math.pi * np.sin(4 * math.pi * y), n)
    ufx, ufy = face_velocity(n)
    state = {}

    def before_vof(solver):
        if exact_uf:
            solver.uf.components[0][...] = xp.asarray(ufx)
            solver.uf.components[1][...] = xp.asarray(ufy)
        if exact_source:
            solver.g[0].interior = xp.asarray(gx)
            solver.g[1].interior = xp.asarray(gy)

    def after_vof(solver):                                       # the transported velocity, before any force
        state["ustar"] = [c.interior.copy() for c in solver.u.components]

    def end(solver):
        if exact_g_final:
            dt = solver.clock.dt
            solver.u[0].interior = state["ustar"][0] + dt * xp.asarray(gx)
            solver.u[1].interior = state["ustar"][1] + dt * xp.asarray(gy)
            solver.u.fill_ghosts()

    ns.on("vof", before_vof)                                     # registered later: runs before TwoPhase's vof hook
    ns.on("tracer_advection", after_vof)
    ns.on("end_timestep", end)
    ns.u[0].interior = xp.asarray(u0)
    ns.u[1].interior = xp.asarray(v0)
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()
    du = np.hypot(to_numpy(ns.u[0].interior) - u0, to_numpy(ns.u[1].interior) - v0)
    return {"n": n, "cfl": cfl, "variant": variant, "exact_uf": exact_uf, "exact_source": exact_source,
            "exact_g_final": exact_g_final, "steps": ns.clock.i, "max": float(du.max())}


class periodic_momentum_ghosts:
    """(Since issue #146 / audit #149 the library's default does this, for every boundary: this patch only matters
    with ``conserving_stale_ghosts=True``.)
    Context manager (the library is not changed): TwoPhase's momentum fields take their ghost cells from the
    step-start velocity (``_MomentumField.fill_ghosts``, after Basilisk's boundary_q1_x). With periodic boundaries
    this leaves the ghosts stale in the second sweep. Here, on a fully periodic grid, the ghosts are the periodic
    images of the field itself."""

    def __enter__(self):
        import basilisk_cupy.interface.two_phase as tp
        from basilisk_cupy import CellField

        self._tp, self._saved = tp, tp._MomentumField.fill_ghosts

        def fill(field):
            if all(field.bc.is_periodic(ax) for ax in range(field.grid.ndim)):
                return CellField.fill_ghosts(field)
            return self._saved(field)
        tp._MomentumField.fill_ghosts = fill
        return self

    def __exit__(self, *exc):
        self._tp._MomentumField.fill_ghosts = self._saved
        return False

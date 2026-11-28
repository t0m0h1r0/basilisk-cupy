"""docs/notes/122 (A-3 stage 1, #166): every modification of R0-P at once, from hooks (the library is unchanged).

Configuration (#166): conserving transport, filtered=False, reduced gravity, the momentum ghosts (#154, library),
c = 1/2 (#160, library), D1 + D2 limited (docs/notes/119), E1-E3 (docs/notes/113: the carrying face velocity
extrapolated to t + dt/2, the viscous acceleration in the face value's half-step source, Crank-Nicolson), and the
face value's source switched off within w cells of the interface (docs/notes/117, w = 1).

``combined()`` patches TwoPhase.__init__ so that every TwoPhase built inside gets the E1-E3 and band hooks
(``conserving_source`` on, the source masked); the D1 / D2 hooks come from ``cases.r0_aligned_free_stream.scheme``
(``full()`` enters both). So the existing test functions of notes 113-120 run unchanged with the full configuration.

The checks (#166, fixed before the results):
  1 single-phase time accuracy (note 113's tg / asym / mix): O(h^2 + h dt + dt^2), read as in note 113
  2 free stream / stability (note 117's shapes, note 118's aligned shapes, ratios 1e3 / 1e6; the CFL bound; E1 can
    raise the effective CFL number): max |u - U| at the pressure tolerance, no growth
  3 total momentum with the pressure and the phase mass (drop in a vortex, ratio 1000): 1e-9
  4 hydrostatic balance (reduced, note 115's geometries without the cause of note 120): round-off to tolerance
  5 volume-fraction overshoot as in the base scheme, rho_lin's floor never reached
  6 accuracy where D2 limited cuts: the grid-parallel two-layer shear flow + boost (exact, ratio 1000), within
    1.2 x the base scheme (D1 only, no cut)
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import numpy as np

BAND = 1


def _setup(ns, tp, e1=True, e2=True, e3=True, band=BAND, record=None, e3_form="velocity", theta=0.5, source=True):
    """The E1-E3 and band hooks on a built solver (TwoPhase(conserving=True))."""
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity

    grid = tp.f.grid
    xp = grid.xp
    tp.conserving_source = source                                # the face value's half-step source (note 112)
    viscous = ns.mu is not None and (tp.mu1 or tp.mu2)
    st = {"uf_prev": None, "dt_prev": None}

    def lu(solver, dt):
        """(1 / rho) L_h u of the current u (the library's viscosity() stencil)."""
        solver.u.fill_ghosts()
        prob = ViscosityProblem.reuse(solver.u, solver.mu, solver.rho, dt)
        res = [xp.zeros(grid.n) for _ in range(2)]
        prob.residual(list(solver.u), [xp.zeros(grid.n) for _ in range(2)], res)
        return [(r + c.interior) / dt for r, c in zip(res, solver.u.components)]

    def mask():
        fi = tp.f.interior
        m = ((fi > 1e-12) & (fi < 1.0 - 1e-12)).astype(float)
        out = xp.zeros_like(m)
        per = tp.f.bc.periodic_axes(2)
        for i in range(-band, band + 1):
            for j in range(-band, band + 1):
                sh = m
                if i:
                    sh = xp.roll(sh, i, axis=0)
                    if not per[0]:                               # no wrap-around at a wall (B, PR #161)
                        sh[(slice(0, i) if i > 0 else slice(i, None)), :] = 0.0
                if j:
                    sh = xp.roll(sh, j, axis=1)
                    if not per[1]:
                        sh[:, (slice(0, j) if j > 0 else slice(j, None))] = 0.0
                out = xp.maximum(out, sh)
        return 1.0 - out

    def before(solver):                                          # "vof": before TwoPhase's transport
        dt = solver.clock.dt
        st["Lun"] = lu(solver, dt) if (viscous and (e2 or e3)) else None
        if e1:
            cur = [c.copy() for c in solver.uf.components]
            if st["uf_prev"] is not None:
                r = 0.5 * dt / st["dt_prev"]
                for c, a, b in zip(solver.uf.components, cur, st["uf_prev"]):
                    c[...] = a + r * (a - b)
                if record is not None:                           # the effective CFL number of the carrying velocity
                    h = float(grid.h[0])
                    record["cfl_max"] = max(record.get("cfl_max", 0.0),
                                            max(float(xp.abs(c).max()) for c in solver.uf.components) * dt / h)
            st["uf_prev"] = cur
        st["dt_prev"] = dt
        st["rho_n"] = solver.rho.copy() if hasattr(solver.rho, "copy") else solver.rho
        st["g"] = [c.interior.copy() for c in solver.g.components]
        w = mask() if band is not None else 1.0
        for k, c in enumerate(solver.g.components):
            src = c.interior + (st["Lun"][k] if (e2 and st["Lun"] is not None) else 0.0)
            c.interior = src * w
        solver.g.fill_ghosts()

    def after(solver):                                           # "tracer_advection": restore g
        for c, g0 in zip(solver.g.components, st["g"]):
            c.interior = g0
        solver.g.fill_ghosts()

    class CrankNicolson:
        """theta-scheme (0.5: Crank-Nicolson, note 113's E3). e3_form "velocity": the explicit part is added as the
        acceleration (1 / rho^n) L u^n (note 113, single phase); "momentum": as the momentum L u^n, divided by the
        present density rho^{n+1} (diagnosis of note 121)."""

        def solve(self, u, mu, rho, dt, tolerance=1e-3):
            w = (1.0 - theta) * dt
            fac = (st["rho_n"] / rho) if e3_form == "momentum" else 1.0
            for c, a in zip(u.components, st["Lun"]):
                c.interior = c.interior + w * a * fac
            u.fill_ghosts()
            return viscosity(u, mu, rho, theta * dt, tolerance=tolerance)

    if viscous and e3:
        ns.viscous = CrankNicolson()
    ns.on("vof", before)
    ns.on("tracer_advection", after)


@contextlib.contextmanager
def combined(**kw):
    """TwoPhase built inside gets the E1-E3 / band hooks (kw: e1, e2, e3, band, record)."""
    from basilisk_cupy.interface import two_phase

    init = two_phase.TwoPhase.__init__

    def patched(self, ns, f, **k):
        init(self, ns, f, **k)
        if k.get("conserving"):
            _setup(ns, self, **kw)

    two_phase.TwoPhase.__init__ = patched
    try:
        yield
    finally:
        two_phase.TwoPhase.__init__ = init


@contextlib.contextmanager
def full(**kw):
    """The whole configuration: D1 + D2 limited and E1-E3 + band."""
    from cases.r0_aligned_free_stream import scheme

    with scheme(True, "limited"), combined(**kw):
        yield


# --- the checks -----------------------------------------------------------------------------------------------------
MU = (1e-2, 1e-2 / 55.0)
EKW = {"full-noE3": {"e3": False}}   # added after the first runs: E3 (Crank-Nicolson) fails in two phases (note 121)


@contextlib.contextmanager
def configuration(conf, record=None):
    """"base": the library; "d1": D1 only; "d1+d2lim": D1 + D2 limited; "d1+E": D1 + E1-E3 + band (no D2);
    "full": everything. For functions that are not ``in_scheme`` (hydrostatic, shear)."""
    from cases.r0_aligned_free_stream import scheme

    d2 = {"base": None, "base-hooked": False, "d1": False, "d1+d2lim": "limited", "d1+E": False, "full": "limited",
          "full-noE3": "limited"}[conf]
    with (scheme(conf != "base-hooked", d2) if d2 is not None else contextlib.nullcontext()), \
            (combined(record=record, **EKW.get(conf, {})) if conf in ("d1+E", "full", "full-noE3")
             else contextlib.nullcontext()):
        yield


def c1_identity(grids=(32, 64, 96), flows=("tg", "asym", "mix")):
    """1: single phase, the full configuration on note 113's "cons" is bit-identical to note 113's corrected
    "cons/ab+visc+cn" (then note 113's verdict, O(h^2 + h dt + dt^2), carries over)."""
    import tempfile

    from cases.r0p1_unsteady_viscous import CONFIGS, NU, run

    out = []
    with tempfile.TemporaryDirectory() as d:
        for flow in flows:
            for n in grids:
                run(n, 0.5, flow=flow, nu=NU[flow], save=d + "/a.npy", **CONFIGS["cons/ab+visc+cn"])
                with full():
                    run(n, 0.5, flow=flow, nu=NU[flow], save=d + "/b.npy", **CONFIGS["cons"])
                out.append({"flow": flow, "n": n,
                            "difference": float(np.abs(np.load(d + "/a.npy") - np.load(d + "/b.npy")).max())})
    return out


def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, conf, kw = a
    from cases import r0_aligned_free_stream as m

    rec = {}
    m.FLOOR[0] = 0
    m.COUNT[0] = m.COUNT[1] = 0
    if kind == "free":                                           # free_stream / vortex_budget enter the D scheme
        from cases.r0_d2_forms import vortex_budget            # noqa: F401
        sch = {"full": "d1+d2lim", "full-noE3": "d1+d2lim", "d1+d2lim": "d1+d2lim", "base": "base"}[conf]
        with combined(record=rec, **EKW.get(conf, {})) if conf.startswith("full") else contextlib.nullcontext():
            r = m.free_stream(sch=sch, **kw)
    elif kind == "vortex":
        from cases.r0_d2_forms import vortex_budget

        sch = {"full": "d1+d2lim", "full-noE3": "d1+d2lim", "d1+d2lim": "d1+d2lim", "base": "base"}[conf]
        with combined(record=rec, **EKW.get(conf, {})) if conf.startswith("full") else contextlib.nullcontext():
            r = vortex_budget(sch=sch, **kw)
    elif kind == "hydrostatic":
        from cases.r0p2_pressure_momentum import hydrostatic

        with configuration(conf, rec):
            r = hydrostatic(**kw)
    elif kind == "shear":
        from cases.r0_shear import run

        with configuration(conf, rec):
            r = run(**kw)
    else:
        raise ValueError(kind)
    r.update({"kind": kind, "conf": conf, "args": kw, "floor_hits": m.FLOOR[0], "cut": m.COUNT[0],
              "partial": m.COUNT[1], "cfl_max": rec.get("cfl_max")})
    return r


def jobs():
    J = []
    U0 = (1.0, 0.37)
    for conf in ("full", "d1+d2lim"):
        for name in ("column", "square", "square-bubble"):
            for r in (1e3, 1e6):
                for U in ((1.0, 0.0), U0):
                    J.append(("free", conf, dict(name=name, ratio=r, U=U)))
        for name in ("column", "square"):
            for r in (1e3, 1e6):
                J.append(("free", conf, dict(name=name, ratio=r, dt=0.4999 / 32)))
        for name in ("drop", "bubble", "small", "film", "slant", "two", "band2"):
            for r in (1e3, 1e6):
                J.append(("free", conf, dict(name=name, ratio=r, U=U0, steps=2000)))
    for conf in ("full", "full-noE3", "d1+d2lim"):
        for name in ("column", "drop", "band2"):                   # viscous: E2 / E3 act
            J.append(("free", conf, dict(name=name, ratio=1e3, U=U0, mu=MU)))
        for n in (32, 64):
            J.append(("vortex", conf, dict(n=n, mu=MU)))
    for conf in ("full", "d1+d2lim"):
        for n in (32, 64):
            J.append(("vortex", conf, dict(n=n)))
        J.append(("vortex", conf, dict(n=64, ratio=1e6, t_end=0.1)))
    # docs/notes/115's 24 geometries without note 120's three (the walls' mirrored ghosts, a 90 degree contact angle)
    GEOMETRIES = ([(th, 0.0137, n) for th in (0.0, 10.0) for n in (32, 64, 128)]
                  + [(th, off, n) for th in (26.565, 45.0) for off in (0.0137, 0.1171, 0.2313) for n in (32, 64, 128)])
    FAILING = [(45.0, 0.0137, 32), (45.0, 0.0137, 64), (26.565, 0.2313, 32)]
    for th, off, n in GEOMETRIES:
        if (th, off, n) in FAILING:
            continue
        for kind in ("smooth", "mean"):
            J.append(("hydrostatic", "full", dict(kind=kind, theta_deg=th, gravity="reduced", n=n, offset=off)))
    for conf in ("base", "d1", "d1+d2lim", "d1+E", "full", "full-noE3"):
        for N in (32, 48, 64, 96, 128):
            J.append(("shear", conf, dict(n=(0, 1), boost=True, N=N)))
            if N <= 64:
                J.append(("shear", conf, dict(n=(1, 1), boost=True, N=N)))
    return J


def _key(a):
    return json.dumps([a[0], a[1], {k: (list(v) if isinstance(v, tuple) else v) for k, v in a[2].items()}],
                      sort_keys=True)


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_combined.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    J = [a for a in jobs() if _key(a) not in done]
    cost = {"hydrostatic": 3, "shear": 2, "vortex": 1, "free": 0}
    J.sort(key=lambda a: -(cost[a[0]] * 1000 + a[2].get("n", a[2].get("N", 32)) if isinstance(a[2].get("n", 0), int)
                           else cost[a[0]] * 1000 + a[2].get("N", 32)))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "22")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job, J, chunksize=1)):
            r["job"] = _key(a)
            with open(part, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            done[r["job"]] = r
    res = {"runs": list(done.values()), "c1_identity": c1_identity()}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()


def shear_inviscid(conf, N=64, cfl=0.5, T=None, U=(0.0, 1.0), amp=0.5, steps_max=None):
    """Condition 6 where D2 limited does cut (added after the viscous shear flow cut no face): the heavy layer
    1/4 <= y < 3/4 (faces-aligned), no viscosity, w(y) = amp sin(4 pi (y - 1/4)) along x (zero and continuous at both
    interfaces: no vortex sheet), carried by the boost U. The exact solution is the translation by U; T = one period
    of the interface's motion (1 / Uy). Returns the error of u (cell averages, L1 / L2 / Linf) at T and the cut count."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    from cases import r0_aligned_free_stream as m

    T = 1.0 / U[1] if T is None else T
    k = 4 * math.pi

    def cells(t):
        y0 = np.arange(N) / N - U[1] * t
        return amp * (np.cos(k * (y0 - 0.25)) - np.cos(k * (y0 + 1.0 / N - 0.25))) / (k / N)   # cell averages

    def frac(t):
        y0 = np.arange(N) / N - U[1] * t
        lo, hi = 0.25, 0.75
        out = np.zeros(N)
        for s in (-2, -1, 0, 1):
            out += np.clip(np.minimum(y0 + 1.0 / N, hi + s) - np.maximum(y0, lo + s), 0, None) * N
        return out

    m.COUNT[0] = m.COUNT[1] = 0
    with configuration(conf):
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        tp = TwoPhase(ns, CellField.from_interior(grid, np.tile(frac(0.0), (N, 1)), BoundaryConditions.periodic(2)),
                      rho1=1.0, rho2=1e-3, conserving=True)
        ns.cfl = cfl
        ns.u[0].interior = U[0] + np.tile(cells(0.0), (N, 1))
        ns.u[1].interior = U[1] + 0.0 * ns.u[1].interior
        ns.start(events=[T])
        failed = None
        try:
            while ns.clock.t < T - 1e-12:
                ns.step()
                if steps_max and ns.clock.i > steps_max:
                    break
                if float(np.abs(to_numpy(ns.u[0].interior)).max()) > 1e3:
                    failed = "blow-up at t = %.4f" % ns.clock.t
                    break
        except Exception as exc:  # noqa: BLE001
            failed = repr(exc)[:200]
        eu = to_numpy(ns.u[0].interior) - U[0] - np.tile(cells(ns.clock.t), (N, 1))
        ev = to_numpy(ns.u[1].interior) - U[1]
        e = np.hypot(eu, ev)
    return {"conf": conf, "N": N, "cfl": cfl, "t": float(ns.clock.t), "failed": failed, "L1": float(e.mean()),
            "L2": float(np.sqrt((e * e).mean())), "Linf": float(e.max()), "cut": m.COUNT[0], "partial": m.COUNT[1],
            "f_L1": float(np.abs(to_numpy(tp.f.interior) - np.tile(frac(ns.clock.t), (N, 1))).mean())}


def plot() -> None:
    """docs/figures/r0_combined.png: (a) condition 6, the error of D1 + D2 limited over D1 only where D2 cuts (the
    inviscid aligned shear flow); (b) the viscous two-phase runs with and without E3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c6 = json.load(open(os.path.join("docs", "figures", "r0_combined_c6.json")))
    R = json.load(open(os.path.join("docs", "figures", "r0_combined.json")))["runs"]
    T = {(r["conf"], r["N"], r["cfl"]): r for r in c6}
    NS = sorted({r["N"] for r in c6 if r["conf"] == "d1+d2lim"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfl, ls in ((0.5, "-"), (0.45, "--")):
        for m, mk in (("L1", "o"), ("L2", "s"), ("Linf", "^")):
            ax[0].semilogx(NS, [T[("d1+d2lim", n, cfl)][m] / T[("d1", n, cfl)][m] for n in NS], ls, marker=mk, ms=4,
                           label="%s, CFL %g" % (m, cfl))
    ax[0].axhline(1.2, color="tab:red", lw=1)
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("error ratio: D1 + D2 limited / D1")
    ax[0].set_title("(a) condition 6: where D2 limited cuts")
    ax[0].legend(fontsize=7)
    labels, vals, cols = [], [], []
    for r in R:
        if r["kind"] == "free" and tuple(r["args"].get("mu", (0, 0)))[0] > 0:
            labels.append("%s\n%s" % (r["args"]["name"], r["conf"]))
            vals.append(max(r["max_err"], 1e-16))
            cols.append({"full": "tab:red", "full-noE3": "tab:blue", "d1+d2lim": "0.5"}[r["conf"]])
    o = np.argsort(labels)
    ax[1].bar(range(len(o)), [vals[k] for k in o], color=[cols[k] for k in o])
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(o)))
    ax[1].set_xticklabels([labels[k] for k in o], fontsize=6, rotation=90)
    ax[1].set_ylabel("max |u - U| (viscous free stream, ratio 1000)")
    ax[1].set_title("(b) E3 (red: with, blue: without)")
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0_combined.png"), dpi=130)


def _c6_job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return shear_inviscid(a[0], N=a[1], cfl=a[2])


def c6() -> None:
    """Condition 6 where D2 limited cuts: docs/figures/r0_combined_c6.json (``python -c "from cases.r0_combined
    import c6; c6()"``, 20 processes, a few minutes)."""
    import multiprocessing as mp

    J = [(c, N, cfl) for c in ("base", "d1", "d1+d2lim", "full-noE3") for N in (32, 48, 64, 96, 128)
         for cfl in (0.5, 0.45)]
    J += [(c, N, cfl) for c in ("d1", "d1+d2lim") for N in (192, 256) for cfl in (0.5, 0.45)]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "20"))) as pool:
        R = pool.map(_c6_job, J)
    with open(os.path.join("docs", "figures", "r0_combined_c6.json"), "w") as fh:
        json.dump(R, fh, indent=1)

"""docs/notes/115 (R0-P2, issue #141): the momentum with the pressure included, and the hydrostatic balance, for the
cell-centred solver as it is (no new scheme here: the measurement that decides whether one is needed).

  momentum()     a drop in a vortex, periodic, no force, density ratio 1000: the change of the total momentum
                 sum rho u for the three face-density options and the two transports.
                   "smooth"  two-phase.h: 1 / rho(mean f) on the faces (the arithmetic mean of the cells' densities);
                   "mean"    TwoPhaseVOFJump (sharp GFM face coefficient for the face velocity) with the cells'
                             correction rescaled to the arithmetic mean (docs/notes/65, the default of that class);
                   "gfm"     the GFM coefficient for the cells too.
  one_d()        the derivation of docs/notes/111, section 4, as a matrix identity in 1-D.
  hydrostatic()  two layers at rest in a closed box, the plane interface at the angle theta to the x axis, gravity
                 ALONG THE NORMAL of the interface (docs/notes/111: with a vertical gravity an oblique plane
                 interface is not an equilibrium); gravity as the plain acceleration on the faces or as reduced.h's
                 interfacial potential. The velocities of the cells and of the faces in time.

    python -m cases.r0p2_pressure_momentum        # docs/figures/r0p2_pressure_momentum.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

TWO_PI = 2 * math.pi
KINDS = ("smooth", "mean", "gfm")


def _bc(periodic):
    from basilisk_cupy import BoundaryConditions
    from basilisk_cupy.boundary import Periodic

    bc = BoundaryConditions()
    for ax in periodic:
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    return bc


def _two_phase(ns, f, kind, ratio, conserving):
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.interface.vof_jump import TwoPhaseVOFJump

    if kind in ("smooth", "smooth-filtered"):   # "smooth-filtered": two-phase.h's FILTERED (audit, PR #157)
        return TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, conserving=conserving, filtered=kind == "smooth-filtered")
    return TwoPhaseVOFJump(ns, f, rho1=1.0, rho2=1.0 / ratio, conserving=conserving, cell_coefficient=kind)


# ------------------------------------------------------------------------------------------- total momentum, periodic
def momentum(kind, conserving=True, n=32, ratio=1000.0, t_end=1.0, samples=(0.05, 0.25, 0.5, 1.0)):
    """A drop (R = 0.15 at (0.5, 0.3)) in the Taylor-Green vortex plus the uniform flow (0.5, 0.2): the largest
    relative change of sum rho u over the run (no force, periodic: zero in the continuum), and in time the
    difference between the face velocity of the projection and the cells' velocity averaged to the faces
    (max |uf - interp(u)|, and its ratio to max |u|): it must not accumulate (issue #141)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    tp = _two_phase(ns, CellField.from_interior(grid, f0, _bc((0, 1))), kind, ratio, conserving)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = xp.asarray(0.5 + np.sin(TWO_PI * X) * np.cos(TWO_PI * Y))
    ns.u[1].interior = xp.asarray(0.2 - np.cos(TWO_PI * X) * np.sin(TWO_PI * Y))

    def mom():
        rho = to_numpy(tp.rho(tp.f.interior))
        return np.array([float((rho * to_numpy(ns.u[a].interior)).sum()) / n ** 2 for a in range(2)])

    def gap():
        ns.u.fill_ghosts()
        d = [to_numpy(ns.uf.components[a]) - to_numpy(ns.ops.interp_f(ns.u[a], a)) for a in range(2)]
        return (max(float(np.abs(x).max()) for x in d), float(np.sqrt(np.mean([float((x * x).mean()) for x in d]))))

    m0 = mom()
    ns.start(events=sorted(set(samples) | {t_end}))
    worst, umax, failed, hist = 0.0, 0.0, None, []
    for te in sorted(x for x in samples if x <= t_end + 1e-12):
        while ns.clock.t < te - 1e-12:
            ns.step()
            if ns.clock.i > 40 * n:                              # (the state at rest would take about 4 n steps)
                failed = "step limit: the time step collapsed at t = %.4f" % ns.clock.t
                break
            worst = max(worst, float(np.hypot(*(mom() - m0)) / np.hypot(*m0)))
            umax = float(np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)).max())
            if not math.isfinite(umax) or umax > 1e3:
                failed = "blow-up at t = %.4f" % ns.clock.t
                break
        if failed:
            break
        g = gap()
        hist.append({"t": te, "momentum_change": worst, "umax": umax, "face_cell_gap": g[0], "face_cell_gap_rms": g[1],
                     "steps": ns.clock.i})
    return {"kind": kind, "conserving": conserving, "n": n, "ratio": ratio, "steps": ns.clock.i,
            "momentum_change": worst, "umax": umax, "failed": failed, "history": hist}


def wall_budget(kind, n=32, ratio=1000.0, t_end=0.5):
    """B's request on PR #157: the momentum budget WITH the walls' pressure force. Closed box (slip walls), no body
    force, a drop (R = 0.15 at (0.5, 0.3), density ratio ``ratio``) in the Taylor-Green vortex (a slip-wall flow),
    conserving transport. The cells take the mean of their two faces' forces and the wall faces carry none, so with
    the arithmetic face density the sum over the cells telescopes to the pressures of the wall-adjacent cells:

        d/dt sum rho u_x h^2 = h sum_j (p[0, j] - p[n-1, j]),      d/dt sum rho u_y h^2 = h sum_i (p[i, 0] - p[i, n-1])

    (the discrete wall force: the wall pressure is the adjacent cell's). Returned: the largest residual of this
    budget over the run, |P(t) - P(0) - sum dt F_wall|, normalised by sum rho |u| h^2 at t = 0, and the size of the
    wall impulse itself (the budget is not trivially zero)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(), tolerance=1e-10)
    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    tp = _two_phase(ns, CellField.from_interior(grid, f0, _bc(())), kind, ratio, True)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = xp.asarray(np.sin(TWO_PI * X) * np.cos(TWO_PI * Y))
    ns.u[1].interior = xp.asarray(-np.cos(TWO_PI * X) * np.sin(TWO_PI * Y))
    h = 1.0 / n

    def mom():
        rho = to_numpy(tp.rho(tp.f.interior))
        return np.array([float((rho * to_numpy(ns.u[a].interior)).sum()) * h * h for a in range(2)])

    rho0 = to_numpy(tp.rho(tp.f.interior))
    scale = float((rho0 * np.hypot(to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior))).sum()) * h * h
    m0, imp = mom(), np.zeros(2)
    ns.start(events=[t_end])
    worst, failed = 0.0, None
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        p = to_numpy(ns.p.interior)
        imp += ns.clock.dt * h * np.array([float((p[0, :] - p[-1, :]).sum()), float((p[:, 0] - p[:, -1]).sum())])
        worst = max(worst, float(np.abs(mom() - m0 - imp).max()) / scale)
        if ns.clock.i > 40 * n:
            failed = "step limit"
            break
    return {"kind": kind, "n": n, "ratio": ratio, "steps": ns.clock.i, "budget_residual": worst,
            "wall_impulse": float(np.abs(imp).max()) / scale, "momentum_change": float(np.abs(mom() - m0).max()) / scale,
            "failed": failed}


def one_d(n=16, ratio=1000.0, seed=0):
    """docs/notes/111, section 4, as a matrix identity (1-D, periodic): the cells take the mean of their two faces'
    accelerations -(dp)_f / (h rho_f). The change of the total momentum for a random pressure,
        dP / dt = -sum_f [(rho_L + rho_R) / (2 rho_f)] (dp)_f / h,
    is zero for every pressure only if rho_f is the arithmetic mean. Returns |dP/dt| for three face densities."""
    rng = np.random.default_rng(seed)
    rho = np.where((np.arange(n) >= n // 4) & (np.arange(n) < 3 * n // 4), 1.0, 1.0 / ratio)
    rho[n // 4] = 0.37 + 0.63 / ratio                            # an interfacial cell
    p = rng.standard_normal(n)
    rl, rr = rho, np.roll(rho, -1)                               # face i + 1/2 between the cells i and i + 1
    dp = np.roll(p, -1) - p
    out = {}
    for name, rf in (("arithmetic", 0.5 * (rl + rr)), ("harmonic", 2 * rl * rr / (rl + rr)),
                     ("gfm, theta = 0.3", 0.3 * rl + 0.7 * rr)):
        a = -dp / rf                                             # face accelerations (h = 1)
        cell = 0.5 * (a + np.roll(a, 1))
        out[name] = {"dP": float(abs((rho * cell).sum())),
                     "formula": float(abs((-(rl + rr) / (2 * rf) * dp).sum()))}
    return out


# ---------------------------------------------------------------------------------------------- hydrostatic balance
def hydrostatic(kind, theta_deg=0.0, gravity="plain", n=32, ratio=1000.0, t_end=1.0, conserving=True, offset=0.0137,
                samples=(0.01, 0.1, 0.25, 0.5, 1.0)):
    """Closed box [0, 1]^2 (slip walls), two layers at rest: the heavy phase where n . (x - x0) < 0, the unit normal
    n = (sin theta, cos theta), x0 = (0.5, 0.5 + offset); gravity -G n (G = 1) along the normal.
      gravity "plain":   the acceleration -G n on every interior face (zero on the wall faces, whose normal
                         velocity is zero);
              "reduced": reduced.h's potential (rho2 - rho1) G . (x - Z) at the interface (TwoPhase.add_reduced_gravity).
    The exact state is u = 0 with p = p0 - G int rho ds along the normal. Returned in time: the max of the cells'
    and of the faces' velocities, and the largest change of the volume fraction. The time step is 0.5 h."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    th = math.radians(theta_deg)
    nx, ny = math.sin(th), math.cos(th)
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(), tolerance=1e-10)
    f0 = to_numpy(fraction(grid, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - offset))))   # a plane: exact
    tp = _two_phase(ns, CellField.from_interior(grid, f0, _bc(())), kind, ratio, conserving)
    if gravity == "plain":
        ns.a_is_constant = False

        def grav(s):
            ax_, ay_ = s.a[0] - nx, s.a[1] - ny
            ax_[0, :] = 0.0
            ax_[-1, :] = 0.0
            ay_[:, 0] = 0.0
            ay_[:, -1] = 0.0
            s.a[0], s.a[1] = ax_, ay_

        ns.on("acceleration", grav)
    else:
        tp.add_reduced_gravity(G=(-nx, -ny), Z=(0.5, 0.5 + offset))
    ns.DT = 0.5 / n
    out = {"kind": kind, "theta": theta_deg, "gravity": gravity, "n": n, "ratio": ratio, "conserving": conserving,
           "offset": offset, "history": [], "failed": None}
    ns.start(events=sorted(set(samples) | {t_end}))
    try:
        for te in sorted(s for s in samples if s <= t_end + 1e-12):
            while ns.clock.t < te - 1e-12:
                ns.step()
                if ns.clock.i > 40 * n * max(1.0, t_end):        # 20 times the steps of the state at rest
                    raise RuntimeError("step limit: the time step collapsed at t = %.4f (max |uf| = %.3g)"
                                       % (ns.clock.t, max(float(abs(to_numpy(c)).max()) for c in ns.uf.components)))
            uc = max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u.components)
            uf = max(float(np.abs(to_numpy(c)).max()) for c in ns.uf.components)
            df = float(np.abs(to_numpy(tp.f.interior) - f0).max())
            out["history"].append({"t": te, "cells": uc, "faces": uf, "df": df, "steps": ns.clock.i})
            if not math.isfinite(uc) or uc > 1e3:
                out["failed"] = "blow-up"
                break
    except Exception as exc:  # noqa: BLE001
        out["failed"] = repr(exc)[:200]
    return out


def filtered(ns_=(32, 64, 128)):
    """Audit on PR #157: with two-phase.h's FILTERED (the face densities from the vertex-averaged f, while the
    momentum counts f itself) the face density is no longer the mean of the two cells' densities (section B), so the
    total momentum is not conserved. Adds "momentum_filtered" to docs/figures/r0p2_pressure_momentum.json."""
    import multiprocessing as mp

    path = "docs/figures/r0p2_pressure_momentum.json"
    res = json.load(open(path))
    with mp.get_context("spawn").Pool(len(ns_)) as pool:
        res["momentum_filtered"] = pool.starmap(momentum, [("smooth-filtered", True, n) for n in ns_])
    json.dump(res, open(path, "w"), indent=1, default=float)


def _job(a):
    kind, args, kw = a
    return kind, {"momentum": momentum, "hydrostatic": hydrostatic, "wall_budget": wall_budget}[kind](*args, **kw)


def jobs():
    out = [("momentum", (k, c, n), {}) for k in KINDS for c in (True, False) for n in (32, 64, 128)]
    for k in KINDS:
        for grav in ("plain", "reduced"):
            for th in (0.0, 10.0, 26.565, 45.0):
                for n in (32, 64, 128):
                    out.append(("hydrostatic", (k, th, grav, n), {}))
    out += [("hydrostatic", (k, 26.565, grav, 64), {"ratio": r}) for k in KINDS for grav in ("plain", "reduced")
            for r in (1.0, 10.0)]
    out += [("hydrostatic", (k, 26.565, grav, 64), {"conserving": False}) for k in KINDS for grav in ("plain", "reduced")]
    # other positions of the interface (with offset 0.0137 the 45 degree line passes within 0.0137 of two corners)
    out += [("hydrostatic", (k, th, grav, n), {"offset": off}) for k in KINDS for grav in ("plain", "reduced")
            for th in (26.565, 45.0) for off in (0.1171, 0.2313) for n in (32, 64, 128)]
    return sorted(out, key=lambda a: -a[1][-1] if a[0] == "momentum" else -a[1][3])


def main() -> None:
    import multiprocessing as mp

    res = {"one_d": one_d(), "momentum": [], "hydrostatic": []}
    print("one_d", res["one_d"], flush=True)
    part = os.environ.get("R0P2_PARTIAL")
    todo = jobs()
    if part and os.path.exists(part):                            # a rerun skips the runs already there
        key = lambda k, o: (k, o["kind"], o.get("theta"), o.get("gravity"), o["n"], o["ratio"], o["conserving"],  # noqa: E731
                            o.get("offset", 0.0137) if k == "hydrostatic" else None)
        done = set()
        for line in open(part):
            k, o = json.loads(line)
            res[k].append(o)
            done.add(key(k, o))
        todo = [a for a in todo if (a[0], a[1][0], a[1][1] if a[0] == "hydrostatic" else None,
                                    a[1][2] if a[0] == "hydrostatic" else None, a[1][-1] if a[0] == "momentum" else a[1][3],
                                    a[2].get("ratio", 1000.0),
                                    a[2].get("conserving", True) if a[0] == "hydrostatic" else a[1][1],
                                    a[2].get("offset", 0.0137) if a[0] == "hydrostatic" else None) not in done]
        print(len(todo), "runs to do", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P2_PROCS", "16"))) as pool:
        for kind, o in pool.imap_unordered(_job, todo):
            res[kind].append(o)
            if part:
                with open(part, "a") as fh:
                    fh.write(json.dumps([kind, o], default=float) + "\n")
            last = o["history"][-1] if o.get("history") else {}
            print(kind, {k: (float("%.3g" % v) if isinstance(v, float) else v) for k, v in o.items() if k != "history"},
                  {k: float("%.3g" % v) for k, v in last.items()}, flush=True)
    for k in ("momentum", "hydrostatic"):
        res[k].sort(key=lambda o: json.dumps({a: b for a, b in o.items() if a != "history"}, sort_keys=True, default=str))
    json.dump(res, open("docs/figures/r0p2_pressure_momentum.json", "w"), indent=1, default=float)
    print("wrote docs/figures/r0p2_pressure_momentum.json")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.exit(main())

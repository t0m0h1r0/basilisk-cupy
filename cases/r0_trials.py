"""docs/notes/123 (#166, stage 2): the bounded trials of the minimal-fix range, with the pass conditions fixed in
#166's body before the runs.

  E3  backward Euler on the faces of the interface band, Crank-Nicolson elsewhere (flux form; widths 1 and 2)
      (a) viscous free stream / stability, total momentum 1e-9, boundedness; single phase bit-identical to note 113;
          local budgets   (b) kappa_L = mu_L dt / (rho_L h^2) from 0.3 to 20 for a moving round drop at a fixed
          CFL (dt not shrunk with h^2)   (c) no explicit viscous term in the band's cells   (d) rising bubble
          (Hysing's case 2) does not blow up
  E1  the interfacial force at the carrying velocity's time ((phi(f^n) + phi(f^{n+1})) / 2), and the extrapolation
      factor capped ("cap") or skipped after a shortened step ("skip")
      (a) capillary damping error not above the base scheme's (N = 32, 64; the exact linear reference of the audit)
      (b) variable steps: two nearly coincident events (factor ~1e4) do not make the error jump; single phase
          bit-identical to note 113 without events   (c) free stream / conservation   (d) a gravity wave does not
          show the same extra damping (one configuration)
  D2  the threshold theta = 3: condition 6 of note 122 (within 1.2 x of D1 alone, N = 32-256) and the aligned free
      stream.

    python -m cases.r0_trials          # docs/figures/r0_trials.json (about 40 min, 20 processes)
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import numpy as np

MU = (1e-2, 1e-2 / 55.0)
EXACT_DAMPING = 0.7146625675736967          # audit C-2, audit/pr177/capwave_exact.py (docs/figures/capwave_exact_audit.json)


@contextlib.contextmanager
def conf(name, record=None):
    """D1 + D2 limited always; then: "base-lib" the library alone; "noE" no E-hooks; "E3band1/2"; "E1" (E1 + E2 +
    band, no E3); "E1mid", "E1mid-cap", "E1mid-skip"; "E1-cap", "E1-skip" (force at the end)."""
    from cases.r0_aligned_free_stream import scheme
    from cases.r0_combined import combined

    if name == "base-lib":
        yield
        return
    kw = {
        "noE": None,
        "E3band1": dict(e3_form="band", e3_band=1), "E3band2": dict(e3_form="band", e3_band=2),
        "E3cn": dict(),
        "E1": dict(e3=False), "E1mid": dict(e3=False, e1_force="mid"),
        "E1mid-cap": dict(e3=False, e1_force="mid", e1_coef="cap"),
        "E1mid-skip": dict(e3=False, e1_force="mid", e1_coef="skip"),
        "E1-cap": dict(e3=False, e1_coef="cap"), "E1-skip": dict(e3=False, e1_coef="skip"),
        "noE1": dict(e1=False, e3=False),
    }[name]
    with scheme(True, "limited"), (combined(record=record, **kw) if kw is not None else contextlib.nullcontext()):
        yield


# --- E3 ------------------------------------------------------------------------------------------------------------
def kappa_drop(cname, kappa, n=32, steps=600, ratio=1e3):
    """A round drop carried by (1, 0.37) with mu_L set so that kappa_L = mu_L dt / (rho_L h^2) at the CFL step
    (dt = h / (2 |U|)); mu_H = 55 mu_L."""
    from cases.r0_aligned_free_stream import free_stream

    rhoL = 1.0 / ratio
    h = 1.0 / n
    U = (1.0, 0.37)
    muL = kappa * rhoL * h * math.hypot(*U) / 0.5
    rec = {}
    with conf(cname, rec):
        r = free_stream("drop", "base", ratio, U=U, n=n, steps=steps, mu=(55.0 * muL, muL)) if cname == "base-lib" \
            else _fs_inner("drop", ratio, U, n, steps, (55.0 * muL, muL))
    r.update({"kappa": kappa, "conf": cname, "record": rec})
    return r


def _fs_inner(name, ratio, U, n, steps, mu, dt=None):
    """free_stream with the D scheme already entered by ``conf`` (no second scheme)."""
    from cases import r0_aligned_free_stream as m

    return m.free_stream.__wrapped__(name, "d1+d2lim", ratio, U=U, n=n, steps=steps, mu=mu, dt=dt)


def bubble(cname, case=2, n=32, T=3.0):
    """Hysing's rising bubble (the audit's configuration of PR #177): [0, 1] x [0, 2], slip side walls, no-slip
    top and bottom, R = 0.25 at (0.5, 0.5), reduced gravity 0.98, surface tension. Rise velocity and centroid."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Dirichlet
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    p = {1: dict(rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0, sigma=24.5),
         2: dict(rho1=1000.0, rho2=1.0, mu1=10.0, mu2=0.1, sigma=1.96)}[case]
    rec = {}
    with conf(cname, rec):
        grid = CartesianGrid((n, 2 * n), length=(1.0, 2.0))
        walls = {w: 0.0 for w in ("left", "right", "bottom", "top")}
        ns = CenteredNS(grid, uf_normal=walls, tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)},
                        tolerance=1e-6, mu=p["mu1"])
        f0 = to_numpy(fraction(grid, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.25 ** 2, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions()), rho1=p["rho1"], rho2=p["rho2"],
                      mu1=p["mu1"], mu2=p["mu2"], conserving=True)
        tp.add_surface_tension(p["sigma"])
        tp.add_reduced_gravity(G=(0.0, -0.98), Z=(0.5, 0.5))
        y = (np.arange(2 * n) + 0.5) / n
        ns.DT = 0.5 / n
        ns.start(events=[T])
        vmax, failed, yc = 0.0, None, None
        while ns.clock.t < T - 1e-12:
            try:
                ns.step()
            except Exception as exc:  # noqa: BLE001
                failed = repr(exc)[:100]
                break
            f = to_numpy(tp.f.interior)
            b = 1.0 - np.clip(f, 0, 1)
            u = [to_numpy(c.interior) for c in ns.u.components]
            vmax = max(vmax, float((b * u[1]).sum() / b.sum()))
            yc = float((b * y[None, :]).sum() / b.sum())
            um = max(float(np.abs(c).max()) for c in u)
            if not math.isfinite(um) or um > 20 or ns.clock.i > 40000:
                failed = "blow-up/step limit t=%.3f |u|=%.3g" % (ns.clock.t, um)
                break
    return {"case": case, "conf": cname, "n": n, "t": float(ns.clock.t), "steps": ns.clock.i, "failed": failed,
            "vrise_max": vmax, "yc": yc, "record": rec}


# --- E1 ------------------------------------------------------------------------------------------------------------
def capwave(cname, n=32):
    from cases.r0_combined_tension import capillary_wave

    rec = {}
    with conf(cname, rec):
        r = capillary_wave("base", n)
    r["conf"] = cname
    r["damping_error"] = (r.get("damping") - EXACT_DAMPING) / EXACT_DAMPING if r.get("damping") else None
    r["record"] = rec
    return r


def gravity_wave(cname, n=32, t_end=2.4, a0=0.01, g=10.0):
    """The capillary wave's configuration without surface tension and with reduced gravity g (one configuration):
    the damping (no exact reference; compared between the configurations)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Neumann, Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA, position
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    rec = {}
    with conf(cname, rec):
        grid = CartesianGrid((n, n), origin=(0.0, -0.5))
        ns = CenteredNS(grid, periodic=(0,), tolerance=1e-10, mu=MU[0])
        bc = BoundaryConditions()
        for side in (0, 1):
            bc.set((0, side), Periodic())
            bc.set((1, side), Neumann(0.0))
        f0 = to_numpy(fraction(grid, lambda x, y: a0 * np.cos(2 * math.pi * x) - y, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=1.0, rho2=1e-3, mu1=MU[0], mu2=MU[1],
                      conserving=True)
        tp.add_reduced_gravity(G=(0.0, -g), Z=(0.0, 0.0))
        samples = list(np.arange(1, int(t_end / 0.008) + 1) * 0.008)
        ns.DT = 0.5 / n
        ns.start(events=samples)
        amp = []
        for te in samples:
            while ns.clock.t < te - 1e-12:
                ns.step()
            p = to_numpy(position(tp.f.interior, tp.f.bc, grid, (0.0, 1.0), (0.0, 0.0)))[0, :]
            ok = p < NODATA / 2
            amp.append(float(p[ok].mean()) if ok.any() else float("nan"))
    t, a = np.array(samples), np.array(amp)
    ext = [k for k in range(1, len(a) - 1) if abs(a[k]) >= abs(a[k - 1]) and abs(a[k]) >= abs(a[k + 1])]
    zc = [t[k] - a[k] * (t[k + 1] - t[k]) / (a[k + 1] - a[k]) for k in range(len(a) - 1) if a[k] * a[k + 1] < 0]
    return {"conf": cname, "n": n, "period": float(2 * np.mean(np.diff(zc))) if len(zc) > 2 else None,
            "damping": float(-np.polyfit(t[ext], np.log(np.abs(a[ext])), 1)[0]) if len(ext) > 2 else None,
            "record": rec}


def events(cname, case="tg", gap=None, n=64):
    """Audit C-5's configuration (#169): events at 0.1, 0.1 + gap, 0.25; the largest error over the run and before
    t = 0.1. "tg": single-phase Taylor-Green + (1, 0.37) (E1 + E2 + E3 as in note 113 for "E*" configurations);
    "drop": a drop in a uniform flow, density ratio 1000."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    rec = {}
    U = (1.0, 0.37)
    with conf(cname, rec):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        if case == "drop":
            f0 = to_numpy(fraction(grid, lambda x_, y_: 0.15 ** 2 - (x_ - 0.5) ** 2 - (y_ - 0.5) ** 2, refine=8))
            rho2 = 1e-3
        else:
            f0 = np.zeros((n, n))
            rho2 = 1.0
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=rho2,
                 conserving=True)
        s = math.sin(math.pi / n) / (math.pi / n)

        def exact(t):
            if case == "drop":
                return U[0] + 0 * X, U[1] + 0 * X
            return (U[0] + s * s * np.sin(2 * np.pi * (X - U[0] * t)) * np.cos(2 * np.pi * (Y - U[1] * t)),
                    U[1] - s * s * np.cos(2 * np.pi * (X - U[0] * t)) * np.sin(2 * np.pi * (Y - U[1] * t)))

        ns.u[0].interior, ns.u[1].interior = exact(0.0)
        ev = [0.1, 0.25] if gap is None else [0.1, 0.1 + gap, 0.25]
        ns.start(events=ev)
        before, worst = 0.0, 0.0
        while ns.clock.t < 0.25 - 1e-12:
            ns.step()
            ue, ve = exact(ns.clock.t)
            e = float(np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve).max())
            if ns.clock.t <= 0.1 + 1e-12:
                before = max(before, e)
            worst = max(worst, e)
            if not math.isfinite(e) or e > 100:
                break
    return {"conf": cname, "case": case, "gap": gap, "err_before": before, "err_max": worst,
            "coef_max": rec.get("coef_max")}


def identity(cname, flows=("tg", "asym", "mix"), grids=(32, 64)):
    """Single phase: the configuration (with E3 for "E3band*", else E1 + E2 + CN as note 113) on note 113's "cons" is
    bit-identical to note 113's corrected "cons/ab+visc+cn" (no events)."""
    import tempfile

    from cases.r0_combined import combined
    from cases.r0p1_unsteady_viscous import CONFIGS, NU, run

    kw = {"E3band1": dict(e3_form="band", e3_band=1), "E3band2": dict(e3_form="band", e3_band=2),
          "E1mid": dict(e1_force="mid"), "E1mid-cap": dict(e1_force="mid", e1_coef="cap"),
          "E1mid-skip": dict(e1_force="mid", e1_coef="skip")}[cname]
    out = []
    with tempfile.TemporaryDirectory() as d:
        for flow in flows:
            for n in grids:
                run(n, 0.5, flow=flow, nu=NU[flow], save=d + "/a.npy", t_ends=(0.25,), **CONFIGS["cons/ab+visc+cn"])
                with combined(**kw):
                    run(n, 0.5, flow=flow, nu=NU[flow], save=d + "/b.npy", t_ends=(0.25,), **CONFIGS["cons"])
                out.append({"flow": flow, "n": n, "difference":
                            float(np.abs(np.load(d + "/a.npy") - np.load(d + "/b.npy")).max())})
    return {"conf": cname, "rows": out}


# --- the driver ---------------------------------------------------------------------------------------------------
def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, kw = a
    from cases import r0_aligned_free_stream as m
    from cases.r0_combined import shear_inviscid
    from cases.r0_d2_forms import vortex_budget

    m.FLOOR[0] = 0
    if kind == "kappa":
        r = kappa_drop(**kw)
    elif kind == "bubble":
        r = bubble(**kw)
    elif kind == "capwave":
        r = capwave(**kw)
    elif kind == "gravity":
        r = gravity_wave(**kw)
    elif kind == "events":
        r = events(**kw)
    elif kind == "identity":
        r = identity(**kw)
    elif kind == "free":
        c = kw.pop("conf")
        rec = {}
        with conf(c, rec):
            r = _fs_inner(kw["name"], kw["ratio"], kw.get("U", (1.0, 0.37)), 32, kw.get("steps", 600),
                          kw.get("mu", (0.0, 0.0)), kw.get("dt"))
        r["conf"], r["record"] = c, rec
        kw["conf"] = c
    elif kind == "vortex":
        c = kw.pop("conf")
        rec = {}
        with conf(c, rec):
            r = vortex_budget.__wrapped__("d1+d2lim", **kw)
        r["conf"], r["record"] = c, rec
        kw["conf"] = c
    elif kind == "theta":
        m.THETA[0] = kw.pop("theta")
        r = shear_inviscid(**kw) if "N" in kw else None
        if r is None:
            r = m.free_stream(**kw)
        r["theta"] = m.THETA[0]
    else:
        raise ValueError(kind)
    r["kind"], r["floor_hits"] = kind, m.FLOOR[0]
    return r


def jobs():
    J = []
    # E3
    for c in ("E3band1", "E3band2"):
        J.append(("identity", dict(cname=c)))
        for name in ("drop", "column", "band2"):
            J.append(("free", dict(conf=c, name=name, ratio=1e3, mu=MU)))
        J.append(("free", dict(conf=c, name="drop", ratio=1e6, mu=MU)))
        for n in (32, 64):
            J.append(("vortex", dict(conf=c, n=n, mu=MU)))
        for k in (0.3, 1.0, 3.0, 10.0, 20.0):
            J.append(("kappa", dict(cname=c, kappa=k)))
        J.append(("bubble", dict(cname=c, case=2, n=32)))
    for c in ("base-lib", "E3cn"):
        for k in (0.3, 1.0, 3.0, 10.0, 20.0):
            J.append(("kappa", dict(cname=c, kappa=k)))
    J.append(("bubble", dict(cname="base-lib", case=2, n=32)))
    # E1
    for c in ("base-lib", "noE1", "E1", "E1mid", "E1mid-cap", "E1mid-skip"):
        for n in (32, 64):
            J.append(("capwave", dict(cname=c, n=n)))
        J.append(("gravity", dict(cname=c, n=32)))
    for c in ("base-lib", "E1", "E1-cap", "E1-skip", "E1mid-cap", "E1mid-skip"):
        for case in ("tg", "drop"):
            for gap in (None, 1e-3, 1e-5, 1e-7):
                J.append(("events", dict(cname=c, case=case, gap=gap)))
    for c in ("E1mid-cap", "E1mid-skip"):
        J.append(("identity", dict(cname=c)))
        for name in ("column", "square", "drop"):
            J.append(("free", dict(conf=c, name=name, ratio=1e3)))
        J.append(("free", dict(conf=c, name="column", ratio=1e3, dt=0.4999 / 32)))
        J.append(("vortex", dict(conf=c, n=32)))
        J.append(("vortex", dict(conf=c, n=32, mu=MU)))
    # D2 theta = 3
    for N in (32, 48, 64, 96, 128, 192, 256):
        for c in ("d1", "d1+d2lim"):
            J.append(("theta", dict(theta=3.0, conf=c, N=N)))
    for name, kw in (("column", {}), ("square", {}), ("column", {"dt": 0.4999 / 32}), ("square", {"dt": 0.4999 / 32})):
        for r in (1e3, 1e6):
            J.append(("theta", dict(theta=3.0, name=name, sch="d1+d2lim", ratio=r, **kw)))
    return J


def _key(a):
    return json.dumps([a[0], {k: v for k, v in a[1].items()}], sort_keys=True, default=str)


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_trials.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    J = [a for a in jobs() if _key(a) not in done]
    cost = {"bubble": 9, "identity": 8, "kappa": 3, "capwave": 4, "gravity": 4, "vortex": 2}
    J.sort(key=lambda a: -cost.get(a[0], 1))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "20")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job, [(k, dict(v)) for k, v in J], chunksize=1)):
            r["job"] = _key(a)
            with open(part, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            done[r["job"]] = r
    with open(out, "w") as fh:
        json.dump(list(done.values()), fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()


# --- #166's item 3: the configuration without E1 and E3 (D1 + D2 limited, E2, the band-switched source) -----------
def item3_jobs():
    J = []
    U0 = (1.0, 0.37)
    for name in ("column", "square", "square-bubble"):
        for r in (1e3, 1e6):
            for U in ((1.0, 0.0), U0):
                J.append(("free", dict(conf="noE1", name=name, ratio=r, U=U)))
        for r in (1e3, 1e6):
            J.append(("free", dict(conf="noE1", name=name, ratio=r, dt=0.4999 / 32)))
    for name in ("drop", "bubble", "small", "film", "slant", "two", "band2"):
        for r in (1e3, 1e6):
            J.append(("free", dict(conf="noE1", name=name, ratio=r, U=U0, steps=2000)))
    for name in ("column", "drop", "band2"):
        J.append(("free", dict(conf="noE1", name=name, ratio=1e3, U=U0, mu=MU)))
    for n in (32, 64):
        J.append(("vortex", dict(conf="noE1", n=n)))
        J.append(("vortex", dict(conf="noE1", n=n, mu=MU)))
    GEOM = ([(th, 0.0137, n) for th in (0.0, 10.0) for n in (32, 64, 128)]
            + [(th, off, n) for th in (26.565, 45.0) for off in (0.0137, 0.1171, 0.2313) for n in (32, 64, 128)])
    FAIL = [(45.0, 0.0137, 32), (45.0, 0.0137, 64), (26.565, 0.2313, 32)]
    for th, off, n in GEOM:
        if (th, off, n) not in FAIL:
            for kind in ("smooth", "mean"):
                J.append(("hydro", dict(kind=kind, theta_deg=th, gravity="reduced", n=n, offset=off)))
    for n in (32, 64):
        for cv in ("exact", "numerical"):
            J.append(("drop_st", dict(conf="noE1E3", n=n, curvature=cv)))
            J.append(("drop_st", dict(conf="base", n=n, curvature=cv)))
    J.append(("budget", {}))
    return J


def _job3(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    kind, kw = a
    from cases import r0_aligned_free_stream as m

    m.FLOOR[0] = 0
    if kind in ("free", "vortex"):
        return _job(a)
    if kind == "hydro":
        from cases.r0_combined import configuration
        from cases.r0p2_pressure_momentum import hydrostatic

        with configuration("noE1E3"):
            r = hydrostatic(**kw)
    elif kind == "drop_st":
        from cases.r0_combined_tension import static_drop

        r = static_drop(**kw)
    elif kind == "budget":
        from cases.r0_combined import configuration
        from cases.r0_local_budget import _flow, capture, summarise
        from cases.r0_aligned_free_stream import shape

        r = {}
        for name in ("drop", "column", "half"):
            import contextlib as _c

            with _c.ExitStack():
                cap = _flow_noE(lambda g, name=name: shape(name, g))
            r[name] = summarise(cap, scale_q=2.5, lin=True)
    r["kind"], r["floor_hits"] = kind, m.FLOOR[0]
    return r


def _flow_noE(f0, steps=6, n=32):
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    from cases.r0_combined import configuration
    from cases.r0_local_budget import capture

    with configuration("noE1E3"), capture() as cap:
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        TwoPhase(ns, CellField.from_interior(grid, f0(grid), BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3,
                 conserving=True)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        ns.start()
        for _ in range(steps):
            ns.step()
    return cap


def main3() -> None:
    """docs/figures/r0_trials_item3.json."""
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_trials_item3.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    J = [a for a in item3_jobs() if _key(a) not in done]
    cost = {"hydro": 9, "drop_st": 5, "vortex": 2}
    J.sort(key=lambda a: -cost.get(a[0], 1))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "20")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job3, [(k, dict(v)) for k, v in J], chunksize=1)):
            r["job"] = _key(a)
            with open(part, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            done[r["job"]] = r
    with open(out, "w") as fh:
        json.dump(list(done.values()), fh, indent=1, default=float)
    print("wrote", out)


def plot() -> None:
    """docs/figures/r0_trials.png: (a) E3's band form against kappa_L; (b) the D2 threshold 3 against 1 (condition 6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(os.path.join("docs", "figures", "r0_trials.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, mk in (("base-lib", "o"), ("E3cn", "x"), ("E3band1", "s"), ("E3band2", "^")):
        rows = sorted((r for r in R if r["kind"] == "kappa" and r["conf"] == c), key=lambda r: r["kappa"])
        ax[0].loglog([r["kappa"] for r in rows], [max(r["max_err"], 1e-16) for r in rows], marker=mk, label=c)
    ax[0].set_xlabel("kappa_L = mu_L dt / (rho_L h^2)")
    ax[0].set_ylabel("max |u - U| (moving drop, 600 steps)")
    ax[0].set_title("(a) E3: band form grows at kappa_L >= 10")
    ax[0].legend()
    th = {(r["conf"], r["N"]): r for r in R if r["kind"] == "theta" and "N" in r}
    c6 = json.load(open(os.path.join("docs", "figures", "r0_combined_c6.json")))
    one = {(r["conf"], r["N"]): r for r in c6 if r["cfl"] == 0.5}
    NS = sorted({n for (_, n) in th})
    for m, mk in (("L1", "o"), ("L2", "s")):
        ax[1].semilogx(NS, [th[("d1+d2lim", n)][m] / th[("d1", n)][m] for n in NS], "-", marker=mk, label="%s, theta 3" % m)
        ax[1].semilogx(NS, [one[("d1+d2lim", n)][m] / one[("d1", n)][m] for n in NS], "--", marker=mk,
                       label="%s, theta 1" % m)
    ax[1].axhline(1.2, color="tab:red", lw=1)
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("error ratio D1 + D2 limited / D1")
    ax[1].set_title("(b) condition 6 with the threshold 3")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0_trials.png"), dpi=130)

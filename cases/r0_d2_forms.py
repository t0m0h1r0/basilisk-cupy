"""docs/notes/119 (A-1 of #145, audit #162): four forms of the D1 / D2 fixes of docs/notes/118, compared at a minimal
scale with B's pass conditions fixed before the measurement (#162, B's comment A-1):

  "d1"        D1 only (control: grows at the CFL bound)
  "d1+d2"     D1 + D2 everywhere: no slope in any donor cell the phase fills in part (control, docs/notes/118)
  "d1+d2lim"  D1 + D2 limited: the slope of a face is dropped only where (D2)'s gain would exceed 1, i.e. where
              rho_k out (1 - C) / 2 > rho_k (c_k - out) + rho_o (1 - c_k): the phase's leaving volume out = cf_k |C|,
              the mass the donor keeps of its own content (inflow not counted). Never at the density ratio 1.
  "d1+geo"    D1 + the centroid form of the slope terms (docs/notes/77, VOFTracer(geometric=True)): the slope is
              taken between the centroid of the phase's part of the swept strip and that of the phase in the cell, so
              it vanishes when all of the phase leaves.

  a  free stream / stability   docs/notes/118's aligned band and square (1e3, 1e6, CFL bound and fixed 0.4999) and
                               the shapes of the audit's break_stability.py: max |u - U| at the pressure tolerance
  b  accuracy                  the audit's accuracy2.py (density ratio 1, passive drop, Taylor-Green + uniform flow,
                               exact solution, N = 32 ... 256) on the second-order base (face-value source + E1) and on
                               the library's base: max, L2, L1 within 1.2 x the base scheme's, on every grid
  c  conservation              total momentum with the pressure 1e-9 and the phase mass (drop in a vortex, ratio
                               1000, filtered=False)
  d  no interface              bit-identical to the base scheme

Recorded only: D1's checks (volume-fraction overshoot and its dependence on the pressure tolerance, the margin of
rho_lin's floor, rho_lin against the clamped density of the properties), the share of faces where "d1+d2lim" cuts.

    python -m cases.r0_d2_forms          # docs/figures/r0_d2_forms.json (about 20 min, NP=22 processes)
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from cases.r0_aligned_free_stream import SCHEMES, free_stream, identical, in_scheme, scheme  # noqa: F401

FORMS = ("base", "d1", "d1+d2", "d1+d2lim", "d1+geo")
P2 = 2 * math.pi
U0 = (1.0, 0.37)
_G, _W = np.polynomial.legendre.leggauss(4)
_G, _W = (_G + 1) / 2, _W / 2


def exact(n, t):
    """Cell averages (4 x 4 Gauss) of the Taylor-Green flow carried by U0 (the audit's accuracy.py)."""
    i = np.arange(n) / n
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for a, wa in zip(_G, _W):
        for b, wb in zip(_G, _W):
            X, Y = np.meshgrid(i + a / n - U0[0] * t, i + b / n - U0[1] * t, indexing="ij")
            u += wa * wb * np.sin(P2 * X) * np.cos(P2 * Y)
            v -= wa * wb * np.cos(P2 * X) * np.sin(P2 * Y)
    return U0[0] + u, U0[1] + v


@in_scheme
def accuracy(sch, n, second_order=True, drop=True, T=0.25):
    """b: the audit's accuracy2.py (second_order: the face-value source of docs/notes/112 and the carrying face
    velocity extrapolated to t + dt/2, E1 of docs/notes/113) or accuracy.py (the library's scheme)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    f0 = (to_numpy(fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.4) ** 2, refine=8)) if drop
          else np.zeros((n, n)))
    TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0,
             conserving=True, conserving_source=second_order)
    if second_order:
        st = {}

        def before(s):
            dt = s.clock.dt
            cur = [c.copy() for c in s.uf.components]
            if "uf" in st:
                r = 0.5 * dt / st["dt"]
                for c, x, y in zip(s.uf.components, cur, st["uf"]):
                    c[...] = x + r * (x - y)
            st["uf"], st["dt"] = cur, dt

        ns.on("vof", before)
    u0, v0 = exact(n, 0.0)
    ns.u[0].interior = u0
    ns.u[1].interior = v0
    ns.start(events=[T])
    while ns.clock.t < T - 1e-12:
        ns.step()
    ue, ve = exact(n, T)
    e = np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve)
    return {"scheme": sch, "n": n, "second_order": second_order, "drop": drop,
            "max": float(e.max()), "L2": float(np.sqrt((e * e).mean())), "L1": float(e.mean())}


@in_scheme
def vortex_budget(sch, n=64, ratio=1000.0, t_end=0.5, cfl=0.5, mu=(0.0, 0.0)):
    """c: docs/notes/114's drop in the Taylor-Green vortex + (0.5, 0.2), filtered=False: the total momentum (the
    transported density, i.e. rho_lin for the D1 forms, and the clamped one), the phase mass sum f, the overshoot of
    f, the margin of rho_lin's floor, and max |rho_lin - rho_clamped| / rho2 (the D1 checks)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=(max(mu) if max(mu) > 0 else None))
    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio,
                  mu1=mu[0], mu2=mu[1], conserving=True)
    ns.cfl = cfl
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = 0.5 + np.sin(P2 * X) * np.cos(P2 * Y)
    ns.u[1].interior = 0.2 - np.cos(P2 * X) * np.sin(P2 * Y)
    lin = SCHEMES[sch][0] in (True, "lin")

    def mom():
        f = to_numpy(tp.f.interior)
        rho = f * (tp.rho1 - tp.rho2) + tp.rho2 if lin else np.clip(f, 0, 1) * (tp.rho1 - tp.rho2) + tp.rho2
        return np.array([(rho * to_numpy(ns.u[k].interior)).sum() / n ** 2 for k in range(2)])

    m0, v0 = mom(), float(to_numpy(tp.f.interior).sum())
    worst = over = dens = 0.0
    floor_margin = math.inf
    failed = None
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()
        f = to_numpy(tp.f.interior)
        worst = max(worst, float(np.hypot(*(mom() - m0)) / np.hypot(*m0)))
        over = max(over, -float(f.min()), float(f.max()) - 1.0)
        rl = f * (tp.rho1 - tp.rho2) + tp.rho2
        floor_margin = min(floor_margin, float(rl.min()) / (0.5 * min(tp.rho1, tp.rho2)))
        dens = max(dens, float(np.abs(rl - (np.clip(f, 0, 1) * (tp.rho1 - tp.rho2) + tp.rho2)).max()) / tp.rho2)
        if not math.isfinite(worst) or float(np.abs(to_numpy(ns.u[0].interior)).max()) > 1e3:
            failed = "blow-up at t = %.4f" % ns.clock.t
            break
    return {"scheme": sch, "n": n, "ratio": ratio, "steps": ns.clock.i, "failed": failed, "momentum_change": worst,
            "mass_change": abs(float(to_numpy(tp.f.interior).sum()) - v0) / v0, "f_overshoot": over,
            "rho_floor_margin": floor_margin, "rho_lin_vs_clamped_over_rho2": dens}


def cut_share():
    """Recorded: how often "d1+d2lim" drops a slope (counted inside the patched sweep): faces with a dropped slope
    over faces with a partial donor (0 < c_k < 1, either phase), in docs/notes/114's vortex (ratio 1000, N = 64,
    t = 0.25) and in the aligned band carried at the CFL bound (ratio 1000, 600 steps)."""
    from cases import r0_aligned_free_stream as m

    out = {}
    for name, run in (("vortex", lambda: _vortex_plain(64, 1000.0, 0.25)),
                      ("aligned band", lambda: m.free_stream("column", "d1+d2lim", 1e3))):
        m.COUNT[0] = m.COUNT[1] = 0
        if name == "vortex":
            with scheme(True, "limited"):
                run()
        else:
            run()
        out[name] = {"dropped": m.COUNT[0], "partial": m.COUNT[1], "share": m.COUNT[0] / max(m.COUNT[1], 1)}
    return out


def _vortex_plain(n, ratio, t_end):
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio,
             conserving=True)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = 0.5 + np.sin(P2 * X) * np.cos(P2 * Y)
    ns.u[1].interior = 0.2 - np.cos(P2 * X) * np.sin(P2 * Y)
    ns.start(events=[t_end])
    while ns.clock.t < t_end - 1e-12:
        ns.step()


def field_differences(directory, ns_=(64, 128, 256), forms=("base", "d1+d2", "d1+d2lim", "d1+geo")):
    """Recorded only (no exact solution at a density ratio): docs/notes/118's drop in the vortex (ratio 1000,
    t = 0.25), each form's velocity against the base's on the same grid, and the base's own grid difference (all
    averaged to N = 64)."""
    import multiprocessing as mp

    from cases.r0_aligned_free_stream import _job as job118

    J = [("vortex", dict(sch=s, n=n, save=os.path.join(directory, "v_%s_%d.npy" % (s, n)))) for s in forms for n in ns_]
    with mp.get_context("spawn").Pool(len(J)) as pool:
        pool.map(job118, J)

    def load(s, n, to=64):
        a = np.load(os.path.join(directory, "v_%s_%d.npy" % (s, n)))[:2]
        r = n // to
        return a.reshape(2, to, r, to, r).mean(axis=(2, 4))

    rms = lambda d: float(np.sqrt((d * d).mean()))  # noqa: E731
    out = {}
    for n in ns_:
        for s in forms[1:]:
            out["%s-base N%d" % (s, n)] = rms(load(s, n) - load("base", n))
    for a, b in zip(ns_[:-1], ns_[1:]):
        out["base N%d-N%d" % (a, b)] = rms(load("base", a) - load("base", b))
    return out


# --- the driver ---------------------------------------------------------------------------------------------------
NS = (32, 48, 64, 96, 128, 192, 256)


def jobs():
    J = []
    for sch in FORMS:
        for so in (True, False):
            for n in NS:
                J.append(("accuracy", dict(sch=sch, n=n, second_order=so)))
        for name, U, r, kw in [("column", (1.0, 0.0), 1e3, {}), ("column", (1.0, 0.37), 1e3, {}),
                               ("column", (1.0, 0.0), 1e6, {}), ("column", (1.0, 0.37), 1e6, {}),
                               ("square", (1.0, 0.0), 1e3, {}), ("square", (1.0, 0.0), 1e6, {}),
                               ("square-bubble", (1.0, 0.0), 1e3, {}),
                               ("column", (1.0, 0.0), 1e3, {"dt": 0.4999 / 32}), ("square", (1.0, 0.0), 1e3, {"dt": 0.4999 / 32}),
                               ("column", (1.0, 0.0), 1e6, {"dt": 0.4999 / 32}), ("square", (1.0, 0.0), 1e6, {"dt": 0.4999 / 32}),
                               ("column", (1.0, 0.0), 1e3, {"cfl": 0.45}), ("column", (1.0, 0.0), 1e3, {"tolerance": 1e-13})]:
            J.append(("free", dict(name=name, sch=sch, ratio=r, U=U, **kw)))
        for name in ("drop", "bubble", "small", "film", "slant", "two"):
            for r in (1e3, 1e6):
                J.append(("free", dict(name=name, sch=sch, ratio=r, U=U0, steps=2000)))
        for n in (32, 64):
            J.append(("vortex", dict(sch=sch, n=n)))
        J.append(("vortex", dict(sch=sch, n=64, ratio=1e6, t_end=0.1)))
    for tol in (1e-10, 1e-13):                                         # D1: the overshoot against the tolerance
        J.append(("free", dict(name="column", sch="d1+d2lim", ratio=1e6, U=U0, tolerance=tol)))
    return J


def _job(a):
    kind, kw = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    f = {"accuracy": accuracy, "free": free_stream, "vortex": vortex_budget}[kind]
    return {"kind": kind, **f(**kw)}


def _key(kind, kw):
    return json.dumps({"kind": kind, **kw}, sort_keys=True)


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_d2_forms.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[d["job"]] = d
    J = [j for j in jobs() if _key(*j) not in done]
    J.sort(key=lambda j: -(j[1].get("n", 32) if j[0] == "accuracy" else 0))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "22")), maxtasksperchild=1) as pool:
        for a, r in zip(J, pool.imap(_job, J, chunksize=1)):
            r["job"] = _key(*a)
            with open(part, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            done[r["job"]] = r
    res = {k: [d for d in done.values() if d["kind"] == k] for k in ("accuracy", "free", "vortex")}
    res["identical"] = {s: bool(np.array_equal(identical(s, steps=20), identical("base", steps=20))) for s in FORMS}
    res["cut_share"] = cut_share()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res["field_differences"] = field_differences(tmp)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()


def plot() -> None:
    """docs/figures/r0_d2_forms.png: (a) condition b, L2 error against N on the second-order base; (b) condition a,
    the largest max |u - U| over the free-stream configurations per form."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(os.path.join("docs", "figures", "r0_d2_forms.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"base": ("k", "o", 9), "d1": ("0.6", "s", 7), "d1+d2": ("tab:red", "^", 6),
             "d1+d2lim": ("tab:blue", "x", 8), "d1+geo": ("tab:green", "v", 6)}
    for s in FORMS:
        rows = sorted((d for d in R["accuracy"] if d["scheme"] == s and d["second_order"]), key=lambda d: d["n"])
        c, m, ms = style[s]
        ax[0].loglog([d["n"] for d in rows], [d["L2"] for d in rows], marker=m, color=c, ms=ms, mfc="none", label=s)
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("L2 error of u (exact solution)")
    ax[0].set_title("(b) accuracy: ratio 1, passive drop, 2nd-order base")
    ax[0].legend()
    worst = [max(d["max_err"] for d in R["free"] if d["scheme"] == s) for s in FORMS]
    ax[1].bar(range(len(FORMS)), worst, color=[style[s][0] for s in FORMS])
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(FORMS)))
    ax[1].set_xticklabels(FORMS)
    ax[1].axhline(1e-8, ls=":", color="0.5")
    ax[1].set_ylabel("largest max |u - U| (25 free-stream cases)")
    ax[1].set_title("(a) free stream: aligned shapes and audit shapes")
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0_d2_forms.png"), dpi=130)

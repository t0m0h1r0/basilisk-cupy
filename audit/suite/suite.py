"""監修の独立の検査一式（#185）。A の cases/ は使わない（ライブラリの公開の入口と、監修の参照解だけ）。合否は判定せず値を記録する。

    PYTHONPATH=<対象のツリー> python audit/suite/suite.py [--config <JSON か file>] [--only aligned,half,...] [--procs 14] [--out f.json]

構成（JSON）: {"twophase": {TwoPhase の引数}, "class": "TwoPhase" | "TwoPhaseVOFJump", "ns": {CenteredNS の属性},
              "tolerance": 1e-10}。名前つき：base（conserving、filtered=False）、r0（D1 + D2 限定 θ = 3 + project_start）。
項目: aligned half mirror events capwave shear hydrostatic poiseuille
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMED = {"base": {"twophase": {"conserving": True, "filtered": False}},
         "r0": {"twophase": {"conserving": True, "filtered": False, "conserving_consistent": True,
                             "conserving_slope_limit": 3.0}, "ns": {"project_start": True}}}


def make(cfg, grid, f0, bc, rho, mu=(0.0, 0.0), ns_kw=None):
    from basilisk_cupy import CellField
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    ns_kw = dict(ns_kw or {})
    ns_kw.setdefault("tolerance", float(cfg.get("tolerance", 1e-10)))
    if mu[0] or mu[1]:
        ns_kw["mu"] = max(mu)
    ns = CenteredNS(grid, **ns_kw)
    for k, v in cfg.get("ns", {}).items():
        if not hasattr(ns, k):
            raise ValueError("unknown CenteredNS attribute " + k)
        setattr(ns, k, v)
    cls = TwoPhase
    if cfg.get("class", "TwoPhase") == "TwoPhaseVOFJump":
        from basilisk_cupy.interface.vof_jump import TwoPhaseVOFJump as cls
    f = CellField.from_interior(grid, f0, bc if bc is not None else ns.p.bc)
    tp = cls(ns, f, rho1=rho[0], rho2=rho[1], mu1=mu[0], mu2=mu[1], **cfg.get("twophase", {}))
    return ns, tp


def periodic_setup(cfg, n, f0, rho, mu=(0.0, 0.0)):
    from basilisk_cupy import BoundaryConditions, CartesianGrid

    return make(cfg, CartesianGrid((n, n)), f0, BoundaryConditions.periodic(2), rho, mu, dict(periodic=(0, 1)))


def centers(n):
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, indexing="ij")


# -- 1. free stream of face-aligned interfaces (#162) ------------------------------------------------------------
def aligned(cfg, a):
    shape, U, cfl, ratio = a
    n = 32
    X, Y = centers(n)
    f0 = (((X > 0.25) & (X < 0.5)) if shape == "band" else ((np.abs(X - 0.5) < 0.125) & (np.abs(Y - 0.5) < 0.125))).astype(float)
    ns, tp = periodic_setup(cfg, n, f0, (1.0, 1.0 / ratio))
    ns.cfl = 0.5
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * X
    if cfl is not None:
        ns.DT = cfl / n / max(abs(U[0]), abs(U[1]))
    ns.start()
    m = 0.0
    for _ in range(600):
        ns.step()
        e = max(float(np.abs(ns.u[k].interior - U[k]).max()) for k in range(2))
        m = max(m, e)
        if not math.isfinite(e) or e > 10:
            break
    return {"shape": shape, "U": list(U), "cfl": cfl or 0.5, "ratio": ratio, "max_err": m}


# -- 2. c = 1/2 with a non-uniform carrying velocity (#156): transport only --------------------------------------
def half(cfg, a):
    kind, ratio, i0 = a
    n = 32
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if kind == "diag":
        k = (i + j) % n
        f0 = np.where((k == 8) | (k == 24), 0.5, np.where((k > 8) & (k < 24), 1.0, 0.0))
    else:
        r = np.random.default_rng(5).random((n, n))
        f0 = np.where(r < 0.4, 0.5, np.where(r < 0.7, 1.0, 0.0))
    ns, tp = periodic_setup(cfg, n, f0, (1.0, 1.0 / ratio))
    P = np.pi
    xv = np.arange(n + 1) / n
    XV, YV = np.meshgrid(xv, xv, indexing="ij")
    psi = (np.sin(2 * P * XV) * np.sin(2 * P * YV) + 0.5 * np.cos(2 * P * XV + 1) * np.sin(2 * P * YV + 0.4)) / (2 * P)
    uf = [(psi[:, 1:] - psi[:, :-1]) * n, -(psi[1:, :] - psi[:-1, :]) * n]
    U = (1.0, 0.37)
    for k in range(2):
        ns.u[k].interior = np.full((n, n), U[k])
    ns.u.fill_ghosts()
    dt = 0.4 / n / max(np.abs(uf[0]).max(), np.abs(uf[1]).max())
    err = 0.0
    for s in range(6):
        for k in range(2):
            ns.uf.components[k][...] = uf[k]
        ns.clock.dt, ns.clock.i = dt, i0 + s
        tp._vof(ns)                                     # the library's transport only
        err = max(err, max(float(np.abs(ns.u[k].interior - U[k]).max()) for k in range(2)))
    return {"kind": kind, "ratio": ratio, "i0": i0, "max_err": err}


# -- 3. slip-wall box = half of the mirrored periodic box (#149, A3) ----------------------------------------------
def mirror(cfg, a):
    mu, heavy = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.interface.fractions import fraction

    n, ratio = 32, 1e3
    phi = lambda x, y: np.maximum(0.2 - np.hypot(x - 0.31, y - 0.12), 0.11 - np.hypot(x - 0.9, y - 0.93))
    f0 = np.asarray(fraction(CartesianGrid((n, n)), phi))
    if not heavy:
        f0 = 1 - f0
    X, Y = centers(n)
    u0 = np.sin(np.pi * X) * np.cos(np.pi * Y)
    v0 = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    cfg = dict(cfg, tolerance=1e-12)
    mus = (mu, mu / 55)
    res = []
    for periodic in (False, True):
        if periodic:
            F = np.block([[f0, f0[:, ::-1]], [f0[::-1, :], f0[::-1, ::-1]]])
            grid = CartesianGrid((2 * n, 2 * n), length=(2.0, 2.0))
            ns, tp = make(cfg, grid, F, BoundaryConditions.periodic(2), (1.0, 1 / ratio), mus, dict(periodic=(0, 1)))
            ns.u[0].interior = np.block([[u0, u0[:, ::-1]], [-u0[::-1, :], -u0[::-1, ::-1]]])
            ns.u[1].interior = np.block([[v0, -v0[:, ::-1]], [v0[::-1, :], -v0[::-1, ::-1]]])
        else:
            walls = ("left", "right", "bottom", "top")
            ns, tp = make(cfg, CartesianGrid((n, n)), f0, BoundaryConditions(), (1.0, 1 / ratio), mus,
                          dict(uf_normal={w: 0.0 for w in walls}))
            ns.u[0].interior = u0
            ns.u[1].interior = v0
        ns.DT = 0.2 / n
        ns.start()
        for _ in range(60):
            ns.step()
        res.append([ns.u[0].interior[:n, :n].copy(), ns.u[1].interior[:n, :n].copy(), tp.f.interior[:n, :n].copy()])
    return {"mu": mu, "heavy": heavy, "max_diff": float(max(np.abs(p - q).max() for p, q in zip(*res)))}


# -- 4. nearly coincident events (#166) -------------------------------------------------------------------------
def events(cfg, a):
    mu, = a
    n = 64
    ns, tp = periodic_setup(cfg, n, np.zeros((n, n)), (1.0, 1.0), (mu, mu))
    X, Y = centers(n)
    U = (1.0, 0.37)
    s = math.sin(math.pi / n) / (math.pi / n)
    k2 = 8 * math.pi ** 2

    def exact(t):
        d = math.exp(-mu * k2 * t)
        return (U[0] + d * s * s * np.sin(2 * np.pi * (X - U[0] * t)) * np.cos(2 * np.pi * (Y - U[1] * t)),
                U[1] - d * s * s * np.cos(2 * np.pi * (X - U[0] * t)) * np.sin(2 * np.pi * (Y - U[1] * t)))
    out = {"mu": mu}
    for name, ev in (("none", [0.1, 0.25]), ("gap1e-7", [0.1, 0.1 + 1e-7, 0.25])):
        ns, tp = periodic_setup(cfg, n, np.zeros((n, n)), (1.0, 1.0), (mu, mu))
        ns.u[0].interior, ns.u[1].interior = exact(0.0)
        ns.start(events=ev)
        m = 0.0
        while ns.clock.t < 0.25 - 1e-12:
            ns.step()
            ue, ve = exact(ns.clock.t)
            m = max(m, float(np.hypot(ns.u[0].interior - ue, ns.u[1].interior - ve).max()))
        out[name] = m
    out["rel_change"] = out["gap1e-7"] / out["none"] - 1.0
    return out


# -- 5. capillary wave against the exact linear solution (t <= 1.2) ----------------------------------------------
EXACT = {"period": 0.40257, "damping": 0.71466}          # audit/pr177/capwave_exact.py (valid for t <= 2)


def capwave(cfg, a):
    n, = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.boundary import Neumann, Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA, position

    grid = CartesianGrid((n, n), origin=(0.0, -0.5))
    bc = BoundaryConditions()
    for side in (0, 1):
        bc.set((0, side), Periodic())
        bc.set((1, side), Neumann(0.0))
    f0 = np.asarray(fraction(grid, lambda x, y: 0.01 * np.cos(2 * math.pi * x) - y, refine=8))
    ns, tp = make(cfg, grid, f0, bc, (1.0, 1e-3), (1e-2, 1e-2 / 55.0), dict(periodic=(0,)))
    tp.add_surface_tension(1.0)
    samples = list(np.arange(1, int(1.2 / 0.004) + 1) * 0.004)
    ns.start(events=samples)
    y = []
    for te in samples:
        while ns.clock.t < te - 1e-12:
            ns.step()
        p = np.asarray(position(tp.f.interior, tp.f.bc, grid, (0.0, 1.0), (0.0, 0.0)))
        col = p[0, :]
        y.append(float(col[col < NODATA / 2].mean()))
    t, y = np.array(samples), np.array(y)
    zc = [t[k] - y[k] * (t[k + 1] - t[k]) / (y[k + 1] - y[k]) for k in range(len(y) - 1) if y[k] * y[k + 1] < 0]
    ext = [k for k in range(1, len(y) - 1) if abs(y[k]) >= abs(y[k - 1]) and abs(y[k]) >= abs(y[k + 1])]
    period = float(2 * np.mean(np.diff(zc)))
    damping = float(-np.polyfit(t[ext], np.log(np.abs(y[ext])), 1)[0])
    return {"n": n, "period": period, "damping": damping, "period_err": period / EXACT["period"] - 1,
            "damping_err": damping / EXACT["damping"] - 1}


# -- 6. oblique two-layer shear + boost against the audit's spectral-element reference ---------------------------
def shear(cfg, a):
    nv, n = a
    sys.path.insert(0, HERE)
    from oblique_shear_reference import Shear

    ref = Shear(n=nv)
    c = ref.coefficients(modes={1: 1.0, 2: 0.3})
    T = 1.0 / abs(nv[0] * 1.0 + nv[1] * 0.37)
    cfg = dict(cfg, ns=dict(cfg.get("ns", {}), project_start=True))      # the initial face velocity: divergence-free
    f0 = ref.fraction_cells(n, 0.0)
    ns, tp = periodic_setup(cfg, n, f0, (1.0, 1e-3), (1e-2, 1e-2 / 55.0))
    ns.u[0].interior, ns.u[1].interior = ref.velocity_cells(n, c, 0.0)
    ns.start(events=[T])

    def momentum():
        f = tp.f.interior
        r = (f if cfg.get("twophase", {}).get("conserving_consistent") else np.clip(f, 0, 1)) * (1 - 1e-3) + 1e-3
        return np.array([(r * ns.u[0].interior).sum(), (r * ns.u[1].interior).sum()]) / n ** 2
    p0, m0 = momentum(), float(tp.f.interior.sum())
    while ns.clock.t < T - 1e-12:
        ns.step()
    ue, ve = ref.velocity_cells(n, c, T)
    th = np.array([-nv[1], nv[0]]) / math.hypot(*nv)
    nu = np.array([nv[0], nv[1]]) / math.hypot(*nv)
    du, dv = ns.u[0].interior - ue, ns.u[1].interior - ve
    et, en = du * th[0] + dv * th[1], du * nu[0] + dv * nu[1]
    return {"n_vec": list(nv), "N": n, "L2_tangential": float(np.sqrt((et ** 2).mean())), "L2_normal": float(np.sqrt((en ** 2).mean())),
            "momentum_change": float(np.abs(momentum() - p0).max() / max(np.abs(p0).max(), 1e-30)),
            "mass_change": abs(float(tp.f.interior.sum()) - m0) / m0}


# -- 7. hydrostatic planes: the reduced potential at t = 0 over many angles / positions, exact fractions ---------
def _clip_area(x0, y0, h, nx, ny, d):
    poly = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
    out = []
    for k in range(4):
        a, b = poly[k], poly[(k + 1) % 4]
        sa, sb = nx * a[0] + ny * a[1] - d, nx * b[0] + ny * b[1] - d
        if sa <= 0:
            out.append(a)
        if sa * sb < 0:
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    if len(out) < 3:
        return 0.0
    s = sum((out[k][0] - x0) * (out[(k + 1) % len(out)][1] - y0) - (out[(k + 1) % len(out)][0] - x0) * (out[k][1] - y0)
            for k in range(len(out)))
    return abs(s) / 2 / (h * h)


def hydrostatic(cfg, a):
    th, off, n = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.heights import NODATA, position

    t = math.radians(th)
    nx, ny = math.sin(t), math.cos(t)
    h = 1.0 / n
    d = nx * 0.5 + ny * (0.5 + off)
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lo = nx * i * h + ny * j * h
            if lo >= d:
                continue
            c[i, j] = 1.0 if lo + (nx + ny) * h <= d else _clip_area(i * h, j * h, h, nx, ny, d)
    grid = CartesianGrid((n, n))
    f = CellField.from_interior(grid, c, BoundaryConditions())
    p = np.asarray(position(f.interior, f.bc, grid, (-nx, -ny), (0.5, 0.5 + off)))
    ok = p < NODATA / 2
    return {"theta": th, "off": off, "n": n, "max_phi": float(np.abs(np.where(ok, p, 0.0)).max())}


# -- 8. two-layer Poiseuille with the interface on a face (#198) ------------------------------------------------
def poiseuille(cfg, a):
    off, = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.boundary import Dirichlet, Neumann, Periodic

    n = 32
    h = 1.0 / n
    grid = CartesianGrid((n, n))
    bc = BoundaryConditions()
    for side in (0, 1):
        bc.set((0, side), Periodic())
        bc.set((1, side), Neumann(0.0))
    f0 = np.tile(np.clip((0.5 + off * h - np.arange(n) * h) / h, 0, 1), (n, 1))
    ns, tp = make(cfg, grid, f0, bc, (1.0, 1e-3), (1.0, 1.0 / 55.0),
                  dict(periodic=(0,), uf_normal={w: 0.0 for w in ("bottom", "top")},
                       tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}))
    ns.a_is_constant = False
    ns.on("acceleration", lambda s: s.a.__setitem__(0, s.a[0] + s.alpha[0]))
    ns.DT = 0.5 * h
    ns.start(events=[0.3])
    vmax = 0.0
    while ns.clock.t < 0.3 - 1e-12:
        ns.step()
        vmax = max(vmax, float(np.abs(ns.u[1].interior).max()))
        if vmax > 50:
            break
    return {"off": off, "uy_max": vmax, "f_change": float(np.abs(tp.f.interior - f0).max())}


def jobs():
    J = {}
    J["aligned"] = [(s, U, cfl, 1e3) for s in ("band", "square") for U in ((1.0, 0.0), (0.0, 1.0), (1.0, 0.37)) for cfl in (None, 0.4999)]
    J["half"] = [(k, r, i0) for k in ("diag", "random") for r in (1e3, 1e-3) for i0 in (0, 1)]
    J["mirror"] = [(mu, hv) for mu in (0.0, 1e-2) for hv in (True, False)]
    J["events"] = [(0.0,), (0.01,)]
    J["capwave"] = [(32,), (64,)]
    J["shear"] = [(nv, n) for nv in ((1, 1), (0, 1)) for n in (32, 64)]
    rng = np.random.default_rng(0)
    J["hydrostatic"] = [(float(th), float(off), n) for n in (32, 64) for th in np.arange(0, 90.01, 2.5)
                        for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)])]
    J["poiseuille"] = [(0.0,), (0.3,)]
    return J


def _run(args):
    name, cfg, a = args
    t0 = time.time()
    try:
        r = globals()[name](cfg, a)
    except Exception as exc:  # noqa: BLE001
        r = {"args": repr(a), "failed": repr(exc)[:200]}
    r["_item"], r["_seconds"] = name, round(time.time() - t0, 1)
    return r


def summary(R):
    out = {}
    for name in jobs():
        rs = [r for r in R if r["_item"] == name]
        if not rs:
            continue
        bad = [r for r in rs if "failed" in r]
        ok = [r for r in rs if "failed" not in r]
        s = {"runs": len(rs), "failed": len(bad)}
        if bad:
            s["first_failure"] = bad[0]["failed"]
        if name in ("aligned", "half"):
            s["worst_max_err"] = max((r["max_err"] for r in ok), default=None)
        elif name == "mirror":
            s["worst_max_diff"] = max((r["max_diff"] for r in ok), default=None)
        elif name == "events":
            s["values"] = [(r["mu"], r["none"], r["gap1e-7"]) for r in ok]
        elif name == "capwave":
            s["values"] = [(r["n"], round(r["period"], 5), round(r["damping"], 5), "%+.2f%%" % (100 * r["damping_err"])) for r in ok]
        elif name == "shear":
            s["values"] = [(r["n_vec"], r["N"], "%.3e" % r["L2_tangential"], "%.3e" % r["L2_normal"], "%.1e" % r["momentum_change"], "%.1e" % r["mass_change"]) for r in ok]
        elif name == "hydrostatic":
            s["configs_with_phi_gt_1e-8"] = sum(r["max_phi"] > 1e-8 for r in ok)
            s["worst_phi"] = max((r["max_phi"] for r in ok), default=None)
        elif name == "poiseuille":
            s["values"] = [(r["off"], "%.2e" % r["uy_max"], "%.2e" % r["f_change"]) for r in ok]
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = NAMED.get(a.config) or json.loads(open(a.config).read() if os.path.exists(a.config) else a.config)
    only = [s for s in a.only.split(",") if s]
    J = [(name, cfg, args) for name, lst in jobs().items() if not only or name in only for args in lst]
    order = {"capwave": 0, "shear": 1, "aligned": 2, "mirror": 3}
    J.sort(key=lambda j: order.get(j[0], 9))
    with mp.get_context("spawn").Pool(a.procs) as pool:
        R = pool.map(_run, J, chunksize=1)
    import basilisk_cupy

    res = {"config": cfg, "library": os.path.dirname(basilisk_cupy.__file__), "summary": summary(R), "runs": R}
    if a.out:
        json.dump(res, open(a.out, "w"), default=float, indent=1)
    print(json.dumps({"config": a.config, "library": res["library"], "summary": res["summary"]}, default=float, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""監修（C）：ノート113 の 6 節（時間 1 次の係数を場の差で測る）の再現と、A が試していない条件。
(1) A の流れ・boost で N = 64・128 を再実行。(2) 別の boost U = (-0.6, 1.1)。
(3) 監修の製造解 mix3：3 つの固有値（k^2 = 4π^2·{5, 2, 9}）、時間因子も別。A の mix の記号式の代わりに差し込む。"""
import os, sys, math, json, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

def mix3(nu, U):
    import sympy as sp
    x, y, t = sp.symbols("x y t")
    X, Y, p = x - U[0] * t, y - U[1] * t, 2 * sp.pi
    psi = (sp.Rational(1, 2) * sp.cos(2 * t) * sp.sin(p * (X + 2 * Y)) + sp.Rational(2, 5) * (1 + sp.sin(t)) * sp.cos(p * (X - Y) + 1)
           + sp.Rational(1, 5) * sp.exp(-2 * t) * sp.sin(p * 3 * X + 2)) / p
    u, v = U[0] + sp.diff(psi, y), U[1] - sp.diff(psi, x)
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
    a = [sp.diff(c, t) + u * sp.diff(c, x) + v * sp.diff(c, y) - nu * lap(c) for c in (u, v)]
    return [sp.lambdify((x, y, t), e, "numpy") for e in (psi, u, v, a[0], a[1])]

def job(a):
    boost, flow, name, n, cfl, path = a
    import cases.r0p1_unsteady_viscous as m
    m.BOOST = boost
    nu = 0.01 if flow in ("mix", "mix3") else m.NU[flow]
    if flow == "mix3":
        m.mix.cache[(float(nu), tuple(boost))] = mix3(nu, boost)
        flow_ = "mix"
    else:
        flow_ = flow
    o = m.run(n, cfl, flow=flow_, nu=nu, boost=True, t_ends=(0.5,), save=path, **m.CONFIGS[name])
    e = o["errors"][-1] if o["errors"] else {}
    return a, e.get("steps"), e.get("max"), o.get("failed")

if __name__ == "__main__":
    d = tempfile.mkdtemp(dir=sys.argv[1])
    cases = [((1.0, 0.37), "asym"), ((1.0, 0.37), "mix"), ((-0.6, 1.1), "asym"), ((-0.6, 1.1), "mix"), ((1.0, 0.37), "mix3"), ((-0.6, 1.1), "mix3")]
    J = []
    for boost, flow in cases:
        for name in ("cons", "cons/ab+visc+cn", "cons/ab+lag+cn"):
            for n in (32, 64, 128):
                for cfl in (0.25, 0.125, 0.0625):
                    J.append((boost, flow, name, n, cfl, os.path.join(d, "%d.npy" % len(J))))
    J.sort(key=lambda a: -a[3] ** 3 / a[4])
    res = {}
    with mp.get_context("spawn").Pool(14, maxtasksperchild=1) as pool:
        for a, steps, mx, failed in pool.imap_unordered(job, J, chunksize=1):
            res[a[:5]] = (a[5], steps, mx, failed)
    for boost, flow in cases:
        for name in ("cons", "cons/ab+visc+cn", "cons/ab+lag+cn"):
            a1, err = [], []
            for n in (32, 64, 128):
                r = [res[(boost, flow, name, n, c)] for c in (0.25, 0.125, 0.0625)]
                f = [np.load(x[0]) for x in r]; dt = [0.5 / x[1] for x in r]
                D = np.hypot(f[1][0] - f[2][0], f[1][1] - f[2][1])
                a1.append(float(np.sqrt((D * D).mean()) / (dt[1] - dt[2]))); err.append(r[2][2])
            print("U=%-12s %-4s %-16s a1(RMS, N=32/64/128) %.3g / %.3g / %.3g  h の次数 %.2f %.2f   max 誤差(CFL .0625) %.2e %.2e %.2e  failed %s"
                  % (boost, flow, name, *a1, math.log2(a1[0] / a1[1]), math.log2(a1[1] / a1[2]), *err, any(res[(boost, flow, name, n, c)][3] for n in (32, 64, 128) for c in (0.25, 0.125, 0.0625))), flush=True)

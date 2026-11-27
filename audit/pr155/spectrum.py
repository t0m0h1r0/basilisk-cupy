"""監修（C）：1 ステップの写像 g^n -> g^{n+1}（幾何を固定）の行列を全て作り、固有値を直接求める（A はべき乗法）。
A の one_step_gain と同じ状態（ステップ k、u = U、g = 0 の周り）。写像は線形なので、単位ベクトルの像が列になる。"""
import os, sys, math
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

def matrix(a):
    ratio, mode, step, n = a
    from basilisk_cupy.backend import to_numpy
    from cases.r0p1_source_feedback import solver, _drop, U0
    grid, ns, tp = solver(n, ratio, mode, f0=_drop, tolerance=1e-11)
    xp = grid.xp
    ns.cfl = 0.5
    for ax in range(2):
        ns.u[ax].interior = U0[ax] + 0.0 * ns.u[ax].interior
    src = tp.conserving_source
    hooks = {k: list(v) for k, v in ns._hooks.items()}
    def plain(on):
        tp.conserving_source = src and not on
        for k in ("vof", "tracer_advection"):
            ns._hooks[k] = [h for h in hooks[k] if getattr(h, "__self__", None) is not None] if on else list(hooks[k])
    plain(True)
    ns.start()
    for _ in range(step):
        ns.step()
    plain(False)
    c = ns.clock
    save = (tp.f.data.copy(), ns.p.data.copy(), ns.pf.data.copy(), [x.copy() for x in ns.uf.components],
            (c.t, c.i, c.dt, c.tnext, c.previous), ns.dtmax)
    def apply(g, eps):
        tp.f.data[...] = save[0]; ns.p.data[...] = save[1]; ns.pf.data[...] = save[2]
        for x, y in zip(ns.uf.components, save[3]):
            x[...] = y
        ns.clock.t, ns.clock.i, ns.clock.dt, ns.clock.tnext, ns.clock.previous = save[4]
        ns.dtmax = save[5]
        for ax in range(2):
            ns.u[ax].interior = U0[ax] + 0.0 * ns.u[ax].interior
            ns.g[ax].interior = xp.asarray(eps * g[ax])
        ns.u.fill_ghosts(); ns.g.fill_ghosts()
        ns.step()
        return np.array([to_numpy(ns.g[ax].interior) / eps for ax in range(2)])
    rng = np.random.default_rng(1)
    v = rng.standard_normal((2, n, n))
    lin = float(np.abs(apply(v, 1e-2) - apply(v, 1e-5)).max() / np.abs(apply(v, 1e-2)).max())
    zero = float(np.abs(apply(0 * v, 1.0)).max())
    M = np.zeros((2 * n * n, 2 * n * n))
    for j in range(2 * n * n):
        e = np.zeros(2 * n * n); e[j] = 1.0
        M[:, j] = apply(e.reshape(2, n, n), 1e-2).ravel()
    lam = np.linalg.eigvals(M)
    k = np.argsort(-np.abs(lam))
    return (ratio, mode, step, n, float(np.abs(lam[k[0]])), [complex(round(float(z.real), 3), round(float(z.imag), 3)) for z in lam[k[:4]]], lin, zero)

if __name__ == "__main__":
    J = [(r, "face", s, 32) for r in (100.0, 300.0, 1000.0, 1e4) for s in (12, 24)] + [(1000.0, "accel", 12, 32), (1e4, "accel", 12, 32)]
    with mp.get_context("spawn").Pool(5) as pool:
        for r in pool.imap_unordered(matrix, J):
            print("ratio %-6g %-6s step %2d N %d: spectral radius %.4g  leading %s  linearity %.1e  image of 0 %.1e" % r, flush=True)

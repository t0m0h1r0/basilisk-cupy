"""監修（C）：ノート114 の「源の項の平均の取り方（上流化・質量加重）では直らない」は、リポジトリにコードも数値も無い。
独立に確かめる。vof._sweep の源の項の行だけを差し替える（"mean" はライブラリのまま）。
  upwind : 面の風上のセルの ρ_k g
  massw  : ρ_k × (面の両側の g の、混合密度 ρ(f) で重み付けた平均)"""
import os, sys, math, inspect
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

LINE = "ff = ff + 0.25 * dt * (_faces(sp, axis, 0) + _faces(sp, axis, 1))"
NEW = {"mean": LINE,
       "upwind": "ff = ff + 0.5 * dt * upwind(sp)",
       "massw": ("rm = _pad(xp.asarray(RHO[0]), periodic, xp); a_, b_ = _faces(rm, axis, 0), _faces(rm, axis, 1); "
                 "rk = 1.0 if not tr.inverse else RHO[1]; "
                 "ff = ff + 0.5 * dt * rk * (a_ * _faces(sp, axis, 0) + b_ * _faces(sp, axis, 1)) / (rk * (a_ + b_))")}

def run(a):
    kind, ratio, n = a
    from basilisk_cupy.backend import to_numpy
    import basilisk_cupy.interface.vof as vof
    from cases.r0p1_source_feedback import solver, _drop, U0
    src = inspect.getsource(vof._sweep)
    assert src.count(LINE) == 1
    ns_ = dict(vof.__dict__); ns_["RHO"] = [None, 1.0 / ratio]
    exec(compile(src.replace(LINE, NEW[kind]), "sweep_" + kind, "exec"), ns_)
    vof._sweep = ns_["_sweep"]
    grid, ns, tp = solver(n, ratio, "face", f0=_drop)
    def before(s):
        ns_["RHO"][0] = tp.rho(tp.f.interior).copy()
    ns.on("vof", before)
    for ax in range(2):
        ns.u[ax].interior = U0[ax] + 0.0 * ns.u[ax].interior
    ns.start()
    es = []
    for _ in range(200):
        ns.step()
        es.append(max(float(np.abs(to_numpy(ns.u[ax].interior) - U0[ax]).max()) for ax in range(2)))
        if not math.isfinite(es[-1]) or es[-1] > 1e2:
            break
    k1 = next((i for i, e in enumerate(es) if e > 1e-9), None)
    k2 = next((i for i, e in enumerate(es) if e > 1e-3), None)
    rate = (es[k2] / es[k1]) ** (1.0 / (k2 - k1)) if k1 is not None and k2 is not None and k2 > k1 else float("nan")
    return kind, ratio, n, rate, max(es), len(es)

if __name__ == "__main__":
    J = [(k, r, n) for k in NEW for r in (100.0, 300.0, 1000.0, 1e4) for n in (32,)] + [(k, 1000.0, 64) for k in NEW]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for r in pool.map(run, J, chunksize=1):
            print("%-7s ratio %-6g N=%d  growth/step %.3g  max %.2e  (%d steps)" % r, flush=True)

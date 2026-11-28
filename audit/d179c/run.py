"""監修 #179 第3版（59d8817）：d1 + V-CN（出し入れは保存した g^n = d1v2-addold）。mix、boost。
kappa  : 大きい kappa で発散しないか（c 現行・cons/pred+visc+cn と比べる）
refine : Δt の 1 次の係数が h に比例するか（T3 (b)）
events : ほぼ重なる事象（間隔 1e-7）で誤差が跳ばないか（c と比べる）"""
import os, sys, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "audit/d179c")
from r0p1_d1v2 import run

CFG = {"d1": dict(level="d1v2-addold", source="g+visc", visc="cn"), "pred": dict(level="pred", source="g+visc", visc="cn"),
       "c(lib)": dict(level="n", source="none", visc="be"), "d1-BE": dict(level="d1v2", source="g", visc="be"),
       "c-cn": dict(level="n", source="g+visc", visc="cn")}

def kappa(a):
    name, n, nu = a
    o = run(n, 0.5, flow="mix", nu=nu, boost=True, t_ends=(0.5,), variant="conserving", **CFG[name])
    e = o["errors"][-1]
    return "kappa %-7s N=%-3d nu=%-5g kappa~%.3g: max err %.3e" % (name, n, nu, nu * 0.5 / e["steps"] * n * n, e["max"])

def refine(a):
    name, n, nu = a
    f, dt = [], []
    for cfl in (0.5, 0.25, 0.125):
        p = "audit/d179c/f_%s_%d_%g_%g.npy" % (name[:2], n, cfl, nu)
        o = run(n, cfl, flow="mix", nu=nu, boost=True, t_ends=(0.5,), variant="conserving", save=p, **CFG[name])
        f.append(np.load(p)); dt.append(0.5 / o["errors"][-1]["steps"]); os.remove(p)
    a1 = [float(np.sqrt(((f[i] - f[i + 1]) ** 2).sum(0).mean())) / (dt[i] - dt[i + 1]) for i in range(2)]
    return "refine %-7s N=%-3d nu=%-5g a1 %.4f %.4f" % (name, n, nu, *a1)

def events(a):
    name, n, nu, gap = a
    t_ends = (0.1, 0.25) if gap is None else (0.1, 0.1 + gap, 0.25)
    o = run(n, 0.5, flow="mix", nu=nu, boost=True, t_ends=t_ends, variant="conserving", **CFG[name])
    return "events %-7s N=%-3d nu=%-5g gap=%-6s: max err at t=0.25 %.4e" % (name, n, nu, gap, o["errors"][-1]["max"])

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "kappa":
        J = [(c, n, nu) for c in ("d1", "pred", "c(lib)") for n, nu in ((64, 0.05), (128, 0.05), (64, 0.5), (64, 2.0), (128, 0.5), (128, 2.0))]
        fn = kappa
    elif mode == "refine":
        J = [("d1", n, nu) for n in (64, 128, 256) for nu in (0.01, 0.05)]
        fn = refine
    else:
        J = [(c, 64, nu, g) for c in ("d1", "c(lib)", "d1-BE", "pred") for nu in (0.0 if False else 0.01, 0.05) for g in (None, 1e-7)]
        fn = events
    with mp.get_context("spawn").Pool(16, maxtasksperchild=1) as pool:
        for line in pool.imap(fn, J, chunksize=1):
            print(line, flush=True)

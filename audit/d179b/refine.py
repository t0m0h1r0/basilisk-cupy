"""監修 #179 第2版：d1（g_P の源の項、予測に L u^n、V-CN）の mix の Δt の 1 次の係数。値は audit/d179 の表と比べる。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "audit/d179b")
from r0p1_d1v2 import run

def job(a):
    n, cfl, nu = a
    path = "audit/d179b/f_%d_%g_%g.npy" % (n, cfl, nu)
    o = run(n, cfl, flow="mix", nu=nu, boost=True, t_ends=(0.5,), save=path, variant="conserving", level="d1v2", source="g+visc", visc="cn")
    return a, path, o["errors"][-1]["steps"], o["errors"][-1]["max"]

if __name__ == "__main__":
    J = sorted([(n, c, nu) for n in (64, 128, 256) for c in (0.5, 0.25, 0.125) for nu in (0.01, 0.05)], key=lambda a: a[0] ** 3 / a[1])
    res = {}
    with mp.get_context("spawn").Pool(18, maxtasksperchild=1) as pool:
        for a, path, steps, err in pool.imap_unordered(job, J, chunksize=1):
            res[a] = (path, steps, err)
            n, c, nu = a
            k = [(n, cc, nu) for cc in (0.5, 0.25, 0.125)]
            if all(x in res for x in k):
                f = [np.load(res[x][0]) for x in k]
                dt = [0.5 / res[x][1] for x in k]
                a1 = [float(np.sqrt(((f[i] - f[i + 1]) ** 2).sum(0).mean())) / (dt[i] - dt[i + 1]) for i in range(2)]
                print("d1v2+cn N=%-3d nu=%-5g a1 %.4f  %.4f   max err(CFL .5) %.3e" % (n, nu, a1[0], a1[1], res[k[0]][2]), flush=True)

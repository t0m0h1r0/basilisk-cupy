"""監修 #179：運ぶ速度の予測に粘性の加速度が無い構成（cons/pred+visc+cn、d1 に最も近い）に、h によらない Δt の 1 次の項が
残るか。非固有モードの流れ mix（boost、粘性）で、ノート113 の「場の差」の方法（#153）。"""
import os, sys, json, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
sys.path.insert(0, ".")
from cases.r0p1_unsteady_viscous import run, CONFIGS

def job(a):
    name, n, cfl, nu = a
    path = "audit/d179/f_%s_%d_%g_%g.npy" % (name.replace("/", "-"), n, cfl, nu)
    o = run(n, cfl, flow="mix", nu=nu, boost=True, t_ends=(0.5,), save=path, **CONFIGS[name])
    return a, path, o["errors"][-1]["steps"], o["errors"][-1]["max"]

if __name__ == "__main__":
    names = ("cons/pred+visc+cn", "cons/exact+visc+cn", "cons/ab+visc+cn")
    J = [(m, n, c, nu) for m in names for n in (64, 128, 256) for c in (0.5, 0.25, 0.125) for nu in (0.01, 0.05) if m in CONFIGS]
    J.sort(key=lambda a: a[1] ** 3 / a[2])
    res = {}
    with mp.get_context("spawn").Pool(18, maxtasksperchild=1) as pool:
        for a, path, steps, err in pool.imap_unordered(job, J, chunksize=1):
            res[a] = (path, steps, err)
            m, n, c, nu = a
            k = [(m, n, cc, nu) for cc in (0.5, 0.25, 0.125)]
            if all(x in res for x in k):
                f = [np.load(res[x][0]) for x in k]
                dt = [0.5 / res[x][1] for x in k]
                a1 = []
                for i in range(2):
                    d = f[i] - f[i + 1]
                    a1.append(float(np.sqrt((d[0] ** 2 + d[1] ** 2).mean())) / (dt[i] - dt[i + 1]))
                print("%-20s N=%-3d nu=%-5g a1(RMS) CFL .5-.25 %.4f   .25-.125 %.4f   max err(CFL .5) %.3e" % (m, n, nu, a1[0], a1[1], res[k[0]][2]), flush=True)

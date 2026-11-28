"""監修 PR #197：面の上の界面の二層 Poiseuille が定常にならない現象の切り分け。cases/r0_parts の関数を土台に、1 つずつ変える。
x 方向の一様性（max |u - u の x 平均|）が 1e-8 を超える最初の時刻と、f が [0,1] を出る量も記録する。"""
import os, sys, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    name, kw = a
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases import r0_parts as P
    info = {"t1": None, "g": 0.0, "first": None}
    orig = CenteredNS.step
    def step(self):
        orig(self)
        u = np.asarray(self.u[0].interior); v = np.asarray(self.u[1].interior)
        dev = max(float(np.abs(u - u.mean(0)).max()), float(np.abs(v - v.mean(0)).max()))
        vm = float(np.abs(v).max())
        if info["t1"] is None and dev > 1e-8:
            info["t1"] = (round(self.clock.t, 4), self.clock.i)
        if info["first"] is None and vm > 1e-8:
            info["first"] = (round(self.clock.t, 4), self.clock.i, "uy %.1e xdev %.1e" % (vm, dev))
        info["g"] = max(info["g"], vm)
    CenteredNS.step = step
    r = P.couette_poiseuille(**kw)
    CenteredNS.step = orig
    return "%-34s: f change %.2e  uy max(all t) %.2e  unsteady %.2e | x-nonuniform first %s | uy>1e-8 first %s" % (
        name, r["f_change"], info["g"], r["unsteady"], info["t1"], info["first"])

if __name__ == "__main__":
    base = dict(kind="poiseuille", n=32, offset=0.0, t_end=2.0)
    J = [("base (A's)", base),
         ("offset 0.3h", dict(base, offset=0.3)),
         ("rho ratio 1", dict(base, rho=(1.0, 1.0))),
         ("mu ratio 1", dict(base, mu=(1.0, 1.0))),
         ("rho 1, mu 1", dict(base, rho=(1.0, 1.0), mu=(1.0, 1.0))),
         ("non-conserving", dict(base, conf={"conserving": False})),
         ("project_start", dict(base, conf={"project_start": True})),
         ("dt 0.1/n", dict(base, dt_max=0.1 / 32)),
         ("dt 0.02/n", dict(base, dt_max=0.02 / 32, t_end=1.0)),
         ("couette", dict(base, kind="couette")),
         ("heavy on top (yi: phase1 = light)", dict(base, rho=(1e-3, 1.0), mu=(1.0 / 55.0, 1.0))),
         ("fixes", dict(base, conf={"fixes": True}))]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

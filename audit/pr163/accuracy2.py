"""監修（C）：accuracy.py の続き。元の方式は時間 1 次（ノート113 の E1 と源の項の欠落）が支配するので、D2 の代償が
隠れる。密度比 1 なら面の値の源の項（ノート112）と運ぶ速度の外挿（ノート113 の E1 の修正）が安全に使えて 2 次になる。
その 2 次の基準の上で D2 の有無を比べる。対照として界面の無い場合（f = 0）も同じ構成で測る。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
from audit.pr163.accuracy import exact

def run(a):
    sch, n, shape, T = a
    from cases.r0_aligned_free_stream import scheme, SCHEMES
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        if shape == "none":
            f0 = np.zeros((n, n))
        else:
            f0 = to_numpy(fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.4) ** 2, refine=8))
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0, conserving=True,
                 conserving_source=True)
        st = {}
        def before(s):                                  # E1：運ぶ面の速度を t^{n+1/2} に外挿（ノート113）
            dt = s.clock.dt
            cur = [c.copy() for c in s.uf.components]
            if "uf" in st:
                r = 0.5 * dt / st["dt"]
                for c, x, y in zip(s.uf.components, cur, st["uf"]):
                    c[...] = x + r * (x - y)
            st["uf"], st["dt"] = cur, dt
        ns.on("vof", before)
        u0, v0 = exact(n, 0.0)
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.start(events=[T])
        while ns.clock.t < T - 1e-12:
            ns.step()
        ue, ve = exact(n, T)
        e = np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve)
        return a, float(e.max()), float(np.sqrt((e * e).mean())), float(e.mean())

if __name__ == "__main__":
    N = (32, 48, 64, 96, 128, 192, 256)
    J = [(s, n, sh, 0.25) for s in ("base", "d1+d2") for n in N for sh in ("none", "drop")]
    R = {}
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for a, mx, l2, l1 in pool.imap_unordered(run, J, chunksize=1):
            R[a] = (mx, l2, l1)
    for sh in ("none", "drop"):
        for s in ("base", "d1+d2"):
            for k, nm in enumerate(("max", "L2", "L1")):
                e = np.array([R[(s, n, sh, 0.25)][k] for n in N]); ns = np.array(N, float)
                sl = -np.polyfit(np.log(ns), np.log(e), 1)[0]; sf = -np.polyfit(np.log(ns[ns >= 96]), np.log(e[ns >= 96]), 1)[0]
                print("interface=%-4s %-6s %-3s %s  slope all %.2f, N>=96 %.2f" % (sh, s, nm, " ".join("%.2e" % v for v in e), sl, sf), flush=True)

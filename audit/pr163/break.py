"""監修（C）：PR #163 の修正（D1+D2）を、A の配置の外で確かめる。別の速度の向き、格子、密度比、固定の CFL 0.4999、
filtered=True、壁に沿う面に揃った帯、渦の中の面に揃った正方形（非一様な流れ：max|u| と総運動量）。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    kind, sch, name, U, ratio, n, extra = a
    from cases.r0_aligned_free_stream import scheme, SCHEMES, shape, fixed_dt
    import contextlib
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.boundary import Periodic
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((n, n))
        wall = kind == "wall"
        ns = CenteredNS(grid, periodic=(1,) if wall else (0, 1), tolerance=1e-10)
        bc = BoundaryConditions().set((1, 0), Periodic()).set((1, 1), Periodic()) if wall else BoundaryConditions.periodic(2)
        if wall:                       # x の両側が滑り壁、帯は y 方向の界面（y = 一定）を y に運ぶ
            j = np.arange(n)[None, :] * np.ones((n, 1))
            f0 = ((j >= n // 4) & (j < n // 2)).astype(float)
        else:
            f0 = shape(name, grid)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=1.0, rho2=1.0 / ratio, conserving=True,
                      filtered=(extra == "filtered"))
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        for k in range(2):
            ns.u[k].interior = U[k] + 0.0 * X
        if kind == "vortex":
            ns.u[0].interior = U[0] + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            ns.u[1].interior = U[1] - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        mom = lambda: np.array([float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[k].interior)).sum()) for k in range(2)])
        m0 = mom()
        m, umax, s = 0.0, 0.0, 0
        steps = 300 if kind == "vortex" else 800
        with fixed_dt(0.4999 / n / max(abs(U[0]), abs(U[1]))) if extra == "cfl.4999" else contextlib.nullcontext():
            ns.start()
            for s in range(steps):
                ns.step()
                uu = [to_numpy(ns.u[k].interior) for k in range(2)]
                if kind != "vortex":
                    m = max(m, max(float(np.abs(uu[k] - U[k]).max()) for k in range(2)))
                umax = max(umax, float(np.hypot(*uu).max()))
                if not math.isfinite(umax) or umax > 50:
                    break
                if kind == "vortex" and ns.clock.t > 1.0:
                    break
        res = ("max|u-U| %.2e" % m) if kind != "vortex" else ("max|u| %.3g, momentum change %.2e, t %.2f" % (umax, float(np.hypot(*(mom() - m0)) / np.hypot(*m0)), ns.clock.t))
        return "%-6s %-6s %-13s U=%-12s ratio=%-6g N=%-3d %-9s %s (%d steps)" % (kind, sch, name, U, ratio, n, extra, res, s + 1)

if __name__ == "__main__":
    J = []
    for sch in ("base", "d1+d2"):
        for name in ("column", "square"):
            for U in ((1.0, 1.0), (-0.6, 1.1), (0.5, 1.0), (-1.0, 0.0)):
                J.append(("free", sch, name, U, 1e3, 32, "-"))
            J += [("free", sch, name, (1.0, 0.37), r, 32, "-") for r in (1e4,)]
            J += [("free", sch, name, (1.0, 0.0), 1e3, 48, "-"), ("free", sch, name, (1.0, 0.37), 1e3, 32, "cfl.4999"),
                  ("free", sch, name, (1.0, 0.0), 1e3, 32, "filtered")]
            J += [("vortex", sch, name, (0.5, 0.2), r, 32, "-") for r in (1e3, 1e6)]
        J += [("wall", sch, "column-y", (0.0, 1.0), r, 32, "-") for r in (1e3, 1e6)]
    with mp.get_context("spawn").Pool(10, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(run, J, chunksize=1):
            print(line, flush=True)

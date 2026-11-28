"""監修（C）：D2（部分的なセルの勾配の項を 0）の精度への代償を、厳密解のある問題で測る（A は「判定できない」）。
密度比 1 の二相は単相と同じ流れなので、Taylor–Green + 一様流の厳密解（非粘性、周期）がそのまま使える。
液滴（受動的な界面）を置き、元の方式と D1+D2 の速度の誤差（max と L2）の次数を比べる。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
P = 2 * math.pi
U = (1.0, 0.37)
G, W = np.polynomial.legendre.leggauss(4); G, W = (G + 1) / 2, W / 2

def exact(n, t):
    i = np.arange(n) / n
    u = np.zeros((n, n)); v = np.zeros((n, n))
    for a, wa in zip(G, W):
        for b, wb in zip(G, W):
            X, Y = np.meshgrid(i + a / n - U[0] * t, i + b / n - U[1] * t, indexing="ij")
            u += wa * wb * np.sin(P * X) * np.cos(P * Y); v += -wa * wb * np.cos(P * X) * np.sin(P * Y)
    return U[0] + u, U[1] + v

def run(a):
    sch, n, ratio, T = a
    from cases.r0_aligned_free_stream import scheme, SCHEMES
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        f0 = to_numpy(fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.4) ** 2, refine=8))
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, conserving=True)
        u0, v0 = exact(n, 0.0)
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.start(events=[T])
        while ns.clock.t < T - 1e-12:
            ns.step()
        ue, ve = exact(n, T)
        e = np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve)
        return a, float(e.max()), float(np.sqrt((e * e).mean())), float(np.abs(e).mean())

if __name__ == "__main__":
    N = (32, 48, 64, 96, 128, 192, 256)
    J = [(s, n, 1.0, 0.25) for s in ("base", "d1", "d2", "d1+d2") for n in N]
    R = {}
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for a, mx, l2, l1 in pool.imap_unordered(run, J, chunksize=1):
            R[a] = (mx, l2, l1)
    for s in ("base", "d1", "d2", "d1+d2"):
        for k, nm in enumerate(("max", "L2", "L1")):
            e = np.array([R[(s, n, 1.0, 0.25)][k] for n in N]); ns = np.array(N, float)
            sl = -np.polyfit(np.log(ns), np.log(e), 1)[0]; sf = -np.polyfit(np.log(ns[ns >= 96]), np.log(e[ns >= 96]), 1)[0]
            print("%-6s %-3s %s  slope all %.2f, N>=96 %.2f" % (s, nm, " ".join("%.2e" % v for v in e), sl, sf), flush=True)

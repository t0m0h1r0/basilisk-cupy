"""監修（C-2）：条件 6 の後から足した配置（非粘性の格子に平行な二層、U = (0,1)、w = 0.5 sin 4π(y − 1/4)、t = 1）を
A のコードを使わずに作り直す（厳密解は並進）。D1 だけと D1 + D2 限定の速度の L1・L2 の比。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    sch, n, cfl = a
    from cases.r0_aligned_free_stream import scheme, SCHEMES
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        y = np.arange(n + 1) / n
        f0 = np.tile(np.clip((np.minimum(y[1:], 0.75) - np.maximum(y[:-1], 0.25)) * n, 0, 1), (n, 1))
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True)
        ns.cfl = cfl
        def cellw(t):                      # w = 0.5 sin 4π(y − t − 1/4) のセル平均
            F = -0.5 * np.cos(4 * np.pi * (y - t - 0.25)) / (4 * np.pi)
            return np.tile((F[1:] - F[:-1]) * n, (n, 1))
        ns.u[0].interior = cellw(0.0); ns.u[1].interior = 1.0 + 0.0 * ns.u[1].interior
        ns.start(events=[1.0])
        while ns.clock.t < 1.0 - 1e-12:
            ns.step()
        e = np.hypot(to_numpy(ns.u[0].interior) - cellw(ns.clock.t), to_numpy(ns.u[1].interior) - 1.0)
    return a, float(e.mean()), float(np.sqrt((e * e).mean())), float(e.max())

if __name__ == "__main__":
    J = [(s, n, 0.5) for s in ("d1", "d1+d2lim", "d1+geo") for n in (32, 64, 128, 256)]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        R = {a: (l1, l2, mx) for a, l1, l2, mx in pool.imap_unordered(run, J, chunksize=1)}
    for s in ("d1+d2lim", "d1+geo"):
        for n in (32, 64, 128, 256):
            b, r = R[("d1", n, 0.5)], R[(s, n, 0.5)]
            print("%-9s N=%-3d L1 %.3e (d1 %.3e) ratio L1 %.3f L2 %.3f Linf %.3f" % (s, n, r[0], b[0], r[0] / b[0], r[1] / b[1], r[2] / b[2]))

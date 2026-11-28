"""監修（#188 の B の依頼）：毛管波の減衰率の、現行と D1 + D2 限定（ライブラリの選択肢、PR #192）の差と、
許容誤差・界面の初期の高さのずれ・振幅の 1e-12 の摂動による変動。配置と読み方は cases/r0_combined_tension.capillary_wave と同じ。"""
import os, sys, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
RHO = (1.0, 1e-3); MU = (1e-2, 1e-2 / 55.0)

def run(a):
    cfg, n, tol, off, pert = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.boundary import Neumann, Periodic
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA, position
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    a0, t_end, h = 0.01 * (1 + pert), 1.2, 1.0 / n
    grid = CartesianGrid((n, n), origin=(0.0, -0.5))
    ns = CenteredNS(grid, periodic=(0,), tolerance=tol, mu=MU[0])
    bc = BoundaryConditions()
    for side in (0, 1):
        bc.set((0, side), Periodic()); bc.set((1, side), Neumann(0.0))
    f0 = np.asarray(fraction(grid, lambda x, y: a0 * np.cos(2 * math.pi * x) - (y - off * h), refine=8))
    kw = dict(conserving_consistent=True, conserving_slope_limit=3.0) if cfg == "r0" else {}
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=RHO[0], rho2=RHO[1], mu1=MU[0], mu2=MU[1], conserving=True, **kw)
    tp.add_surface_tension(1.0)
    samples = list(np.arange(1, int(t_end / 0.004) + 1) * 0.004)
    ns.start(events=samples)
    amp = []
    for te in samples:
        while ns.clock.t < te - 1e-12:
            ns.step()
        p = np.asarray(position(tp.f.interior, tp.f.bc, grid, (0.0, 1.0), (0.0, off * h)))
        col = p[0, :]
        amp.append((te, float(col[col < NODATA / 2].mean())))
    t = np.array([x[0] for x in amp]); y = np.array([x[1] for x in amp])
    ext = [k for k in range(1, len(y) - 1) if abs(y[k]) >= abs(y[k - 1]) and abs(y[k]) >= abs(y[k + 1])]
    return a, float(-np.polyfit(t[ext], np.log(np.abs(y[ext])), 1)[0])

if __name__ == "__main__":
    J = [(c, 32, tol, 0.0, 0.0) for c in ("base", "r0") for tol in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)]
    J += [(c, 32, 1e-10, off, 0.0) for c in ("base", "r0") for off in (0.2, 0.4, 0.6, 0.8)]
    J += [(c, 32, 1e-10, 0.0, p) for c in ("base", "r0") for p in (1e-12, -1e-12)]
    J += [(c, 64, tol, 0.0, 0.0) for c in ("base", "r0") for tol in (1e-10, 1e-11)]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        R = dict(pool.map(run, J, chunksize=1))
    for key in J:
        if key[0] == "base":
            r = ("r0",) + key[1:]
            print("N=%d tol=%-6g off=%gh pert=%+g : base %.8f  r0 %.8f  diff %+.2e" % (key[1], key[2], key[3], key[4], R[key], R[r], R[r] - R[key]))

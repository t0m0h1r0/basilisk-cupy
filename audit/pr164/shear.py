"""監修（C）C-1：D2 の各形の精度を、密度比 1000 の厳密解（C-3 の二層せん断流 + boost）で測る。
A の条件 b は密度比 1 で、限定の適用が一度も切らないことが証明できる（ノート119 の B 節）。ここでは密度の跳びがあり、
界面が boost で格子を横切る。n = (1,1)：斜め、n = (0,1)：格子の面に平行（N が 4 の倍数なら初期は面の上）。
誤差：共動の速度（接線）のセル平均の L1・L2・max。体積分率の L1 も。限定の適用が実際に切った面の数も数える。"""
import os, sys, math, warnings, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
from audit.pr164.oblique_shear_reference import Shear

RHO, MU, U, T = (1.0, 1e-3), (1e-2, 1e-2 / 55.0), (1.0, 0.37), 0.5

def run(a):
    sch, nvec, N, mu_on, amp, U_ = a
    U = U_
    from cases.r0_aligned_free_stream import scheme, SCHEMES, COUNT
    ref = Shear(n=nvec, U=U, rho=RHO, mu=MU if mu_on else (0.0, 0.0), elements=8, order=14) if mu_on else Shear(n=nvec, U=U, rho=RHO, mu=MU)
    c = ref.coefficients(modes={1: amp, 2: 0.4 * amp})
    COUNT[0] = COUNT[1] = 0
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=max(MU))
        f0 = np.clip(ref.fraction_cells(N, 0.0), 0.0, 1.0)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=RHO[0], rho2=RHO[1],
                      mu1=MU[0], mu2=MU[1], conserving=True)
        u0, v0 = ref.velocity_cells(N, c, 0.0)
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.start(events=[T])
        failed = ""
        while ns.clock.t < T - 1e-12:
            ns.step()
            if not np.isfinite(to_numpy(ns.u[0].interior)).all() or float(np.abs(to_numpy(ns.u[0].interior)).max()) > 50:
                failed = "blow-up t=%.3f" % ns.clock.t; break
        ue, ve = ref.velocity_cells(N, c, ns.clock.t)
        e = np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve)
        ef = float(np.abs(to_numpy(tp.f.interior) - ref.fraction_cells(N, ns.clock.t)).mean())
    return {"sch": sch, "nvec": list(nvec), "N": N, "max": float(e.max()), "L2": float(np.sqrt((e * e).mean())), "L1": float(e.mean()),
            "f_L1": ef, "cut": COUNT[0], "partial": COUNT[1], "failed": failed, "amp": amp, "U": list(U)}

if __name__ == "__main__":
    J = [(s, n, N, True, 1.0, U_) for s in ("base", "d1+d2lim", "d1+d2", "d1+geo") for n, U_ in (((0, 1), (0.3, 1.0)), ((0, 1), (0.0, 1.0)), ((1, 1), (1.0, 1.0)), ((1, 1), (0.9, 1.0))) for N in (32, 64, 128)]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool, open("audit/pr164/shear2.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r, flush=True)

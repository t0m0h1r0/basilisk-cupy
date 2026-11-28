"""監修（C）C-1：限定の適用（利得 > 1 だけ切る）を壊しにいく。利得は残留 δ（= 界面の位置と CFL）で決まるので、
固定の CFL を 0.3〜0.4999 で細かく、界面の位置（面からのずれ）を数通り走査する。利得が 1 のすぐ下の配置が繰り返されると
育つ可能性がある。比較に全面の適用と元の方式。"""
import os, math, warnings, json, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    sch, cfl, off, ratio, U, N = a
    from cases.r0_aligned_free_stream import scheme, SCHEMES, fixed_dt
    with scheme(*SCHEMES[sch]):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        i = np.arange(N)[:, None] * np.ones((1, N))
        f0 = np.clip(np.minimum(i + 1, N // 2 + off) - np.maximum(i, N // 4 + off), 0, 1)
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, conserving=True)
        for k in range(2):
            ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
        m = 0.0
        with fixed_dt(cfl / N / max(abs(U[0]), abs(U[1]))):
            ns.start()
            for s in range(600):
                ns.step()
                e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
                m = max(m, e)
                if not math.isfinite(e) or e > 10:
                    break
    return {"sch": sch, "cfl": cfl, "off": off, "ratio": ratio, "U": U, "N": N, "max": m}

if __name__ == "__main__":
    cfls = [0.26, 0.3, 0.33, 0.36, 0.38, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.495, 0.498, 0.499, 0.4999]
    J = [("d1+d2lim", c, o, r, (1.0, 0.0), 32) for c in cfls for o in (0.0, 0.05, 0.13, 0.25, 0.4, 0.5) for r in (1e3, 1e6)]
    J += [("d1+d2lim", c, o, r, (1.0, 0.37), 32) for c in (0.4, 0.45, 0.49) for o in (0.0, 0.13, 0.4) for r in (1e3, 1e4, 1e6)]
    J += [("d1+d2", c, o, 1e6, (1.0, 0.0), 32) for c in (0.36, 0.42, 0.46, 0.49) for o in (0.0, 0.13, 0.4)]
    with mp.get_context("spawn").Pool(16, maxtasksperchild=1) as pool, open("audit/pr164/threshold.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()

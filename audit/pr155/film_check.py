"""監修（C）：薄い膜・密度比 1e6 では、対照（源の項なし）も崩れる。圧力の求解の収束の問題かを見る。"""
import math, numpy as np
from basilisk_cupy.backend import to_numpy
from cases.r0p1_source_feedback import solver, U0
from audit.pr155.break_stability import geometry
for ratio in (1e3, 1e4, 1e6):
    for mode in ("none", "accel", "trapezoid"):
        for tol in (1e-10, 1e-12):
            grid, ns, tp = solver(32, ratio, mode, f0=geometry("film"), tolerance=tol)
            for ax in range(2):
                ns.u[ax].interior = U0[ax] + 0.0 * ns.u[ax].interior
            ns.start()
            m, worst_res, its = 0.0, 0.0, 0
            for _ in range(60):
                ns.step()
                m = max(m, max(float(np.abs(to_numpy(ns.u[a].interior) - U0[a]).max()) for a in range(2)))
                worst_res = max(worst_res, float(getattr(ns.mgp, "resa", float("nan")))); its = max(its, int(getattr(ns.mgp, "i", -1)))
            print("film ratio %-6g %-9s tol %.0e: max|u-U| (60 steps) %.2e   pressure solve: worst residual %.2e, max V-cycles %d" % (ratio, mode, tol, m, worst_res, its), flush=True)

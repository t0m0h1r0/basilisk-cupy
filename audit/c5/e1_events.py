"""監修 C-5：E1（運ぶ面の速度の外挿 uf + (Δt/(2Δt_prev))(uf − uf_prev)）と、時間刻みの跳び。
Basilisk の dtnext は、事象の時刻を正確に踏むために刻みを縮める（次の事象まで Δt 未満なら、その残りを 1 歩で進む）。
ほぼ重なる 2 つの事象（出力の列が非整合のときなど）があると、直前の刻みが非常に小さく、次の刻みの外挿の係数
Δt/(2Δt_prev) が非常に大きくなる。予測：E1 を含む構成は壊れる／E1 の無い構成は壊れない。
配置：(a) 単相の Taylor–Green + boost（ノート113 の E1 + E2 + E3）、(b) 二相の一様流の液滴（E3 を外した統合構成）。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    case, conf, gap = a
    import contextlib
    from cases.r0_combined import configuration, combined
    ctx = contextlib.nullcontext()
    if case == "drop" and conf != "base":
        ctx = configuration(conf)
    elif case == "tg" and conf == "E1E2":
        ctx = combined(e3=False, band=None)
    with ctx:
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        n = 64
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        if case == "drop":
            f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
            rho2 = 1e-3
        else:
            f0 = np.zeros((n, n)); rho2 = 1.0
        TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=rho2, conserving=True)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        U = (1.0, 0.37)
        def exact(t):
            if case == "drop":
                return U[0] + 0 * X, U[1] + 0 * X
            s = math.sin(math.pi / n) / (math.pi / n)     # セル平均
            return (U[0] + s * s * np.sin(2 * np.pi * (X - U[0] * t)) * np.cos(2 * np.pi * (Y - U[1] * t)),
                    U[1] - s * s * np.cos(2 * np.pi * (X - U[0] * t)) * np.sin(2 * np.pi * (Y - U[1] * t)))
        u0, v0 = exact(0.0)
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ev = [0.1, 0.25] if gap is None else [0.1, 0.1 + gap, 0.25]
        ns.start(events=ev)
        err_before, err_max, prev_dt, rmax, hist = 0.0, 0.0, None, 0.0, []
        while ns.clock.t < 0.25 - 1e-12:
            ns.step()
            if prev_dt:
                rmax = max(rmax, ns.clock.dt / (2 * prev_dt))
            prev_dt = ns.clock.dt
            ue, ve = exact(ns.clock.t)
            e = float(np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve).max())
            if ns.clock.t <= 0.1:
                err_before = max(err_before, e)
            err_max = max(err_max, e)
            if not math.isfinite(e) or e > 100:
                break
    return "%-4s %-10s gap=%-8s: max error before t=0.1 %.2e, over the run %.2e (end t=%.3f), largest extrapolation factor %.3g" % (
        case, conf, gap, err_before, err_max, ns.clock.t, rmax)

if __name__ == "__main__":
    J = [("tg", c, g) for c in ("base", "E1E2") for g in (None, 1e-3, 1e-5, 1e-7)]
    J += [("drop", c, g) for c in ("base", "full-noE3") for g in (None, 1e-3, 1e-5, 1e-7)]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

"""監修（C）PR #188：事象の重なりの誤差の跳びは E1 と面の値の源の項のどちらか。C-5 の配置（単相 TG + boost、非粘性、N=64）。"""
import os, math, warnings, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    conf, gap, tol = a
    from cases.r0_combined import combined
    kw = {"base": None, "source only": dict(e1=False, e2=False, e3=False, band=None),
          "E1 only (no source)": dict(e2=False, e3=False, band=None, source=False),
          "E1 + source": dict(e2=False, e3=False, band=None)}[conf]
    with (combined(**kw) if kw else contextlib.nullcontext()):
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        n = 64
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=tol)
        TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0, conserving=True)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        U = (1.0, 0.37)
        def exact(t):
            s = math.sin(math.pi / n) / (math.pi / n)
            return (U[0] + s * s * np.sin(2 * np.pi * (X - U[0] * t)) * np.cos(2 * np.pi * (Y - U[1] * t)),
                    U[1] - s * s * np.cos(2 * np.pi * (X - U[0] * t)) * np.sin(2 * np.pi * (Y - U[1] * t)))
        ns.u[0].interior, ns.u[1].interior = exact(0.0)
        ns.start(events=[0.1, 0.25] if gap is None else [0.1, 0.1 + gap, 0.25])
        m = 0.0
        while ns.clock.t < 0.25 - 1e-12:
            ns.step()
            ue, ve = exact(ns.clock.t)
            m = max(m, float(np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve).max()))
    return "%-20s gap=%-6s tol=%g: max error %.3e" % (conf, gap, tol, m)

if __name__ == "__main__":
    J = [(c, g, t) for c in ("base", "source only", "E1 only (no source)", "E1 + source") for g in (None, 1e-7) for t in (1e-10, 1e-13)]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

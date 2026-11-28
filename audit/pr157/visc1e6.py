"""監修（C）：密度比 1e6・粘性ありの総運動量 1.7e-3 は、保存の破れか、圧力・粘性の求解の未収束かを分ける。"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from basilisk_cupy import CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.r0p2_pressure_momentum import _bc
for ratio in (1e3, 1e6):
    for tol in (1e-10, 1e-13):
        n = 32
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=tol, mu=1e-2)
        f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, _bc((0, 1))), rho1=1.0, rho2=1.0 / ratio, conserving=True, mu1=1e-2, mu2=1e-2 / 55)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        mom = lambda: np.array([float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[k].interior)).sum()) for k in range(2)])
        m0 = mom(); ns.start(events=[0.05]); worst = 0.0
        while ns.clock.t < 0.05 - 1e-12 and ns.clock.i < 200:
            ns.step()
            worst = max(worst, float(np.hypot(*(mom() - m0)) / np.hypot(*m0)))
        info = {k: (getattr(v, "i", None), getattr(v, "resa", None)) for k, v in vars(ns).items() if k.startswith("mg")}
        print("ratio %-6g tolerance %.0e: momentum change %.2e (t = %.3f, %d steps)  solver stats (cycles, residual) %s" % (ratio, tol, worst, ns.clock.t, ns.clock.i, info), flush=True)

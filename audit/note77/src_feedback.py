"""監修（C）：ノート77 の「源の項（-src）は密度比 1000 で 1 ステップに約 1.8 倍で育つ」の再現と、条件への依存。
一様流に乗る液滴（cases.pure_advection.free_stream と同じ設定）。各ステップの max|u - U| を記録し、成長率を当てはめる。
  ghosts="image"：運動量のゴーストセルを周期の像にする（ノート112 の差し替え）。不安定がゴーストセルと独立かを見る。"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.convection_stability import conserving_options
from cases.r0p1_local_error import periodic_momentum_ghosts

def run(variant, n, ratio, steps, U=(1.0, 0.37), cfl=None):
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8),
                                BoundaryConditions.periodic(2))
    TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, **conserving_options(variant))
    if cfl is not None:
        ns.cfl = cfl
    for ax in range(2):
        ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
    ns.start()
    hist = []
    for _ in range(steps):
        ns.step()
        e = max(float(np.abs(to_numpy(ns.u[ax].interior) - U[ax]).max()) for ax in range(2))
        hist.append(e)
        if not np.isfinite(e) or e > 1e3:
            break
    return np.array(hist)

def growth(h):
    """丸めより上（1e-11〜1e-1）の区間の、1 ステップあたりの成長率（対数の最小二乗）。"""
    k = np.where((h > 1e-11) & (h < 1e-1))[0]
    if len(k) < 6 or h[-1] < 1e-8:
        return float("nan")
    return float(np.exp(np.polyfit(k, np.log(h[k]), 1)[0]))

if __name__ == "__main__":
    for ghosts in ("library", "image"):
        for variant in ("conserving", "conserving-src"):
            for n in (32, 64):
                for ratio in (1.0, 10.0, 100.0, 1000.0):
                    if ghosts == "image":
                        with periodic_momentum_ghosts():
                            h = run(variant, n, ratio, 60)
                    else:
                        h = run(variant, n, ratio, 60)
                    print("%-8s %-15s n=%2d ratio=%-6g steps=%2d  max %.2e  step10 %.1e  last %.1e  growth/step %.3f"
                          % (ghosts, variant, n, ratio, len(h), h.max(), h[min(9, len(h) - 1)], h[-1], growth(h)), flush=True)

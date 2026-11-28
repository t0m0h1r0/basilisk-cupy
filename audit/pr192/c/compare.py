"""ライブラリ（PR #192 の選択肢）と C 版の独立の実装の、切る流れの最終の場の比較。実行は PR のツリーを PYTHONPATH に。"""
import sys, math, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
N, k = 64, 4 * math.pi
def cells(t):
    y0 = np.arange(N) / N - t
    return 0.5 * (np.cos(k * (y0 - 0.25)) - np.cos(k * (y0 + 1.0 / N - 0.25))) / (k / N)
y = (np.arange(N) + 0.5) / N
f0 = np.tile(((y > 0.25) & (y < 0.75)).astype(float), (N, 1))
for th, kw in ((3.0, dict(conserving_consistent=True, conserving_slope_limit=3.0)), (1.0, dict(conserving_consistent=True, conserving_slope_limit=1.0)),
               (1e30, dict(conserving_consistent=True)), ("base", {})):
    grid = CartesianGrid((N, N))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, conserving_stale_ghosts=True, **kw)
    ns.cfl = 0.5
    ns.u[0].interior = np.tile(cells(0.0), (N, 1)); ns.u[1].interior = np.ones((N, N))
    ns.start(events=[1.0])
    while ns.clock.t < 1.0 - 1e-12:
        ns.step()
    ux, uy, f = ns.u[0].interior, ns.u[1].interior, tp.f.interior
    err = float(np.abs(ux - np.tile(cells(1.0), (N, 1))).mean())
    cth = th if th != "base" else None
    line = "theta=%-5s steps %d  L1(ux) %.5e" % (th, ns.clock.i, err)
    if cth is not None:
        d = np.loadtxt("%s/out_%s.txt" % (sys.argv[1], {3.0: "3", 1.0: "1", 1e30: "1e30"}[cth]))
        i, j = d[:, 0].astype(int), d[:, 1].astype(int)
        cu, cv, cf = np.empty((N, N)), np.empty((N, N)), np.empty((N, N))
        cu[i, j], cv[i, j], cf[i, j] = d[:, 2], d[:, 3], d[:, 4]
        line += "   |library - C|: ux %.1e  uy %.1e  f %.1e   C L1 %.5e" % (np.abs(ux - cu).max(), np.abs(uy - cv).max(), np.abs(f - cf).max(),
                                                                   np.abs(cu - np.tile(cells(1.0), (N, 1))).mean())
    print(line, "  t=%.12g f col0 j=14..18 %s  j=46..50 %s" % (ns.clock.t, np.round(f[0,14:19],3), np.round(f[0,46:51],3)), flush=True)

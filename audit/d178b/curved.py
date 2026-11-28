"""監修：曲面と薄膜が壁に接する静的な配置で、界面の点 x_I の、真の界面からの距離（セル）。現行と A-1（+B-1）。
x_I は position を G = (1,0)、(0,1) で 2 回呼んで取り出す。"""
import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
spec = importlib.util.spec_from_file_location("d178", "design/checks/178_heights.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy import CartesianGrid, BoundaryConditions

def run(name, dist, phi, n):
    h = 1.0 / n
    grid = CartesianGrid((n, n))
    c = np.asarray(fraction(grid, phi, refine=16))
    bc = BoundaryConditions()
    out = []
    for var in ("library", "A-1"):
        eps = D.EPS if var != "library" else None
        px = D.position(c, bc, grid, (1.0, 0.0), (0.0, 0.0), variant=var, eps=eps)
        py = D.position(c, bc, grid, (0.0, 1.0), (0.0, 0.0), variant=var, eps=eps)
        ok = (px < D.H.NODATA / 2) & (py < D.H.NODATA / 2) & (c > 1e-6) & (c < 1 - 1e-6)
        d = dist(px[ok], py[ok]) / h
        out.append("%s: n=%d max %.2f mean %.3f cells, >1 cell: %d" % (var, int(ok.sum()), float(np.abs(d).max()), float(np.abs(d).mean()), int((np.abs(d) > 1).sum())))
    print("%-26s N=%-3d %s" % (name, n, " | ".join(out)))

for n in (32, 64):
    h = 1.0 / n
    for R, cx, cy, name in ((0.25, 0.5, 0.0, "half drop on wall 90deg"), (0.25, 0.5, 0.12, "drop cut by wall (~60deg)"),
                            (0.25, 0.5, -0.12, "drop cut by wall (~120deg)"), (2.5 * h, 0.5, 0.5 * h, "tiny drop R=2.5h at wall"),
                            (1.2 * h, 0.5 + 0.3 * h, 0.4 * h, "tiny drop R=1.2h at wall"), (0.2, 0.0, 0.0, "quarter drop in corner"), (0.3, 0.08, 0.05, "drop cut in corner")):
        phi = lambda x, y, R=R, cx=cx, cy=cy: R ** 2 - (x - cx) ** 2 - (y - cy) ** 2
        run(name, lambda a, b, R=R, cx=cx, cy=cy: np.hypot(a - cx, b - cy) - R, phi, n)
    for t, name in ((0.5 * h, "film thickness 0.5h"), (1.5 * h, "film thickness 1.5h")):
        run(name, lambda a, b, t=t: b - t, lambda x, y, t=t: t - y, n)
    t = 0.7 * h
    run("wavy film 0.7h +- 0.5h", lambda a, b: b - (0.7 * h + 0.5 * h * np.sin(2 * np.pi * a)), lambda x, y: 0.7 * h + 0.5 * h * np.sin(2 * np.pi * x) - y, n)

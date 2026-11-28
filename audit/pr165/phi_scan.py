"""監修（C）C-2：reduced の界面のポテンシャル φ = G·(x_I − Z) が、厳密に釣り合う平面で 0 かを、角度と位置を広く走査する。
A の主張「0 でないのは接点が直交する壁から 1 セル以内のときだけ」を、ノート115 の 24 配置の外で確かめる。σ = 0 の
静止解は、壁との角度によらず厳密解である（接触角は表面張力が無いと意味を持たない）。"""
import math, numpy as np
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.heights import NODATA, position

def check(n, th, off):
    t = math.radians(th); nx, ny = math.sin(t), math.cos(t)
    grid = CartesianGrid((n, n))
    f = CellField.from_interior(grid, np.asarray(fraction(grid, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off)))), BoundaryConditions())
    G = (-nx, -ny); Z = (0.5, 0.5 + off)
    phi = np.asarray(position(f.interior, f.bc, grid, G, Z))
    ok = phi < NODATA
    bad = np.abs(np.where(ok, phi, 0.0)) > 1e-8
    # 接点（平面と箱の辺の交点）から、最も近い角までの距離
    pts = []
    for x in (0.0, 1.0):
        y = 0.5 + off - (nx / ny) * (x - 0.5) if ny > 1e-12 else None
        if y is not None and 0 <= y <= 1: pts.append((x, y))
    for y in (0.0, 1.0):
        x = 0.5 - (ny / nx) * (y - 0.5 - off) if nx > 1e-12 else None
        if x is not None and 0 <= x <= 1: pts.append((x, y))
    d = min((math.hypot(px - cx, py - cy) for px, py in pts for cx in (0, 1) for cy in (0, 1)), default=9)
    far_bad = 0.0
    if bad.any():
        i, j = np.nonzero(bad)
        xc, yc = (i + 0.5) / n, (j + 0.5) / n
        far_bad = float(max(min(math.hypot(a - cx, b - cy) for cx in (0, 1) for cy in (0, 1)) for a, b in zip(xc, yc)) * n)
    return float(np.abs(np.where(ok, phi, 0)).max()), d * n, far_bad, int(ok.sum())

rng = np.random.default_rng(0)
rows = []
for n in (32, 64):
    for th in np.arange(0, 90.01, 2.5):
        for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)]):
            m, dc, fb, cnt = check(n, float(th), float(off))
            if cnt: rows.append((n, th, off, m, dc, fb))
bad = [r for r in rows if r[3] > 1e-8]
print("configurations", len(rows), " phi != 0:", len(bad))
print("phi != 0 with every contact point farther than 2 cells from every corner:", sum(r[4] > 2 for r in bad))
print("largest corner distance (cells) of a contact point among bad configurations: %.2f" % max(r[4] for r in bad))
print("largest distance (cells) of a cell with phi != 0 from the nearest corner: %.2f" % max(r[5] for r in bad))
for r in sorted(bad, key=lambda r: -r[4])[:6]:
    print("   n=%d theta=%.1f off=%+.4f max|phi|=%.2e nearest contact-corner %.2f cells, bad cells up to %.2f cells from a corner" % r)

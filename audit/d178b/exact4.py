"""1794 走査で残る 4 配置（45°、δ = 0・0.375）：D.plane の f と、厳密な面積（多角形の切り取り）の差。厳密な f での phi。"""
import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
spec = importlib.util.spec_from_file_location("d178", "design/checks/178_heights.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

def clip_area(x0, y0, h, nx, ny, d):
    """area of {nx x + ny y < d} in the cell, by polygon clipping."""
    poly = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
    out = []
    for k in range(4):
        a, b = poly[k], poly[(k + 1) % 4]
        sa, sb = nx * a[0] + ny * a[1] - d, nx * b[0] + ny * b[1] - d
        if sa <= 0:
            out.append(a)
        if sa * sb < 0:
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    if len(out) < 3:
        return 0.0
    s = 0.0
    for k in range(len(out)):
        a, b = out[k], out[(k + 1) % len(out)]
        s += (a[0] - x0) * (b[1] - y0) - (b[0] - x0) * (a[1] - y0)
    return abs(s) / 2 / (h * h)

for n, off in ((32, 0.0), (32, 0.375), (64, 0.0), (64, 0.375), (32, 0.1171)):
    th = 45.0
    grid, c, G, Z = D.plane(n, th, off)
    nx, ny = -G[0], -G[1]
    h = 1.0 / n
    d0 = nx * Z[0] + ny * Z[1]
    e1 = np.array([[clip_area(i * h, j * h, h, nx, ny, d0) for j in range(n)] for i in range(n)])
    best = e1 if np.abs(e1 - c).mean() < np.abs(1 - e1 - c).mean() else 1 - e1
    for name, f in (("D.plane", c), ("exact", best)):
        p = D.position(f, D.BoundaryConditions(), grid, G, Z, variant="A-1", eps=D.EPS)
        ok = p < D.H.NODATA / 2
        print("n=%d off=%+.4f %-8s: |f - exact| %.1e  cells not 0/1 but within 1e-12 of 0/1: %d  max|phi| %.2e" % (
            n, off, name, np.abs(f - best).max(), int((((f > 0) & (f < 1e-12)) | ((f < 1) & (f > 1 - 1e-12))).sum()), np.abs(np.where(ok, p, 0)).max()))

"""A-1+A-2 で残る配置：角度の分布、悪いセルの角からの距離、そのセルの高さ、写し（eps）の有無。"""
import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
exec(open("audit/d178/scan1850.py").read().split('if __name__')[0])
from collections import Counter
rng = np.random.default_rng(0)
J = [(n, float(th), float(off)) for n in (32, 64) for th in np.arange(0, 90.01, 2.5) for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)])]
for eps in (D.EPS, None):
    ths = Counter(); far = []; total = 0; examples = []
    for n, th, off in J:
        grid, c, G, Z = D.plane(n, th, off)
        p = D.position(c, D.BoundaryConditions(), grid, G, Z, variant="A-1+A-2", eps=eps)
        bad = (p < D.H.NODATA / 2) & (np.abs(np.where(p < D.H.NODATA / 2, p, 0.0)) > 1e-8)
        if not bad.any():
            continue
        total += 1; ths[th] += 1
        i, j = np.nonzero(bad)
        d = [min(max(min(a, n - 1 - a), min(b, n - 1 - b)) for _ in (0,)) for a, b in zip(i, j)]
        wall = [min(a, n - 1 - a, b, n - 1 - b) for a, b in zip(i, j)]
        far.append(max(d))
        if len(examples) < 3 or (max(d) > 2 and len(examples) < 6):
            a, b = int(i[0]), int(j[0])
            hs = D.H.heights(c, D.BoundaryConditions())
            examples.append((n, th, off, [(int(x), int(y)) for x, y in zip(i[:4], j[:4])], ["%.3g" % v for v in c[i[:4], j[:4]]], max(d)))
    print("eps=%s: %d configurations; theta: %s" % (eps, total, dict(sorted(ths.items()))))
    print("   distance (cells, Chebyshev) of the bad cells from the nearest corner: max over configs %d; configs with a bad cell farther than 2 cells: %d" % (max(far), sum(f > 2 for f in far)))
    for e in examples:
        print("   n=%d theta=%.1f off=%+.4f bad cells %s f=%s (corner distance %d)" % e)

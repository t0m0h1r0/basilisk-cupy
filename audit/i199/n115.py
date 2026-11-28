import sys, math
import numpy as np
sys.path.insert(0, "audit/i199")
from exact import halfplane
from basilisk_cupy import CartesianGrid
from basilisk_cupy.interface.fractions import fraction
worst, nbad, tot = 0.0, [], 0
for th in (0.0, 10.0, 26.565, 45.0):
    for off in (0.0137, 0.1171, 0.2313):
        for n in (32, 64, 128):
            t = math.radians(th); nx, ny = math.sin(t), math.cos(t)
            g = CartesianGrid((n, n))
            f = np.asarray(fraction(g, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off))))
            d = float(np.abs(f - halfplane(n, nx, ny, nx * 0.5 + ny * (0.5 + off))).max()); tot += 1
            worst = max(worst, d)
            if d > 1e-10:
                nbad.append((th, off, n, d))
print("note 115 planes: %d configurations, worst |fraction - exact| %.1e, differing: %s" % (tot, worst, nbad))

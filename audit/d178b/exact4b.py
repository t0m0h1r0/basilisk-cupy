import sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
exec(open("audit/d178b/exact4.py").read().split('for n, off in')[0])
offs = np.linspace(-0.45, 0.45, 13)
for n in (32, 64):
    for off in (float(offs[6]), float(offs[11]), 0.0, 0.375):
        grid, c, G, Z = D.plane(n, 45.0, off)
        nx, ny = -G[0], -G[1]; h = 1.0 / n; d0 = nx * Z[0] + ny * Z[1]
        e1 = np.array([[clip_area(i * h, j * h, h, nx, ny, d0) for j in range(n)] for i in range(n)])
        best = e1 if np.abs(e1 - c).mean() < np.abs(1 - e1 - c).mean() else 1 - e1
        row = []
        for f in (c, best):
            for eps in (None, D.EPS):
                p = D.position(f, D.BoundaryConditions(), grid, G, Z, variant="A-1", eps=eps)
                ok = p < D.H.NODATA / 2
                row.append(np.abs(np.where(ok, p, 0)).max())
        print("n=%d off=%r: |f - exact| %.1e | D.plane: no snap %.1e, B-1 %.1e | exact f: no snap %.1e, B-1 %.1e | slivers in D.plane %d" % (
            n, off, np.abs(c - best).max(), *row, int((((c > 0) & (c < 1e-10)) | ((c < 1) & (c > 1 - 1e-10))).sum())))

import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
exec(open("audit/d178/scan1850.py").read().split('if __name__')[0])
for n, th, off, cell in ((32, 5.0, -0.45, (23, 0)),):
    i, j = cell
    grid, c, G, Z = D.plane(n, th, off)
    bc = D.BoundaryConditions()
    p = D.position(c, bc, grid, G, Z, variant="A-1+A-2", eps=D.EPS)
    bad = np.argwhere((p < D.H.NODATA / 2) & (np.abs(np.where(p < D.H.NODATA / 2, p, 0)) > 1e-8))
    saved = D.H.pad_bc; D.H.pad_bc = D._pad_constant
    hs = D.H.heights(D.snap(c), bc); ext = D._extrapolate(hs, D.snap(c), bc); D.H.pad_bc = saved
    def val(h, a, b):
        v = float(D.H.height(np.asarray(h), np)[a, b]); return "none" if abs(v) > 100 else "%.3f" % v
    print("n=%d theta=%g off=%g: %d bad cells; cell (%d,%d) f=%.4f phi=%.2e (h=%.4f)" % (n, th, off, len(bad), i, j, c[i, j], p[i, j], 1.0/n))
    for ax, name in ((0, "x"), (1, "y")):
        t = 1 - ax
        idx = [(i + (d if t == 0 else 0), j + (d if t == 1 else 0)) for d in (-2, -1, 0, 1, 2)]
        print("   %s-heights along the other direction (-2..+2): raw %s | after A-2 %s" % (name,
              [val(hs[ax], a, b) if 0 <= a < n and 0 <= b < n else "wall" for a, b in idx],
              [val(ext[ax], a, b) if 0 <= a < n and 0 <= b < n else "wall" for a, b in idx]))

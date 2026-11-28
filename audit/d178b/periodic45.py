"""壁の無い周期の 45° の帯が格子の頂点を通る場合（と通らない場合）の、ライブラリの位置の phi。"""
import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
spec = importlib.util.spec_from_file_location("d178", "design/checks/178_heights.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
bc = D.BoundaryConditions.periodic(2)
for n in (32, 64):
    for off in (0.0, 0.5 / n, 1.0 / n, 0.0137, 0.25 / n):
        grid, c, r = D.band(n, 1, 1, off)
        for var in ("library", "A-1"):
            p = D.position(c, bc, grid, (1 / r, 1 / r), (0.0, 0.0), variant=var, eps=D.EPS if var != "library" else None)
            ok = p < D.H.NODATA / 2
            d = np.where(ok, (((p * r - off + 0.25) % 0.5) - 0.25) / r * n, 0.0)
            f = D.face_force(np.where(ok, (((p * r - off + 0.25) % 0.5) - 0.25) / r * n, D.H.NODATA), c, True)
            nbad = int((ok & (np.abs(d) > 1e-6)).sum())
            print("n=%-3d off=%.5f (%.2f h) %-7s: cells off the line %3d  max %.3f cells  face force %.2e  partial %d" % (
                n, off, off * n, var, nbad, float(np.abs(d).max()), f, int(((c > 1e-10) & (c < 1 - 1e-10)).sum())))

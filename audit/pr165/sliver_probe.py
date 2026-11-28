import math, numpy as np
from basilisk_cupy import CartesianGrid, CellField, BoundaryConditions
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.interface.heights import NODATA, position
from basilisk_cupy.time.basilisk_centered import CenteredNS
for th, off, n in ((37.5, 0.0, 64), (30.0, 0.40, 64)):
    t = math.radians(th); nx, ny = math.sin(t), math.cos(t)
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(), tolerance=1e-10)
    f0 = np.asarray(fraction(grid, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off))))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions()), rho1=1.0, rho2=1e-3, conserving=True)
    tp.add_reduced_gravity(G=(-nx, -ny), Z=(0.5, 0.5 + off))
    ns.DT = 0.5 / n
    ns.start()
    for k in range(4):
        p = np.asarray(position(tp.f.interior, tp.f.bc, grid, (-nx, -ny), (0.5, 0.5 + off)))
        f = tp.f.interior
        interf = (f > 1e-12) & (f < 1 - 1e-12)
        a = max(float(np.abs(c).max()) for c in ns.a.components) if not ns.a_is_constant else 0.0
        u = max(float(np.abs(c.interior).max()) for c in ns.u.components)
        ok = p < NODATA / 2
        # f が跳ぶのに両側の φ が無い面
        miss = 0; jump = 0
        for ax in (0, 1):
            fl = f.take(range(0, n - 1), ax); fr = f.take(range(1, n), ax)
            ol = ok.take(range(0, n - 1), ax); orr = ok.take(range(1, n), ax)
            j = np.abs(fr - fl) > 1e-12
            jump += int(j.sum()); miss += int((j & ~ol & ~orr).sum())
        print("theta=%g off=%g step %d: interfacial cells %d, with phi %d, max|phi| %.2e, faces with f jump %d (no phi on either side %d), max|a| %.2e max|u| %.2e  f-slivers %d" % (
            th, off, k, int(interf.sum()), int((ok & interf).sum()), float(np.abs(np.where(ok, p, 0)).max()), jump, miss, a, u, int(((f != 0) & (f != 1) & ~interf).sum())))
        bad = np.argwhere(ok & (np.abs(np.where(ok, p, 0)) > 1e-8))
        if len(bad):
            for i, j in bad[:4]:
                print("      bad cell", (int(i), int(j)), "phi %.2e f %.3g" % (p[i, j], f[i, j]), "column f (x):", ["%.3g" % v for v in f[max(0,i-3):i+4, j]], "(y):", ["%.3g" % v for v in f[i, max(0,j-3):j+4]])
        ns.step()

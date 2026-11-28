import os, sys, math, warnings
NS = int(os.environ.get("NS", "1"))
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
N, k = 64, 4 * math.pi
h = 1.0 / N
lo, hi = 0.25 + (1 - 0.033) * h, 0.75 + (1 - 0.033) * h
yl = np.arange(N) * h
f0 = np.tile(np.clip((np.minimum(yl + h, hi) - np.maximum(yl, lo)) / h, 0, 1), (N, 1))
ux = 0.5 * (np.cos(k * (yl - 0.25)) - np.cos(k * (yl + h - 0.25))) / (k * h)
for t in ("3", "1e30"):
    d = np.loadtxt("%s/one_%s.txt" % (sys.argv[1], t))
    i, j = d[:, 0].astype(int), d[:, 1].astype(int)
    C = [np.empty((N, N)) for _ in range(3)]
    for a, col in zip(C, (2, 3, 4)):
        a[i, j] = d[:, col]
    for stale in (False, True):
        kw = dict(conserving_consistent=True, conserving_stale_ghosts=stale)
        if t == "3":
            kw["conserving_slope_limit"] = 3.0
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-12)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, **kw)
        ns.cfl = 0.5; ns.DT = 0.007
        ns.u[0].interior = np.tile(ux, (N, 1)); ns.u[1].interior = np.ones((N, N))
        ns.start()
        dt0 = ns.clock.dt
        for _ in range(NS): ns.step()
        L = (ns.u[0].interior, ns.u[1].interior, tp.f.interior)
        dd = [np.abs(a - b) for a, b in zip(L, C)]
        print("t=%.8g " % ns.clock.t + "theta=%-4s stale=%d dt=%.6g: |lib - C| ux %.1e uy %.1e f %.1e   (ux: worst column-0 row %d)" % (t, stale, ns.clock.dt, dd[0].max(), dd[1].max(), dd[2].max(), int(np.argmax(dd[0][0]))))
a = np.loadtxt("%s/one_3.txt" % sys.argv[1]); b = np.loadtxt("%s/one_1e30.txt" % sys.argv[1])
print("C: theta=3 vs no limit, max |du_x| %.2e (nonzero = the limiter cut in this step)" % np.abs(a[:, 2] - b[:, 2]).max())

import warnings, numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
n = 32
def run(label, conserving=True, stale=False, tol=1e-10, trace=False, steps=600, noise=0.0, U=(1.0, 0.0), ratio=1e3):
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tol)
    i = np.arange(n)[:, None] * np.ones((1, n))
    f0 = ((i >= 8) & (i < 16)).astype(float)
    kw = dict(conserving=conserving)
    if stale:
        kw["conserving_stale_ghosts"] = True
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, **kw)
    ns.cfl = 0.5
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    ns.start()
    m = 0.0
    for s in range(steps):
        ns.step()
        du = [to_numpy(ns.u[k].interior) - U[k] for k in range(2)]
        e = max(float(np.abs(d).max()) for d in du); m = max(m, e)
        if trace and (s < 6 or s % 25 == 0):
            f = to_numpy(tp.f.interior)[:, 0]
            part = f[(f > 0) & (f < 1)]
            uf = to_numpy(ns.uf.components[0])
            print("  step %3d dt %.10g  err %.2e  f outside [0,1]: %.1e / %.1e  partial values %s  uf spread %.1e  mg i=%s resa=%.1e" % (
                s + 1, ns.clock.dt, e, min(f.min(), 0), max(f.max() - 1, 0), ["%.17g" % v for v in part][:4], float(uf.max() - uf.min()), ns.mgp.i, ns.mgp.resa))
        if not np.isfinite(e) or e > 10:
            break
    print("%-34s max|u-U| %.2e after %d steps" % (label, m, s + 1), flush=True)
run("conserving (main + PR160)", trace=True, steps=400)
run("conserving, stale ghosts (old)", stale=True)
run("non-conserving (BCG)", conserving=False)
run("conserving, tolerance 1e-13", tol=1e-13)
run("conserving, ratio 1e3, U=(1,0) N=32, 3000", steps=1500)

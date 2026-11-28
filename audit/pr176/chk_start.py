import warnings, numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.r0_shear import Shear
for n in ((1, 1), (1, 2)):
    for N in (32, 128):
        A = Shear(n=n)
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=1e-2)
        tp = TwoPhase(ns, CellField.from_interior(grid, np.clip(A.fraction_cells(N, 0.0), 0, 1), BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, mu1=1e-2, mu2=1e-2/55, conserving=True)
        u0, v0 = A.velocity_cells(N, 0.0)
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.start()
        m0 = float(to_numpy(tp.f.interior).sum())
        ns.step()
        uf = [to_numpy(c) for c in ns.uf.components]
        m1 = float(to_numpy(tp.f.interior).sum())
        print("n=%s N=%d: relative phase-mass change in the first step %.2e" % (n, N, abs(m1 - m0) / m0))

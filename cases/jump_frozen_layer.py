"""Frozen flat interface driven by a prescribed GFM jump ``j = J0 cos kx sin wt`` (docs/notes/40): the Stokes
layer with and without the jump correction (:mod:`cases.jump_viscous`), against N = 256.

    python -m cases.jump_frozen_layer
"""

import math, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.jump_viscous import attach
K, W, J0, NU = 2 * math.pi, 10.0, 0.08, 0.0182571749236
def run(n, corr=None, T=0.9, dt=1e-3):
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10, stokes=True)
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, y + 0 * x, BoundaryConditions(), ops=FCCDFaceComplex(order=4))
    cls.advect = lambda uf, dt: None
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=NU, mu2=NU, pressure_jump=True)
    tp.sigma = 1.0
    tp.potentials.append(lambda t_: J0 * np.cos(K * x) * math.sin(W * ns.clock.t) * np.ones_like(y))
    if corr is not None:
        attach(ns, tp, **corr)
    ns.DT = dt
    ns.start()
    while ns.clock.t < T - 1e-12:
        ns.step()
    xc = np.asarray(grid.centers(0)); w = np.sin(K * xc)[:, None] * 2.0 / n
    return np.asarray(grid.centers(1)), np.sum(ns.u[0].interior * w, axis=0)
def job(a):
    n, c = a
    return a, run(n, c)
if __name__ == "__main__":
    cases = [(256, None), (32, None), (32, {}), (32, {"kink": "face"}), (64, None), (64, {})]
    with ProcessPoolExecutor(6) as ex:
        res = dict(((n, str(c)), r) for (n, c), r in ex.map(job, cases))
    yr, ur = res[(256, "None")]
    for key, (y, u) in res.items():
        if key[0] == 256: continue
        ui = np.interp(y, yr, ur)
        m = np.abs(y) < 0.3
        print(key, f"max|u - u_256| near interface {np.abs(u - ui)[m].max():.3e}  (|u| max {np.abs(ui[m]).max():.3e})",
              " u at +-h/2:", np.round(u[len(y)//2 - 1: len(y)//2 + 1], 5), "ref", np.round(ui[len(y)//2 - 1: len(y)//2 + 1], 5))

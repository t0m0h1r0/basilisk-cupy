"""監修（C）：PR #160 を、A の試験（格子に平行な帯）以外の c = 1/2 の配置で確かめる。
配置：斜め 45° の帯（対角のセルが c = 1/2）、無作為の c（4 割が 1/2 ちょうど）、壁に接する 1/2 の列（滑り壁）。
密度比 1e3・1e6、両方の向き、両方の掃過の順序。(a) 自由流 max|u-U|、(b) 非一様な速度場の総運動量の保存。"""
import warnings, itertools
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
P = np.pi
N = 32

def field(kind):
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if kind == "diag":
        k = (i + j) % N
        return np.where(k == 8, 0.5, np.where(k == 24, 0.5, np.where((k > 8) & (k < 24), 1.0, 0.0)))
    if kind == "random":
        r = np.random.default_rng(5).random((N, N))
        return np.where(r < 0.4, 0.5, np.where(r < 0.7, 1.0, 0.0))
    return np.where(i < 10, 1.0, np.where(i == 10, 0.5, 0.0)) if kind == "column" else np.where(j == 0, 0.5, 0.0)

def run(kind, walls, rho1, rho2, half_rule, i0, uniform, steps=6):
    grid = CartesianGrid((N, N))
    ns = CenteredNS(grid, periodic=(1,) if walls else (0, 1), tolerance=1e-10)
    from basilisk_cupy.boundary import Periodic
    bc = BoundaryConditions().set((1, 0), Periodic()).set((1, 1), Periodic()) if walls else BoundaryConditions.periodic(2)
    f = CellField.from_interior(grid, field(kind), bc)
    tp = TwoPhase(ns, f, rho1=rho1, rho2=rho2, conserving=True, conserving_vof_half=half_rule)
    xv = np.arange(N + 1) / N
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    psi = (np.sin(P * X) * np.sin(2 * P * Y + 0.3) + 0.5 * np.sin(2 * P * X) * np.cos(2 * P * Y)) / (2 * P) if walls else (np.sin(2 * P * X) * np.sin(2 * P * Y) + 0.5 * np.cos(2 * P * X + 1) * np.sin(2 * P * Y + 0.4)) / (2 * P)
    uf = [(psi[:, 1:] - psi[:, :-1]) * N, -(psi[1:, :] - psi[:-1, :]) * N]
    for a in (0, 1):
        ns.uf.components[a][...] = uf[a]
    x = (np.arange(N) + 0.5) / N
    Xc, Yc = np.meshgrid(x, x, indexing="ij")
    U = (0.0, 0.37) if walls else (1.0, 0.37)            # 壁（x の両側）に沿う一様流
    for a in (0, 1):
        ns.u[a].interior = (U[a] + 0.0 * Xc) if uniform else (U[a] + 0.3 * np.sin(2 * P * (Xc + 2 * Yc) + a))
    ns.u.fill_ghosts()
    dt = 0.4 / N / max(np.abs(uf[0]).max(), np.abs(uf[1]).max())
    mom = lambda: np.array([float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[a].interior)).sum()) for a in (0, 1)])
    m0, err = mom(), 0.0
    for k in range(steps):
        ns.clock.dt, ns.clock.i = dt, i0 + k
        tp._vof(ns)
        if uniform:
            err = max(err, max(float(np.abs(to_numpy(ns.u[a].interior) - U[a]).max()) for a in (0, 1)))
    return err if uniform else float(np.abs(mom() - m0).max() / np.abs(m0).max())

if __name__ == "__main__":
  for kind, walls in (("diag", False), ("random", False), ("column", True), ("random", True)):
    for half_rule in (False, True):
        fs, mom = 0.0, 0.0
        for rho1, rho2 in ((1.0, 1e-3), (1e-3, 1.0), (1.0, 1e-6), (1e-6, 1.0)):
            for i0 in (0, 1):
                fs = max(fs, run(kind, walls, rho1, rho2, half_rule, i0, True))
                if not walls:
                    mom = max(mom, run(kind, walls, rho1, rho2, half_rule, i0, False))
        print("%-7s walls=%-5s %-22s free stream max|u-U| %.2e   momentum (relative change) %s" % (kind, walls, "vof.h rule" if half_rule else "default (PR)", fs, "%.2e" % mom if not walls else "-"), flush=True)

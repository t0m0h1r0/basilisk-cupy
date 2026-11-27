"""監修（C）：PR #154 で A が試していない境界。
(1) 動く壁（y の両側で u.t = U）：定常の製造解 ψ = U y + sin 2πx sin² πy / 2π、輸送だけ、次数。
(2) 流入・流出（左：法線・接線とも Dirichlet、右：Neumann。f も流入で Dirichlet）：非一様な流れで多ステップ、
    速度が有界か（相の体積がごく小さいセルの相ごとの速度 q_k/(c_k ρ_k) が暴れないか）。stale と比べる。
(3) 渦で引き伸ばす界面（密度比 1000、200 ステップ、輸送だけ）で、速度の最大が有界か。"""
import math, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic, Dirichlet, Neumann
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
P = np.pi
G, W = np.polynomial.legendre.leggauss(4); G, W = (G + 1) / 2, W / 2

def avg(fun, n):
    i = np.arange(n) / n
    return sum(wa * wb * fun(*np.meshgrid(i + a / n, i + b / n, indexing="ij")) for a, wa in zip(G, W) for b, wb in zip(G, W))

def faces(psi, n):
    xv = np.arange(n + 1) / n
    S = psi(*np.meshgrid(xv, xv, indexing="ij"))
    return (S[:, 1:] - S[:, :-1]) * n, -(S[1:, :] - S[:-1, :]) * n

def moving_wall(n, U, stale, cfl=0.4, T=0.25):
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0,), tangential={"bottom": Dirichlet(U), "top": Dirichlet(U)}, tolerance=1e-10)
    bc = BoundaryConditions().set((0, 0), Periodic()).set((0, 1), Periodic())
    tp = TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), bc), rho1=1.0, rho2=1.0, conserving=True,
                  conserving_source=True, conserving_stale_ghosts=stale)
    psi = lambda x, y: U * y + np.sin(2 * P * x) * np.sin(P * y) ** 2 / (2 * P)
    u = lambda x, y: U + np.sin(2 * P * x) * np.sin(2 * P * y) / 2
    v = lambda x, y: -np.cos(2 * P * x) * np.sin(P * y) ** 2
    ux = lambda x, y: P * np.cos(2 * P * x) * np.sin(2 * P * y)
    uy = lambda x, y: P * np.sin(2 * P * x) * np.cos(2 * P * y)
    vx = lambda x, y: 2 * P * np.sin(2 * P * x) * np.sin(P * y) ** 2
    vy = lambda x, y: -P * np.cos(2 * P * x) * np.sin(2 * P * y)
    g = [avg(lambda x, y: u(x, y) * ux(x, y) + v(x, y) * uy(x, y), n), avg(lambda x, y: u(x, y) * vx(x, y) + v(x, y) * vy(x, y), n)]
    q0 = [avg(u, n), avg(v, n)]
    uf = faces(psi, n)
    for a in (0, 1):
        ns.u[a].interior = grid.xp.asarray(q0[a]); ns.g[a].interior = grid.xp.asarray(g[a])
        ns.uf.components[a][...] = grid.xp.asarray(uf[a])
    ns.u.fill_ghosts()
    steps = int(math.ceil(T / (cfl / n / (abs(U) + 0.6)))); dt = T / steps
    for k in range(steps):
        ns.clock.dt, ns.clock.i = dt, k
        tp._vof(ns)
        for a in (0, 1):
            ns.u[a].interior = ns.u[a].interior + dt * ns.g[a].interior
        ns.u.fill_ghosts()
    return max(float(np.abs(to_numpy(ns.u[a].interior) - q0[a]).max()) for a in (0, 1))

def stretched(kind, stale, n=48, steps=300, ratio=1000.0):
    """kind "vortex": 滑り壁の箱の中の渦。"inflow": 左が流入（u, f とも Dirichlet）、右が流出。"""
    grid = CartesianGrid((n, n))
    if kind == "vortex":
        ns = CenteredNS(grid, tolerance=1e-10)
        bc = BoundaryConditions()
        psi = lambda x, y: np.sin(P * x) ** 2 * np.sin(P * y) ** 2 / P
        f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.75) ** 2, refine=8))
        u0 = (np.zeros((n, n)) + 0.3, np.zeros((n, n)) - 0.2)
    else:
        ns = CenteredNS(grid, periodic=(1,), normal={"left": Dirichlet(1.0), "right": Neumann(0.0)},
                        tangential={"left": Dirichlet(0.5), "right": Neumann(0.0)}, tolerance=1e-10)
        bc = BoundaryConditions().set((1, 0), Periodic()).set((1, 1), Periodic()).set((0, 0), Dirichlet(1.0))
        psi = lambda x, y: y + 0.12 * np.sin(2 * P * y) * np.sin(P * x) ** 2 / P + 0.08 * np.cos(2 * P * y) * np.sin(2 * P * x) / P
        f0 = to_numpy(fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
        u0 = (np.ones((n, n)), np.zeros((n, n)) + 0.5)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=1.0, rho2=1.0 / ratio, conserving=True,
                  conserving_stale_ghosts=stale)
    uf = faces(psi, n)
    for a in (0, 1):
        ns.u[a].interior = grid.xp.asarray(u0[a]); ns.uf.components[a][...] = grid.xp.asarray(uf[a])
    ns.u.fill_ghosts()
    dt = 0.4 / n / max(np.abs(uf[0]).max(), np.abs(uf[1]).max())
    lo = [float(a.min()) for a in u0]; hi = [float(a.max()) for a in u0]
    m, wisps = 0.0, 0
    for k in range(steps):
        ns.clock.dt, ns.clock.i = dt, k
        tp._vof(ns)
        uu = [to_numpy(ns.u[a].interior) for a in (0, 1)]
        m = max(m, max(float(np.abs(a).max()) for a in uu))
        fi = to_numpy(tp.f.interior)
        wisps = max(wisps, int(((fi > 1e-10) & (fi < 1e-6)).sum()))
        if not np.isfinite(m) or m > 1e6:
            break
    return m, wisps, k + 1

if __name__ == "__main__":
    for U in (1.0, -0.7):
        for stale in (False, True):
            r = [moving_wall(n, U, stale) for n in (32, 64, 128)]
            print("moving wall U=%-5s %-5s %s  ratios %.2f %.2f" % (U, "stale" if stale else "new", " ".join("%.3e" % e for e in r), r[0] / r[1], r[1] / r[2]), flush=True)
    for kind in ("vortex", "inflow"):
        for ratio in (1.0, 1000.0, 1e6):
            for stale in (False, True):
                m, w, k = stretched(kind, stale, ratio=ratio)
                print("%-7s ratio %-7g %-5s max|u| over %d steps %.4g   (max number of wisp cells %d)" % (kind, ratio, "stale" if stale else "new", k, m, w), flush=True)

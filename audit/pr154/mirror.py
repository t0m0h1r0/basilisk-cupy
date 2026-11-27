"""監修（C）：PR #154 の独立の検証（A の試験と別の判定）。滑り壁の領域は、鏡映した 2 倍の周期の領域の半分と
厳密に一致しなければならない（ゴーストセルが掃過の時点の場の鏡像なら）。二相（密度比 1000、壁に切られる液滴）、
非一様な流れ、面の値の源の項あり、両方の掃過の順序。conserving.h の定義（stale）は一致しないはず。
続けて、圧力の求解を含む連成のソルバーでも同じ比較をする。"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic, Dirichlet
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

P = np.pi
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "numpy"

def psi(x, y, walls):           # 壁の直線の上で 0、鏡映で奇
    if walls == "x":
        return (np.sin(P * x) * (1.0 + 0.5 * np.sin(2 * P * y + 0.7)) + 0.4 * np.sin(2 * P * x) * np.cos(2 * P * y)) / (2 * P)
    return (np.sin(P * x) * np.sin(P * y) + 0.4 * np.sin(2 * P * x) * np.sin(P * y) + 0.3 * np.sin(P * x) * np.sin(2 * P * y)) / (2 * P)

def build(walls, n, mirrored, stale, ratio=1000.0, coupled=False):
    """walls "x": 壁 x = 0, 1、y は周期。"xy": 全て壁。mirrored: 壁の方向を 2 倍にした周期の領域。"""
    wx, wy = True, walls == "xy"
    nn = (n * (2 if mirrored and wx else 1), n * (2 if mirrored and wy else 1))
    L = (2.0 if mirrored and wx else 1.0, 2.0 if mirrored and wy else 1.0)
    grid = CartesianGrid(nn, length=L, backend=BACKEND)
    xp = grid.xp
    per = tuple(a for a in (0, 1) if mirrored or (a == 1 and not wy))
    ns = CenteredNS(grid, periodic=per, tolerance=1e-12)
    bc = BoundaryConditions()
    for ax in per:
        for side in (0, 1):
            bc.set((ax, side), Periodic())
    def fold(x):                # [0, 2] の点を [0, 1] へ
        return np.where(x > 1.0, 2.0 - x, x)
    def phi(x, y):              # x = 0 の壁に切られる液滴（鏡映して対称）
        X = fold(x) if mirrored else x
        Y = (fold(y) if mirrored else y) if wy else y
        return 0.27 ** 2 - (X - 0.12) ** 2 - (Y - 0.55) ** 2
    f0 = to_numpy(fraction(CartesianGrid(nn, length=L), phi, refine=8))
    f0 = f0.reshape(f0.shape)[: ] if f0.shape == nn else f0
    f = CellField.from_interior(grid, xp.asarray(f0), bc)
    tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0 / ratio, conserving=True, conserving_source=True,
                  conserving_stale_ghosts=stale)
    h = 1.0 / n
    xv, yv = np.arange(nn[0] + 1) * h, np.arange(nn[1] + 1) * h
    XV, YV = np.meshgrid(xv, yv, indexing="ij")
    PS = psi(XV, YV, walls)       # 2 倍の領域では sin が奇の延長になる
    ufx, ufy = (PS[:, 1:] - PS[:, :-1]) / h, -(PS[1:, :] - PS[:-1, :]) / h
    xc, yc = (np.arange(nn[0]) + 0.5) * h, (np.arange(nn[1]) + 0.5) * h
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    sx = np.sin(P * X); cx = np.cos(P * X)
    if wy:
        u = np.sin(P * X) * np.cos(P * Y) + 0.3 * np.sin(2 * P * X) * np.cos(2 * P * Y) + 0.2 * sx
        v = -np.cos(P * X) * np.sin(P * Y) + 0.5 * np.sin(2 * P * Y) * np.cos(P * X) + 0.1 * np.sin(P * Y)
        gx, gy = np.sin(2 * P * X) * np.cos(P * Y), np.sin(P * Y) * np.cos(2 * P * X)
    else:
        u = np.sin(P * X) * np.cos(2 * P * Y + 0.3) + 0.2 * sx
        v = 0.4 + np.cos(P * X) * np.sin(2 * P * Y) + 0.3 * np.cos(2 * P * X)
        gx, gy = np.sin(2 * P * X) * np.cos(2 * P * Y), np.cos(P * X) * np.sin(2 * P * Y)
    return grid, ns, tp, (ufx, ufy), (u, v), (gx, gy)

def transport(walls, n, mirrored, stale, steps, i0):
    grid, ns, tp, uf, uv, g = build(walls, n, mirrored, stale)
    xp = grid.xp
    for a in (0, 1):
        ns.u[a].interior = xp.asarray(uv[a])
        ns.g[a].interior = xp.asarray(g[a])
        ns.uf.components[a][...] = xp.asarray(uf[a])
    ns.u.fill_ghosts()
    dt = 0.4 / n / max(np.abs(uf[0]).max(), np.abs(uf[1]).max())
    for k in range(steps):
        ns.clock.dt, ns.clock.i = dt, i0 + k
        tp._vof(ns)
    return [to_numpy(a) for a in (tp.f.interior, ns.u[0].interior, ns.u[1].interior)]

def coupled(walls, n, mirrored, stale, steps):
    grid, ns, tp, uf, uv, g = build(walls, n, mirrored, stale, coupled=True)
    xp = grid.xp
    for a in (0, 1):
        ns.u[a].interior = xp.asarray(uv[a])
    ns.start()
    for _ in range(steps):
        ns.step()
    return [to_numpy(a) for a in (tp.f.interior, ns.u[0].interior, ns.u[1].interior)], ns.clock.t

if __name__ == "__main__":
    n = 24
    for walls in ("x", "xy"):
        for stale in (False, True):
            for i0 in (0, 1):
                for steps in (1, 8):
                    a = transport(walls, n, False, stale, steps, i0)
                    b = transport(walls, n, True, stale, steps, i0)
                    d = [float(np.abs(p - q[:p.shape[0], :p.shape[1]]).max()) for p, q in zip(a, b)]
                    print("transport walls=%-2s %-5s first=%s steps=%d: max diff f %.1e  u %.1e  v %.1e" %
                          (walls, "stale" if stale else "new", "xy"[i0], steps, *d), flush=True)
    for walls in ("x", "xy"):
        for stale in (False, True):
            (a, ta), (b, tb) = coupled(walls, 16, False, stale, 4), coupled(walls, 16, True, stale, 4)
            d = [float(np.abs(p - q[:p.shape[0], :p.shape[1]]).max()) for p, q in zip(a, b)]
            print("coupled   walls=%-2s %-5s 4 steps (t %.6f / %.6f): max diff f %.1e  u %.1e  v %.1e" %
                  (walls, "stale" if stale else "new", ta, tb, *d), flush=True)

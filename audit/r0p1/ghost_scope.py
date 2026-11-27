"""監修（C）：ノート112 D 節（運動量のゴーストセル）の適用の範囲。
(a) 周期：ノート77 の設定（tg_cfl）で、ライブラリのままと周期の像。C 版（quadtree / multigrid）の値と比べる。
(b) 壁（対称の境界、Taylor–Green は [0,1]^2 の滑り壁の解でもある）：ライブラリのまま／q 自身に u の境界条件を
    適用したゴーストセル。A の修正案（周期の方向だけ）が壁で残す誤差を見る。"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic
from basilisk_cupy.interface.two_phase import TwoPhase
import basilisk_cupy.interface.two_phase as tp
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.convection_stability import conserving_options

SAVED = tp._MomentumField.fill_ghosts

def self_ghosts(field):                 # q 自身に、その境界条件（u のもの）を適用する
    return CellField.fill_ghosts(field)

def tg(variant, n, cfl, periodic, ghosts="library", t_end=0.5):
    tp._MomentumField.fill_ghosts = SAVED if ghosts == "library" else self_ghosts
    try:
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1) if periodic else (), tolerance=1e-8)
        if variant != "plain":
            bc = BoundaryConditions()
            if periodic:
                for ax in range(2):
                    for side in (0, 1):
                        bc.set((ax, side), Periodic())
            TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), bc), rho1=1.0, rho2=1.0, **conserving_options(variant))
        ns.cfl = cfl
        x, y = (to_numpy(a) for a in grid.meshgrid())
        u0 = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        v0 = -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        ns.u[0].interior = grid.xp.asarray(u0)
        ns.u[1].interior = grid.xp.asarray(v0)
        ns.start(events=[t_end])
        while ns.clock.t < t_end - 1e-12:
            ns.step()
        du = np.hypot(to_numpy(ns.u[0].interior) - u0, to_numpy(ns.u[1].interior) - v0)
        k = np.unravel_index(du.argmax(), du.shape)
        return float(du.max()), k
    finally:
        tp._MomentumField.fill_ghosts = SAVED

NS = (32, 64, 128)
for periodic in (True, False):
    print("== periodic" if periodic else "== walls (symmetry)")
    for variant, ghosts in (("plain", "library"), ("conserving", "library"), ("conserving", "self"),
                            ("conserving-src", "library"), ("conserving-src", "self")):
        r = [tg(variant, n, 0.5, periodic, ghosts) for n in NS]
        print("%-15s %-8s" % (variant, ghosts), " ".join("%.4e" % a for a, _ in r), " argmax", [tuple(int(i) for i in k) for _, k in r], flush=True)

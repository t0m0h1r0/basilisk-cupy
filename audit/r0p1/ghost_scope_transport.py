"""監修（C）：ノート112 D 節（運動量のゴーストセル）が壁（対称の境界）でも同じ誤差を作るか。
圧力の求解を使わず、厳密な面の速度と厳密な g で、TwoPhase._vof と同じ手順（q のゴーストセルを _MomentumField で作る、
輸送、u = q + dt g）を x 方向の運動量について繰り返す。Taylor–Green は [0,1]^2 の滑り壁の解でもある。
  ghosts = "library"：_MomentumField.fill_ghosts（開始時の u から）
  ghosts = "self"   ：q 自身に u の境界条件（周期なら周期の像、壁なら法線成分は反対称・接線成分は対称）"""
import math, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic, vector_bcs
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection
import basilisk_cupy.interface.two_phase as tp
from cases.r0p1_local_error import cell_average, face_velocity, predicted_terms

def run(n, cfl, periodic, ghosts, source=True, t_end=0.25):
    grid = CartesianGrid((n, n)); xp = grid.xp
    T = predicted_terms()
    u0, g = cell_average(T["u"], n), cell_average(T["gx"], n)
    ubc = vector_bcs(2, (0, 1) if periodic else ())[0]
    fbc = BoundaryConditions.periodic(2) if periodic else BoundaryConditions()
    f = CellField.from_interior(grid, np.zeros((n, n)), fbc)
    u = CellField.from_interior(grid, u0.copy(), ubc)
    uf = FaceField(grid, name="u")
    a, b = face_velocity(n)
    uf.components[0][...] = xp.asarray(a); uf.components[1][...] = xp.asarray(b)
    steps = int(math.ceil(t_end / (cfl / n) - 1e-9)); dt = t_end / steps
    for i in range(steps):
        q = tp._MomentumField(grid, bc=ubc, name="q2").bind(f, u, 1.0, True)
        if ghosts == "self":
            q.fill_ghosts = lambda q=q: CellField.fill_ghosts(q)
        q.interior = u.interior
        itf = VOFInterface(f, [VOFTracer(q, inverse=True, source=xp.asarray(g) if source else None)])
        vof_advection([itf], uf, dt, i)
        u.interior = q.interior + dt * xp.asarray(g)
        u.fill_ghosts()
    e = np.abs(to_numpy(u.interior) - u0)
    k = np.unravel_index(e.argmax(), e.shape)
    inner = e[4:-4, 4:-4].max()
    return float(e.max()), float(inner), tuple(int(v) for v in k)

for periodic in (True, False):
    print("== periodic" if periodic else "== walls (symmetry)")
    for ghosts in ("library", "self"):
        for n in (32, 64, 128, 256):
            r = run(n, 0.4, periodic, ghosts)
            print("src=True ghosts=%-8s n=%3d cfl=0.4: max %.3e  (4 セルより内側 %.3e)  argmax %s" % ((ghosts, n) + r), flush=True)

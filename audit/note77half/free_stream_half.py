"""監修（C）：ノート77 は「c = 1/2 ちょうどのセルでは濃度が 1.7e-2 ずれる（C 版も同じ）。complementary=True で直るが
既定は Basilisk のまま」と記録している。R0 の必須の要件「一様な速度の厳密な保存（自由流）」に、運動量でどう出るかを測る。
一様な速度の運動量を、一様でない発散 0 の面の速度で運ぶ（TwoPhase(conserving=True)._vof）。界面は格子に平行。
x0 は界面の位置（セル幅の単位の小数部分 0.5 で c = 1/2 ちょうど）。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.two_phase import TwoPhase
import basilisk_cupy.interface.two_phase as tpm
from basilisk_cupy.time.basilisk_centered import CenteredNS
P = np.pi
U = (1.0, 0.37)

def run(n, frac, rho1, rho2, complementary, steps):
    saved = tpm.VOFTracer
    if complementary:
        tpm.VOFTracer = lambda *a, **k: saved(*a, **{**k, "complementary": True})
    try:
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        bc = BoundaryConditions.periodic(2)
        i = np.arange(n)
        a, b = (n // 4 + frac) , (3 * n // 4 + frac)              # 相 1 は a <= x/h < b の帯
        col = np.clip(np.minimum(i + 1, b) - np.maximum(i, a), 0.0, 1.0)
        f = CellField.from_interior(grid, np.repeat(col[:, None], n, axis=1), bc)
        tp = TwoPhase(ns, f, rho1=rho1, rho2=rho2, conserving=True)
        xv = i[:, None] / n
        xv = np.arange(n + 1) / n
        X, Y = np.meshgrid(xv, xv, indexing="ij")
        psi = np.sin(2 * P * X) * np.sin(2 * P * Y) / (2 * P) + np.cos(2 * P * X) * np.sin(2 * P * Y) / (4 * P)
        ufx, ufy = (psi[:, 1:] - psi[:, :-1]) * n, -(psi[1:, :] - psi[:-1, :]) * n
        ns.uf.components[0][...] = ufx
        ns.uf.components[1][...] = ufy
        for ax in (0, 1):
            ns.u[ax].interior = U[ax] + 0.0 * ns.u[ax].interior
        ns.u.fill_ghosts()
        dt = 0.4 / n / max(np.abs(ufx).max(), np.abs(ufy).max())
        err = []
        for k in range(steps):
            ns.clock.dt, ns.clock.i = dt, k
            tp._vof(ns)
            err.append(max(float(np.abs(to_numpy(ns.u[ax].interior) - U[ax]).max()) for ax in (0, 1)))
        return err[0], max(err)
    finally:
        tpm.VOFTracer = saved

for n in (32, 64, 128):
    for frac in (0.5, 0.5 + 1e-9, 0.3):
        for rho1, rho2 in ((1.0, 1e-3), (1e-3, 1.0), (1.0, 1.0)):
            for comp in (False, True):
                e1, em = run(n, frac, rho1, rho2, comp, 10)
                print("n=%3d x0 の小数部=%-12r rho1=%-6g rho2=%-6g %-13s  max|u-U|: 1 ステップ %.2e, 10 ステップの最大 %.2e"
                      % (n, frac, rho1, rho2, "complementary" if comp else "default", e1, em), flush=True)

"""切る流れ（条件 6 の非粘性の格子に平行なせん断流）と、渦の中の液滴で、PR #192 のライブラリの選択肢と、監査済みのフック
（e-trials の cases、θ=3）が一致するか。引数 lib：PR のツリーで選択肢。hook：e-trials のツリーでフック。"""
import sys, math, warnings, contextlib
import numpy as np
warnings.filterwarnings("ignore")
mode, path = sys.argv[1], sys.argv[2]
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
kw, ctx = {}, contextlib.nullcontext
if mode == "lib":
    kw = dict(conserving_consistent=True, conserving_slope_limit=3.0)
else:
    from cases import r0_aligned_free_stream as m
    m.THETA[0] = 3.0
    ctx = lambda: m.scheme(True, "limited")
out, N, k = {}, 64, 4 * math.pi
for case in ("shear", "vortex"):
    with ctx():
        grid = CartesianGrid((N, N))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        x = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(x, x, indexing="ij"); yl = x - 0.5 / N
        if case == "shear":
            f0 = np.tile(((x > 0.25) & (x < 0.75)).astype(float), (N, 1))
            u0 = np.tile(0.5 * (np.cos(k * (yl - 0.25)) - np.cos(k * (yl + 1 / N - 0.25))) / (k / N), (N, 1)); v0 = np.ones((N, N))
        else:
            f0 = np.asarray(fraction(grid, lambda a, b: 0.15 - np.hypot(a - 0.5, b - 0.3)))
            u0 = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y); v0 = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, **kw)
        ns.cfl = 0.5
        ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.start(events=[1.0])
        while ns.clock.t < 1.0 - 1e-12:
            ns.step()
        out[case] = np.stack([ns.u[0].interior, ns.u[1].interior, tp.f.interior])
        out[case + "_cuts"] = np.array(getattr(sys.modules.get("cases.r0_aligned_free_stream"), "COUNT", [0, 0]))
np.savez(path, **out)

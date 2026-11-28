"""監修（C）：面に揃った界面を一様流で運ぶと、源の項なしの基準でも max|u-U| が指数的に育つ（ノート117 の検証で発見）。
c = 1/2 の欠陥（#156、PR #160）が原因かを、PR #160 のブランチの上で確かめる。CenteredNS + TwoPhase だけを使う。"""
import os, math, warnings, inspect
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    shape, U, half_rule, cfl, ratio, steps, n = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if shape == "square":
        f0 = (((i >= 3 * n // 8) & (i < 5 * n // 8)) & ((j >= 3 * n // 8) & (j < 5 * n // 8))).astype(float)
    elif shape == "column":
        f0 = ((i >= n // 4) & (i < n // 2)).astype(float)
    elif shape == "column+0.3":                          # 面から 0.3 セルずらした帯
        f0 = np.clip(np.minimum(i + 1, n // 2 + 0.3) - np.maximum(i, n // 4 + 0.3), 0, 1)
    else:
        f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
    kw = {"conserving_vof_half": half_rule} if "conserving_vof_half" in inspect.signature(TwoPhase.__init__).parameters else {}
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, conserving=True, **kw)
    ns.cfl = cfl
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    ns.start()
    m, half = 0.0, 0
    for s in range(steps):
        ns.step()
        m = max(m, max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2)))
        half += int((to_numpy(tp.f.interior) == 0.5).sum())
        if not math.isfinite(m) or m > 10:
            break
    return "%-10s U=%-11s %-22s cfl=%-5g ratio=%-6g N=%d: max|u-U| %.2e after %d steps  (cell-steps with c == 0.5 exactly: %d)" % (
        shape, U, "vof.h rule (c=1/2)" if half_rule else "default of PR #160", cfl, ratio, n, m, s + 1, half)

if __name__ == "__main__":
    J = [(s, U, h, 0.5, r, 600, 32) for s in ("square", "column") for U in ((1.0, 0.0), (1.0, 0.37)) for h in (True, False) for r in (1.0, 1e3, 1e6)]
    J += [(s, (1.0, 0.0), h, c, 1e3, 600, 32) for s in ("square", "column") for h in (True, False) for c in (0.45, 0.25)]
    J += [(s, (1.0, 0.0), h, 0.5, 1e3, 600, 32) for s in ("column+0.3", "drop") for h in (True, False)]
    J += [("column", (1.0, 0.0), h, 0.5, 1e3, 1200, 64) for h in (True, False)]
    with mp.get_context("spawn").Pool(10, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

"""監修（C）：PR #148 を壊しにいく。有理比の界面位置（近接した対が出やすい）、密度比 1e6、無作為の位置。"""
import numpy as np
from audit.pr148._stokes_reference import rates
from cases.viscous_stress import mus, stokes_rate
from cases.viscous_walls import wall_rates
rng = np.random.default_rng(7)
worst = 0.0; nfix = 0
for kind, ratio in (("nu", 1e3), ("nu", 1e6), ("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0)):
    mH, mL = mus(ratio, kind)
    for a in [0.4, 1 / 3, 0.25, 0.5, 7 / 19] + list(rng.uniform(0.05, 0.95, 4)):
        lams, _ = wall_rates(mH, mL, ratio, a, which=50)
        C = rates([(a, 1.0, mH), (1 - a, 1 / ratio, mL)], "wall", n=140, k=50)
        d = np.max(np.abs(np.array(lams) - C) / C); worst = max(worst, d)
        if d > 1e-7: print("WALL MISMATCH", kind, ratio, a, d)
    for a, b in [(0.25, 0.75), (0.2, 0.6), (0.1, 0.9), (0.3, 0.3 + 1 / 3)] + [tuple(sorted(rng.uniform(0.05, 0.95, 2))) for _ in range(4)]:
        LH = b - a
        if LH < 0.05 or LH > 0.95: continue
        A = np.array(stokes_rate(mH, mL, ratio, a, b, which=40))
        C = rates([(LH, 1.0, mH), (1 - LH, 1 / ratio, mL)], "periodic", n=140, k=60)
        C = np.sort(np.abs(C))[1:41]                     # 零モードを除く
        d = np.max(np.abs(A - C) / C); worst = max(worst, d)
        if d > 1e-6: print("PERIODIC MISMATCH", kind, ratio, a, b, d)
print("worst relative difference", worst)

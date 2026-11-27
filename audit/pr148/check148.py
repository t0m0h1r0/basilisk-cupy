"""監修（C）：PR #148 の固有値を、層ごとの Chebyshev 選点法（過去の監修の audit/stokes_reference.py）で照合する。"""
import math, sys
import numpy as np
from audit.pr148._stokes_reference import rates
from cases.viscous_stress import mus, stokes_rate, dirichlet_count
from cases.viscous_walls import wall_rates

# 1. 周期、動粘度一定、密度比 1e3、a = 0.2613、b = 0.7137
r = 1e3
mH, mL = mus(r, "nu")
LH = 0.7137 - 0.2613
A = stokes_rate(mH, mL, r, 0.2613, 0.7137, which=43)
for n in (100, 140):
    C = rates([(LH, 1.0, mH), (1 - LH, 1 / r, mL)], "periodic", n=n, k=60)
    C = C[C > 1e-6]                                    # 零モードを除く
    print("periodic n=%d: Chebyshev 39..44 番目" % n, ["%.3f" % v for v in C[38:44]], " lambda<=18959 の個数", int((C <= 18959).sum()))
print("A: 39..43 番目", ["%.3f" % v for v in A[38:43]], " 最初の 43 個の最大相対差 %.2e" % np.max(np.abs(np.array(A) - C[:43]) / C[:43]))

# 2. 壁：ノート103 の 12 条件、最初の 60 個
worst = 0.0
for kind, ratio in (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3)):
    mH, mL = mus(ratio, kind)
    for off in (0.0, 0.0071, 0.0193):
        a = 0.3613 + off
        lams, _ = wall_rates(mH, mL, ratio, a, which=60)
        C = rates([(a, 1.0, mH), (1 - a, 1 / ratio, mL)], "wall", n=140, k=60)
        d = np.max(np.abs(np.array(lams) - C) / C)
        worst = max(worst, d)
        gaps = np.diff(np.sqrt(C)).min()
        print("wall %-9s ratio %-6g a=%.4f: 60 個の最大相対差 %.2e, sqrt(lambda) の最小の間隔 %.4f" % (kind, ratio, a, d, gaps))
print("worst", worst)

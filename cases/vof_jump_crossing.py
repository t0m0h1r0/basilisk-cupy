"""Crossing ``theta`` of the VOF ghost-fluid jump (``TwoPhaseVOFJump``, docs/notes/64) against the exact interface:
the line sum of f against the PLIC line, on the initial capillary wave and drop of the paper (N = 32, 64). Faces whose
exact level has the same sign at both centres (cut by ``f > 1/2`` only) are counted and left out.

    python -m cases.vof_jump_crossing
"""

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from cases.paper_benchmarks import setup  # noqa: E402

def exact_theta(level, xl, yl, xr, yr):
    # crossing of the level function along the segment between the two centres (bisection)
    a, b = np.zeros_like(xl), np.ones_like(xl)
    fl = level(xl, yl)
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = level(xl + m * (xr - xl), yl + m * (yr - yl))
        left = np.sign(fm) == np.sign(fl)
        a, b = np.where(left, m, a), np.where(left, b, m)
    return 0.5 * (a + b)

for case, n in (("capwave", 32), ("capwave", 64), ("drop", 32), ("drop", 64)):
    ns, info = setup(case, n, interface="vof-gfm")
    tp = ns.tp
    g = ns.grid
    h = [float(v) for v in g.h]
    x, y = [np.asarray(v) for v in g.meshgrid()]
    if case == "capwave":
        k = 2 * math.pi / 0.01; level = lambda X, Y: 0.01 + 2e-4 * np.cos(k * X) - Y
    else:
        level = lambda X, Y: 1.0 - ((X - 0.01) / 5.5e-3) ** 2 - ((Y - 0.01) / 4.5e-3) ** 2
    for mode in ("sum", "plic"):
      tp.crossing = mode
      for ax in range(2):
        hl, hr, cut, theta = [np.asarray(v) for v in tp._cut(ax)]
        # face i between cells i-1 and i along ax; faces 1..n-1 interior
        sl = [slice(None), slice(None)]; sl[ax] = slice(1, -1)
        c = cut[tuple(sl)]; th = theta[tuple(sl)]
        lo = [slice(None), slice(None)]; lo[ax] = slice(0, -1)
        hi = [slice(None), slice(None)]; hi[ax] = slice(1, None)
        xl, yl, xr, yr = x[tuple(lo)], y[tuple(lo)], x[tuple(hi)], y[tuple(hi)]
        te = exact_theta(level, xl, yl, xr, yr)
        real = np.sign(level(xl, yl)) != np.sign(level(xr, yr))
        print("   cut but same exact sign:", int((c & ~real).sum()), " real crossing not cut:", int((real & ~c).sum()))
        c = c & real
        err = np.abs(th - te)[c]
        if not c.any(): print(case, n, ax, "no cut"); continue
        print(f"{mode:4s} {case} N {n} ax {ax}: cut {c.sum():3d} |theta err| max {err.max():.3f} mean {err.mean():.3f}", flush=True)

"""各形の安定の上限 z*（スペクトル半径 <= 1 が 0 から続く範囲）を、利得 G ごとに。"""
import numpy as np, sys
sys.path.insert(0, "audit/d179")
from collocated_gain import radius
zz = np.linspace(1e-3, 5, 20000)
print("G      " + "  ".join("%-9s" % n for n in ("c", "a:E1", "a:E1+avg", "d1", "b1")))
for G in (1.0, 0.99, 0.9, 0.75, 0.5, 0.25, 0.0):
    row = []
    for n in ("c", "a:E1", "a:E1+avg", "d1", "b1"):
        bad = next((z for z in zz if radius(n, z, G) > 1 + 1e-7), None)
        row.append(">5" if bad is None else "%.3f" % bad)
    print("%-6g " % G + "  ".join("%-9s" % v for v in row))

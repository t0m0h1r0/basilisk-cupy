"""#199 の 1：fraction() の誤りの範囲。(a) 滑らかな円の収束の次数、(b) 格子点を通る直線（退化）、(c) 折れ（角）と退化の分離。"""
import sys, math
import numpy as np
sys.path.insert(0, "audit/i199")
from exact import halfplane, disk
from basilisk_cupy import CartesianGrid
from basilisk_cupy.interface.fractions import fraction

def err(f, e, h):
    d = np.abs(f - e)
    return "max %.1e  L1(area) %.1e  cells>1e-3: %d" % (d.max(), d.sum() * h * h, int((d > 1e-3).sum()))

print("(a) 円 R=0.2 中心 (0.5123, 0.4871)：滑らか、退化なし")
for refine in (1, 8):
    prev = None
    for n in (32, 64, 128, 256):
        g = CartesianGrid((n, n))
        f = np.asarray(fraction(g, lambda x, y: 0.2 ** 2 - (x - 0.5123) ** 2 - (y - 0.4871) ** 2, refine=refine))
        e = disk(n, n, 0.5123, 0.4871, 0.2, 1.0 / n)
        d = np.abs(f - e).max()
        print("   refine=%d N=%-3d %s%s" % (refine, n, err(f, e, 1.0 / n), "" if prev is None else "  order(max) %.2f" % math.log2(prev / d)))
        prev = d

print("(b) 直線 y < x + c が格子点を通る場合（45 度）。c を格子点の上に置き、丸めの ±1e-17 を与える")
n = 32
g = CartesianGrid((n, n))
for name, c in (("c = 0（厳密）", 0.0), ("c = 0 - 5.6e-17", -5.551115123125783e-17), ("c = 0.25 - 5.6e-17", 0.25 - 5.551115123125783e-17), ("c = 0.0137（点を通らない）", 0.0137)):
    for refine in (1, 8):
        f = np.asarray(fraction(g, lambda x, y: x + c - y, refine=refine))
        e = halfplane(n, -1.0, 1.0, c)
        print("   %-24s refine=%d  %s" % (name, refine, err(f, e, 1.0 / n)))

print("(c) 折れ：角を持つ正方形（辺 0.4、中心をずらす）と、45 度の角（ひし形）。格子点を通らない")
for name, cx, cy in (("square (smooth center)", 0.5123, 0.4871), ("square (center 0.5, 0.5)", 0.5, 0.5)):
    for refine in (1, 8):
        for n in (32, 64, 128):
            g = CartesianGrid((n, n))
            a = 0.2
            f = np.asarray(fraction(g, lambda x, y: np.minimum(a - np.abs(x - cx), a - np.abs(y - cy)), refine=refine))
            h = 1.0 / n
            xl = np.arange(n) * h
            ox = np.clip(np.minimum(xl + h, cx + a) - np.maximum(xl, cx - a), 0, None) / h
            oy = np.clip(np.minimum(xl + h, cy + a) - np.maximum(xl, cy - a), 0, None) / h
            e = ox[:, None] * oy[None, :]
            print("   %-24s refine=%d N=%-3d %s" % (name, refine, n, err(f, e, h)))

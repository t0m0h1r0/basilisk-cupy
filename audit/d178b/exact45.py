"""45° の帯の厳密な体積分率（解析式）と、fraction(refine=8) の差。厳密な f で位置を作り直す。"""
import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
spec = importlib.util.spec_from_file_location("d178", "design/checks/178_heights.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

def area_below(s, h):
    """area of {(a, b) in [0, h]^2 : a + b < s}."""
    s = np.clip(s, 0.0, 2 * h)
    return np.where(s <= h, 0.5 * s * s, h * h - 0.5 * (2 * h - s) ** 2)

def exact_band(n, off):
    h = 1.0 / n
    i = np.arange(n)[:, None] * h; j = np.arange(n)[None, :] * h
    s0 = i + j                                    # x + y at the lower-left corner
    c = np.zeros((n, n))
    for k in range(-3, 4):                         # phase 1: off + k <= x + y < off + k + 1/2
        c += (area_below(off + k + 0.5 - s0, h) - area_below(off + k - s0, h)) / (h * h)
    return c

bc = D.BoundaryConditions.periodic(2)
for n in (32, 64):
    for off in (0.0, 0.25 / n, 0.5 / n, 0.0137):
        grid, c, r = D.band(n, 1, 1, off)
        e = exact_band(n, off)
        for name, f in (("refine=8", c), ("exact", e)):
            p = D.position(f, bc, grid, (1 / r, 1 / r), (0.0, 0.0), variant="library")
            ok = p < D.H.NODATA / 2
            d = np.where(ok, (((p * r - off + 0.25) % 0.5) - 0.25) / r * n, 0.0)
            force = D.face_force(np.where(ok, d, D.H.NODATA), f, True)
            print("n=%-3d off=%.2f h  %-8s: |f - exact| %.1e   position off %.3f cells  face force %.2e" % (n, off * n, name, np.abs(f - e).max(), np.abs(d).max(), force))

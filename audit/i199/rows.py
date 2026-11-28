"""#199 の 2：design/168 の各行の初期化を、fraction()（refine=1 / 8）と厳密値で比べる。最大のセルの誤差と、面積の相対誤差。
平面（R5・R10）は監修の多角形の切り取り、円（R2 など）は解析式、余弦（R9）は求積。"""
import sys, math
import numpy as np
sys.path.insert(0, "audit/i199")
from exact import halfplane, disk, under_curve
from basilisk_cupy import CartesianGrid
from basilisk_cupy.interface.fractions import fraction

def cmp(f, e):
    return float(np.abs(f - e).max()), float(abs(f.sum() - e.sum()) / max(e.sum(), 1e-300))

def worst(rows):
    return "max cell %.1e  area rel %.1e" % (max(r[0] for r in rows), max(r[1] for r in rows))

def circle_row(name, R, centers, Ns, length=(1.0, 1.0), periodic=True):
    for refine in (1, 8):
        rows = []
        for n in Ns:
            nx, ny = n, int(round(n * length[1]))
            g = CartesianGrid((nx, ny), length=length)
            h = 1.0 / n
            for (ax, ay) in centers(h):
                f = np.asarray(fraction(g, lambda x, y: R * R - (x - ax) ** 2 - (y - ay) ** 2, refine=refine))
                rows.append(cmp(f, disk(nx, ny, ax, ay, R, h, periodic)))
        print("%-44s refine=%d: %s" % (name, refine, worst(rows)))

five = lambda h: [(0.5 + a * h, 0.5 + b * h) for a, b in ((0, 0), (0.5, 0), (0.3, 0.2), (0.25, 0.25), (0.1, 0.4))]
circle_row("R2 drop R=.15 (0.5,0.5)", 0.15, lambda h: [(0.5, 0.5)], (32, 64, 128))
circle_row("R3 vortex drop R=.15 (0.5,0.3)", 0.15, lambda h: [(0.5, 0.3)], (32, 64, 128))
circle_row("R6/R7 static drop R=.2, 5 centres", 0.2, five, (32, 64, 128))
circle_row("A7 translating drop R=.2", 0.2, lambda h: [(0.5, 0.5)], (32, 64))
circle_row("R11 bubble R=.25 (0.5,0.5), 1x2", 0.25, lambda h: [(0.5, 0.5)], (32, 64, 128), length=(1.0, 2.0), periodic=False)

# R8 shear layer: band 1/4 < y < 3/4 (faces)
for refine in (1, 8):
    rows = []
    for n in (64, 128):
        g = CartesianGrid((n, n))
        f = np.asarray(fraction(g, lambda x, y: np.minimum(y - 0.25, 0.75 - y), refine=refine))
        yl = np.arange(n) / n
        e = np.tile(np.clip(np.minimum(yl + 1 / n, 0.75) - np.maximum(yl, 0.25), 0, None) * n, (n, 1))
        rows.append(cmp(f, e))
    print("%-44s refine=%d: %s" % ("R8 band 1/4<y<3/4", refine, worst(rows)))

# R9 capillary wave y < 0.01 cos 2 pi (x - x0) + y0 on [0,1] x [-0.5, 0.5]
for refine in (1, 8):
    rows = []
    for n in (32, 64, 128):
        h = 1.0 / n
        g = CartesianGrid((n, n), origin=(0.0, -0.5))
        for x0, y0 in ((0, 0), (0.25, 0), (0.5, 0), (0, 0.3), (0.5, 0.5)):
            fn = lambda x: 0.01 * math.cos(2 * math.pi * (x - x0 * h)) + y0 * h
            f = np.asarray(fraction(g, lambda x, y: 0.01 * np.cos(2 * math.pi * (x - x0 * h)) + y0 * h - y, refine=refine))
            rows.append(cmp(f, under_curve(n, fn)))
    print("%-44s refine=%d: %s" % ("R9 capillary wave, 5 shifts", refine, worst(rows)))

# R10 planes parallel to the grid: face, centre, 0.3h
for refine in (1, 8):
    rows = []
    for n in (32, 64, 128):
        g = CartesianGrid((n, n))
        h = 1.0 / n
        for off in (0.0, 0.5, 0.3):
            yi = 0.5 + off * h
            f = np.asarray(fraction(g, lambda x, y: yi - y, refine=refine))
            rows.append(cmp(f, np.tile(np.clip((yi - np.arange(n) * h) / h, 0, 1), (n, 1))))
    print("%-44s refine=%d: %s" % ("R10/R5(a) horizontal planes", refine, worst(rows)))

# R5(b) 1850 scan: how many configurations differ
rng = np.random.default_rng(0)
J = [(n, float(th), float(off)) for n in (32, 64) for th in np.arange(0, 90.01, 2.5) for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)])]
bad = []
for n, th, off in J:
    t = math.radians(th); nx, ny = math.sin(t), math.cos(t)
    g = CartesianGrid((n, n))
    f = np.asarray(fraction(g, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off))))
    e = halfplane(n, nx, ny, nx * 0.5 + ny * (0.5 + off))
    d = float(np.abs(f - e).max())
    if d > 1e-8:
        bad.append((n, th, off, d))
print("R5(b) %d plane configurations (refine=1): differ by > 1e-8 in %d: %s" % (len(J), len(bad), [(b[0], b[1], "%.4f" % b[2], "%.2f" % b[3]) for b in bad[:6]]))

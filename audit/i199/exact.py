"""厳密な体積分率（独立の実装）：多角形（平面・箱）、円と長方形の交わり（解析式）、y < g(x)（区分的な求積）。"""
import math
import numpy as np
from scipy.integrate import quad


def halfplane(n, nx, ny, d):
    """{nx x + ny y < d} の [0, 1]^2 の格子の体積分率（多角形の切り取り）。"""
    h = 1.0 / n
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x0, y0 = i * h, j * h
            poly = [(0.0, 0.0), (h, 0.0), (h, h), (0.0, h)]
            e = d - nx * x0 - ny * y0
            out = []
            for k in range(4):
                a, b = poly[k], poly[(k + 1) % 4]
                sa, sb = nx * a[0] + ny * a[1] - e, nx * b[0] + ny * b[1] - e
                if sa <= 0:
                    out.append(a)
                if sa * sb < 0:
                    t = sa / (sa - sb)
                    out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if len(out) >= 3:
                s = sum(out[k][0] * out[(k + 1) % len(out)][1] - out[(k + 1) % len(out)][0] * out[k][1] for k in range(len(out)))
                c[i, j] = min(max(abs(s) / 2 / (h * h), 0.0), 1.0)
    return c


def _disk_prim(x, R):
    x = min(max(x, -R), R)
    return 0.5 * (x * math.sqrt(max(R * R - x * x, 0.0)) + R * R * math.asin(x / R))


def disk_cell(x0, x1, y0, y1, cx, cy, R):
    """area of {(x - cx)^2 + (y - cy)^2 < R^2} in [x0, x1] x [y0, y1] (closed form, split at the crossings)."""
    a, b = max(x0 - cx, -R), min(x1 - cx, R)
    if a >= b:
        return 0.0
    v0, v1 = y0 - cy, y1 - cy
    cuts = {a, b}
    for v in (v0, v1):
        if abs(v) < R:
            s = math.sqrt(R * R - v * v)
            for z in (-s, s):
                if a < z < b:
                    cuts.add(z)
    cuts = sorted(cuts)
    area = 0.0
    for p, q in zip(cuts[:-1], cuts[1:]):
        m = 0.5 * (p + q)
        s = math.sqrt(max(R * R - m * m, 0.0))
        top, bot = min(v1, s), max(v0, -s)
        if top <= bot:
            continue
        dq = _disk_prim(q, R) - _disk_prim(p, R)
        ti = dq if s < v1 else v1 * (q - p)
        bi = -dq if -s > v0 else v0 * (q - p)
        area += ti - bi
    return area


def disk(nx_, ny_, cx, cy, R, h, periodic=True):
    c = np.zeros((nx_, ny_))
    for i in range(nx_):
        for j in range(ny_):
            x0, y0 = i * h, j * h
            shifts = (-1, 0, 1) if periodic else (0,)
            c[i, j] = sum(disk_cell(x0, x0 + h, y0, y0 + h, cx + sx * nx_ * h, cy + sy * ny_ * h, R) for sx in shifts for sy in shifts) / (h * h)
    return c


def under_curve(n, g, y_origin=-0.5, ny=None):
    """{y < g(x)}, y from y_origin; adaptive quadrature of the clipped height."""
    h = 1.0 / n
    ny = ny or n
    c = np.zeros((n, ny))
    for i in range(n):
        for j in range(ny):
            y0 = y_origin + j * h
            lo = g(i * h + 0.5 * h)
            if lo - y0 > 2 * h:
                c[i, j] = 1.0
                continue
            if y0 - lo > 2 * h:
                continue
            c[i, j] = quad(lambda x: min(max(g(x) - y0, 0.0), h), i * h, (i + 1) * h, epsabs=1e-15, epsrel=1e-14, limit=200)[0] / (h * h)
    return c

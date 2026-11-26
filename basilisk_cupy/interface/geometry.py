"""Basic VOF geometry of a unit square cell cut by a line (Basilisk ``geometry.h``, 2-D).

The cell is ``[-1/2, 1/2]^2``; the interface is ``n_x x + n_y y = alpha`` and the volume
fraction ``c`` is the part of the cell opposite to ``n`` ("inside"). The functions work on
arrays (``xp`` = numpy or cupy) and keep Basilisk's operation order, so that results agree
to round-off with the C code.
"""

from __future__ import annotations

from ..backend import array_module


def _clamp(xp, v, lo, hi):
    return xp.minimum(xp.maximum(v, lo), hi)


def line_alpha(c, nx, ny):
    """Intercept ``alpha`` of the line with normal ``(nx, ny)`` cutting volume fraction ``c``."""
    xp = array_module(c)
    n1, n2 = xp.abs(nx), xp.abs(ny)
    a, b = xp.minimum(n1, n2), xp.maximum(n1, n2)  # swap so that n1 <= n2
    c = _clamp(xp, c, 0.0, 1.0)
    v1 = a / 2.0
    alpha = xp.where(c <= v1 / b, xp.sqrt(2.0 * c * a * b),
                     xp.where(c <= 1.0 - v1 / b, c * b + v1, a + b - xp.sqrt(xp.maximum(2.0 * a * b * (1.0 - c), 0.0))))
    alpha = xp.where(nx < 0.0, alpha + nx, alpha)
    alpha = xp.where(ny < 0.0, alpha + ny, alpha)
    return alpha - (nx + ny) / 2.0


def line_area(nx, ny, alpha):
    """Volume fraction of the unit cell below the line ``(nx, ny) . x = alpha``."""
    xp = array_module(alpha)
    alpha = alpha + (nx + ny) / 2.0
    alpha = xp.where(nx < 0.0, alpha - nx, alpha)
    nx = xp.abs(nx)
    alpha = xp.where(ny < 0.0, alpha - ny, alpha)
    ny = xp.abs(ny)
    one = xp.ones_like(alpha)
    safe = lambda d: xp.where(d == 0.0, one, d)  # noqa: E731  (the guarded branches never use it)
    v = alpha * alpha
    a = alpha - nx
    v = xp.where(a > 0.0, v - a * a, v)
    a = alpha - ny
    v = xp.where(a > 0.0, v - a * a, v)
    area = xp.where(nx < 1e-10, alpha / safe(ny), xp.where(ny < 1e-10, alpha / safe(nx), v / safe(2.0 * nx * ny)))
    area = _clamp(xp, area, 0.0, 1.0)
    area = xp.where(alpha <= 0.0, 0.0, area)
    return xp.where(alpha >= nx + ny, 1.0, area)


def rectangle_fraction(nx, ny, alpha, ax, ay, bx, by):
    """Fraction of the rectangle ``[a, b]`` (cell coordinates) inside the interface ``(n, alpha)``."""
    alpha = alpha - nx * (bx + ax) / 2.0
    n1x = nx * (bx - ax)
    alpha = alpha - ny * (by + ay) / 2.0
    n1y = ny * (by - ay)
    return line_area(n1x, n1y, alpha)


def strip_moments(nx, ny, alpha, x0, x1):
    """Area and first moments (x, y) of the part of the strip ``[x0, x1] x [-1/2, 1/2]`` (cell coordinates) below the
    line ``nx x + ny y = alpha`` (the "inside" of :func:`line_area`). At each x the inside is an interval
    [y_lo(x), y_hi(x)], piecewise linear in x with kinks where the line meets y = -1/2 or 1/2 (a step where ny = 0):
    two-point Gauss on each of the three pieces integrates the area and both moments exactly."""
    xp = array_module(alpha)
    nx, ny, alpha = (xp.asarray(v) + xp.zeros_like(alpha) for v in (nx, ny, alpha))
    x0 = xp.zeros_like(alpha) + x0
    x1 = xp.zeros_like(alpha) + x1
    flat = xp.abs(nx) < 1e-14
    snx = xp.where(flat, 1.0, nx)
    k1 = xp.where(flat, x0, (alpha - 0.5 * ny) / snx)
    k2 = xp.where(flat, x0, (alpha + 0.5 * ny) / snx)
    b = xp.sort(xp.stack([x0, x1, _clamp(xp, k1, x0, x1), _clamp(xp, k2, x0, x1)]), axis=0)

    def interval(x):
        sny = xp.where(ny == 0.0, 1.0, ny)
        y = _clamp(xp, (alpha - nx * x) / sny, -0.5, 0.5)
        lo = xp.where(ny > 0.0, -0.5, y)
        hi = xp.where(ny > 0.0, y, 0.5)
        inside = nx * x <= alpha
        lo = xp.where(ny == 0.0, xp.where(inside, -0.5, 0.0), lo)
        hi = xp.where(ny == 0.0, xp.where(inside, 0.5, 0.0), hi)
        return lo, hi

    g = 0.5 / 3.0 ** 0.5
    area = xp.zeros_like(alpha)
    mx = xp.zeros_like(alpha)
    my = xp.zeros_like(alpha)
    for i in range(3):
        a, c = b[i], b[i + 1]
        w = 0.5 * (c - a)
        for t in (0.5 - g, 0.5 + g):
            x = a + t * (c - a)
            lo, hi = interval(x)
            area = area + (hi - lo) * w
            mx = mx + (hi - lo) * x * w
            my = my + 0.5 * (hi * hi - lo * lo) * w
    return area, mx, my

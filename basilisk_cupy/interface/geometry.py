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

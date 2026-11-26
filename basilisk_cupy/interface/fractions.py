"""Volume fractions, interface normals and reconstruction (Basilisk ``fractions.h``, ``myc2d.h``, 2-D).

Arrays are cell-centred with one ghost layer (``pad``: shape ``(nx + 2, ny + 2)``) where a
3x3 stencil is needed; vertex fields have shape ``(nx + 1, ny + 1)``.
"""

from __future__ import annotations

from ..backend import array_module
from .geometry import line_alpha, line_area

NOT_ZERO = 1e-30


def _sign(xp, v):
    """Basilisk ``sign()``: +1 for ``v > 0``, else -1."""
    return xp.where(v > 0.0, 1.0, -1.0)


def mycs(cp):
    """Mixed Youngs-Centred normal (Aulisa et al., Basilisk ``mycs()``) of every interior cell.

    ``cp``: volume fraction with one ghost layer. Returns ``(nx, ny)`` normalised so that
    ``|nx| + |ny| = 1`` (pointing out of the phase ``c = 1``).
    """
    xp = array_module(cp)
    c = lambda i, j: cp[1 + i: cp.shape[0] - 1 + i, 1 + j: cp.shape[1] - 1 + j]  # noqa: E731
    c_t = c(-1, 1) + c(0, 1) + c(1, 1)
    c_b = c(-1, -1) + c(0, -1) + c(1, -1)
    c_r = c(1, -1) + c(1, 0) + c(1, 1)
    c_l = c(-1, -1) + c(-1, 0) + c(-1, 1)
    mx0 = 0.5 * (c_l - c_r)
    my0 = 0.5 * (c_b - c_t)
    ix = xp.abs(mx0) <= xp.abs(my0)  # minimum coefficient between mx0 and my0 wins
    my0 = xp.where(ix, xp.where(my0 > 0.0, 1.0, -1.0), my0)
    mx0 = xp.where(ix, mx0, xp.where(mx0 > 0.0, 1.0, -1.0))
    mm1 = c(-1, -1) + 2.0 * c(-1, 0) + c(-1, 1)  # Youngs' normal
    mm2 = c(1, -1) + 2.0 * c(1, 0) + c(1, 1)
    mx1 = mm1 - mm2 + NOT_ZERO
    mm1 = c(-1, -1) + 2.0 * c(0, -1) + c(1, -1)
    mm2 = c(-1, 1) + 2.0 * c(0, 1) + c(1, 1)
    my1 = mm1 - mm2 + NOT_ZERO
    youngs = xp.where(ix, xp.abs(mx1) / xp.abs(my1) > xp.abs(mx0), xp.abs(my1) / xp.abs(mx1) > xp.abs(my0))
    mx0 = xp.where(youngs, mx1, mx0)
    my0 = xp.where(youngs, my1, my0)
    mm1 = xp.abs(mx0) + xp.abs(my0)
    return mx0 / mm1, my0 / mm1


def reconstruction(cp):
    """Interface normal ``(nx, ny)`` and intercept ``alpha`` of every interior cell (zero in full/empty cells)."""
    xp = array_module(cp)
    c = cp[1:-1, 1:-1]
    nx, ny = mycs(cp)
    interface = (c > 0.0) & (c < 1.0)
    alpha = xp.where(interface, line_alpha(c, nx, ny), 0.0)
    return xp.where(interface, nx, 0.0), xp.where(interface, ny, 0.0), alpha


def fractions(phi, val: float = 0.0):
    """Volume fractions from a level-set function sampled on the vertices (``phi > val`` is inside).

    ``phi``: shape ``(nx + 1, ny + 1)``. Basilisk ``fractions()`` in 2-D: line fractions on the
    cell edges by linear interpolation, the average normal from them, the intercept as the
    mean over the cut edges.
    """
    xp = array_module(phi)
    d = phi - val
    # line fractions of the edges along x (between vertices (i, j), (i + 1, j)): shape (nx, ny + 1)
    def edge(p0, p1, f0, f1):
        cut = p0 * p1 < 0.0
        frac = p0 / xp.where(cut, f0 - f1, 1.0)
        frac = xp.where(p0 < 0.0, 1.0 - frac, frac)
        return xp.where(cut, frac, ((p0 > 0.0) | (p1 > 0.0)).astype(phi.dtype))

    px = edge(d[:-1, :], d[1:, :], phi[:-1, :], phi[1:, :])   # along x, (nx, ny + 1)
    py = edge(d[:, :-1], d[:, 1:], phi[:, :-1], phi[:, 1:])   # along y, (nx + 1, ny)
    nx = py[:-1, :] - py[1:, :]
    ny = px[:, :-1] - px[:, 1:]
    nn = xp.abs(nx) + xp.abs(ny)
    full = nn == 0.0
    nns = xp.where(full, 1.0, nn)
    nx, ny = nx / nns, ny / nns
    alpha = xp.zeros_like(nx)
    ni = xp.zeros_like(nx)
    for i in (0, 1):
        # x: edge along x at row j + i (left vertex (i_cell, j + i))
        p = px[:, i: px.shape[1] - 1 + i]
        ph = d[:-1, i: d.shape[1] - 1 + i]
        cut = (p > 0.0) & (p < 1.0)
        alpha = xp.where(cut, alpha + (nx * (_sign(xp, ph) * (p - 0.5)) + ny * (i - 0.5)), alpha)
        ni = ni + cut
        # y: edge along y at column i_cell + i (bottom vertex (i_cell + i, j))
        p = py[i: py.shape[0] - 1 + i, :]
        ph = d[i: d.shape[0] - 1 + i, :-1]
        cut = (p > 0.0) & (p < 1.0)
        alpha = xp.where(cut, alpha + (ny * (_sign(xp, ph) * (p - 0.5)) + nx * (i - 0.5)), alpha)
        ni = ni + cut
    area = line_area(nx, ny, alpha / xp.where(ni == 0, 1.0, ni))
    area = xp.where(ni == 4, 0.0, area)
    area = xp.where(ni == 0, xp.maximum(px[:, :-1], py[:-1, :]), area)
    return xp.where(full, px[:, :-1], area)


def fraction(grid, func, val: float = 0.0, refine: int = 1):
    """Basilisk's ``fraction(f, func(x, y))``: ``func`` evaluated on the vertices of ``grid``.

    ``refine = k > 1``: the mean of the fractions of the k x k sub-cells of every cell (``func`` on the vertices
    of a grid k times finer). The edge interpolation of ``func`` misplaces the interface by O(h^2) differently in
    every cell, which the height functions turn into an O(1) curvature noise that does not converge (a circle:
    5 % at N = 32-256); the sub-cells divide that offset by k^2 (docs/notes/66)."""
    xp = grid.xp
    k = int(refine)
    if k <= 1:
        xv, yv = xp.meshgrid(grid.faces(0), grid.faces(1), indexing="ij")
        return fractions(func(xv, yv), val)
    fx, fy = (xp.asarray(grid.faces(ax)) for ax in range(2))
    t = xp.arange(k + 1) / k
    sub = [(f[:-1, None] + t[None, :] * (f[1:] - f[:-1])[:, None]) for f in (fx, fy)]
    sx = xp.concatenate([sub[0][:, :-1].reshape(-1), fx[-1:]])
    sy = xp.concatenate([sub[1][:, :-1].reshape(-1), fy[-1:]])
    xv, yv = xp.meshgrid(sx, sy, indexing="ij")
    c = fractions(func(xv, yv), val)
    n = (fx.size - 1, fy.size - 1)
    return c.reshape(n[0], k, n[1], k).mean(axis=(1, 3))

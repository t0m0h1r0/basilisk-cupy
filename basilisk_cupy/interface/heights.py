"""Height functions and interface position (Basilisk ``heights.h`` multigrid version, ``curvature.h``
``position()``), 2-D.

``h[d]`` is the distance, in cells and along axis ``d``, from the cell centre to the interface,
from the column integral of the volume fraction (up to 5.5 cells away), or ``NODATA``. The
orientation is encoded by adding ``HSHIFT`` when the volume fraction is one at the top end.
"""

from __future__ import annotations

import numpy as np

from ..backend import array_module
from ..boundary import LOWER, UPPER, Dirichlet, Neumann
from .fractions import mycs
from .geometry import line_alpha

HSHIFT = 20.0
NODATA = float(np.float32(1e30))  # Basilisk: #define nodata HUGE, HUGE 1e30f
COMPLETE = -1.0


def height(H, xp):
    return xp.where(H > HSHIFT / 2.0, H - HSHIFT, xp.where(H < -HSHIFT / 2.0, H + HSHIFT, H))


def pad_bc(arr, bc, width: int):
    """``arr`` (interior) with ``width`` ghost layers from the boundary conditions ``bc``
    (periodic, symmetric Neumann(0), constant Dirichlet), axis by axis (corners included)."""
    xp = array_module(arr)
    out = arr
    for ax in range(arr.ndim):
        parts = []
        for side in (LOWER, UPPER):
            cond = bc.get(ax, side)
            idx = [slice(None)] * arr.ndim
            if cond.periodic:
                idx[ax] = slice(-width, None) if side == LOWER else slice(0, width)
                parts.append(out[tuple(idx)])
                continue
            idx[ax] = slice(width - 1, None, -1) if side == LOWER else slice(-1, -width - 1, -1)
            mirror = out[tuple(idx)]
            if isinstance(cond, Neumann):
                if not (isinstance(cond.gradient, (int, float)) and cond.gradient == 0):
                    raise NotImplementedError("height functions: only Neumann(0) walls")
                parts.append(mirror)
            elif isinstance(cond, Dirichlet) and isinstance(cond.value, (int, float)):
                parts.append(2.0 * cond.value - mirror)
            else:
                raise NotImplementedError(f"height functions: boundary condition {cond!r}")
        out = xp.concatenate([parts[0], out, parts[1]], axis=ax)
    return out


def _sl(a, axis, k, g, n):
    """Interior rows shifted by ``k`` along ``axis`` of an array padded with ``g`` layers."""
    idx = [slice(g, g + m) for m in n]
    idx[axis] = slice(g + k, g + k + n[axis])
    return a[tuple(idx)]


def _half_column(xp, cpad, spad, h, axis, j, n):
    G = 2
    c0 = _sl(cpad, axis, 0, G, n)
    S = c0.copy()
    H = c0.copy()
    if j == 1:
        inconsistent = h == 300.0
        s = xp.trunc((h + HSHIFT / 2.0) / 100.0)
        st_h = xp.where(inconsistent, NODATA, h - 100.0 * s)
        st_s = xp.where(inconsistent, COMPLETE, s - 1.0)
        fresh = st_s == COMPLETE
        S = xp.where(fresh, S, st_s)
        H = xp.where(fresh, H, st_h)
    active = xp.ones(S.shape, dtype=bool)
    for i in range(1, 5):
        ci = _sl(cpad, axis, i * j, G, n) if i <= 2 else _sl(spad, axis, (i - 2) * j, G, n)
        Hn = H + ci
        inter = (S > 0.0) & (S < 1.0)
        leave = inter & ((ci <= 0.0) | (ci >= 1.0))
        down = ~inter & (S >= 1.0) & (ci <= 0.0)
        up = ~inter & ~down & (S <= 0.0) & (ci >= 1.0)
        incons = ~inter & ~down & ~up & (S == ci) & (xp.trunc(Hn) != Hn)
        Hnew = xp.where(leave, Hn - i * ci, Hn)
        Hnew = xp.where(down, (Hn - 0.5) * j + (HSHIFT if j == -1 else 0.0), Hnew)
        Hnew = xp.where(up, (i + 0.5 - Hn) * j + (HSHIFT if j == 1 else 0.0), Hnew)
        Snew = xp.where(inter, ci, S)
        Snew = xp.where(down | up, COMPLETE, Snew)
        H = xp.where(active, Hnew, H)
        S = xp.where(active, Snew, S)
        active = active & ~(leave | down | up | incons)
    if j == -1:
        partial = S != COMPLETE
        bad = partial & (((c0 <= 0.0) | (c0 >= 1.0)) | ((S > 0.0) & (S < 1.0)))
        out = xp.where(S == COMPLETE, H, H + 100.0 * (1.0 + (S >= 1.0)))
        return xp.where(bad, 300.0, out)
    take = (st_s != COMPLETE) | ((S == COMPLETE) & (xp.abs(height(H, xp)) < xp.abs(height(st_h, xp))))
    st_s = xp.where(take, S, st_s)
    st_h = xp.where(take, H, st_h)
    return xp.where(st_s != COMPLETE, NODATA, xp.where(st_h > 1e10, NODATA, st_h))


def heights(c, bc):
    """Height functions ``[h_x, h_y]`` of the volume fraction ``c`` (interior array) with conditions ``bc``."""
    xp = array_module(c)
    n = c.shape
    G = 2
    cpad = pad_bc(c, bc, G)
    hs = []
    for axis in range(2):
        h = None
        for j in (-1, 1):
            shifted = _sl(cpad, axis, 2 * j, G, n)       # s.x[] = c[2 j] on the interior ...
            spad = pad_bc(shifted, bc, G)                 # ... with the boundary conditions of c
            h = _half_column(xp, cpad, spad, h, axis, j, n)
        hs.append(h)
    return _column_propagation(xp, hs)


def _column_propagation(xp, hs):
    """Smallest height within +-2 cells along each column (Basilisk ``column_propagation``)."""
    out = []
    if xp.__name__ == "cupy":
        from ..kernels.heights_gpu import column_propagation  # noqa: PLC0415

        for axis, h in enumerate(hs):
            h = xp.ascontiguousarray(h, dtype=xp.float64).copy()
            column_propagation(h, axis)
            out.append(h)
        return out
    for axis, h in enumerate(hs):
        # Basilisk updates in place, cell by cell in memory order; heights from farther cells are only
        # taken when |height| <= 3.5, so the order matters only through already-updated neighbours.
        h = h.copy()
        n = h.shape[axis]
        hn = xp.moveaxis(h, axis, 0)
        for idx in range(n):
            cur = hn[idx]
            for i in range(-2, 3):
                k = idx + i
                if k < 0 or k >= n:
                    continue
                nb = hn[k]
                cond = (xp.abs(height(nb, xp)) <= 3.5) & (xp.abs(height(nb, xp) + i) < xp.abs(height(cur, xp)))
                cur = xp.where(cond, nb + i, cur)
            hn[idx] = cur
        out.append(xp.moveaxis(hn, 0, axis))
    return out


def interfacial(cpad):
    """Basilisk ``interfacial()``: interfacial cells, or full/empty cells next to the opposite (1 ghost layer)."""
    xp = array_module(cpad)
    c = cpad[1:-1, 1:-1]
    nbrs = [cpad[2:, 1:-1], cpad[:-2, 1:-1], cpad[1:-1, 2:], cpad[1:-1, :-2]]
    near_empty = nbrs[0] <= 0.0
    near_full = nbrs[0] >= 1.0
    for v in nbrs[1:]:
        near_empty = near_empty | (v <= 0.0)
        near_full = near_full | (v >= 1.0)
    return xp.where(c >= 1.0, near_empty, xp.where(c <= 0.0, near_full, True))


def line_length_center(mx, my, alpha):
    """Centre ``(px, py)`` (cell coordinates) of the interface segment (Basilisk ``line_length_center``)."""
    xp = array_module(alpha)
    alpha = alpha + (mx + my) / 2.0
    nx, ny = mx, my
    alpha = xp.where(nx < 0.0, alpha - nx, alpha)
    nx = xp.abs(nx)
    alpha = xp.where(ny < 0.0, alpha - ny, alpha)
    ny = xp.abs(ny)
    one = xp.ones_like(alpha)
    px = xp.where(alpha >= nx, 1.0, alpha / xp.where(nx == 0, one, nx))
    py = xp.where(alpha >= nx, (alpha - nx) / xp.where(ny == 0, one, ny), 0.0)
    above = alpha >= ny
    py = xp.where(above, py + 1.0, py + alpha / xp.where(ny == 0, one, ny))
    px = xp.where(above, px + (alpha - ny) / xp.where(nx == 0, one, nx), px)
    px = xp.minimum(xp.maximum(px / 2.0, 0.0), 1.0)
    px = xp.where(mx < 0.0, 1.0 - px, px) - 0.5
    py = xp.minimum(xp.maximum(py / 2.0, 0.0), 1.0)
    py = xp.where(my < 0.0, 1.0 - py, py) - 0.5
    # degenerate (axis-aligned) lines
    vx = (xp.where(my < 0.0, 1.0 - alpha, alpha) - 0.5)
    vy = (xp.where(mx < 0.0, 1.0 - alpha, alpha) - 0.5)
    px, py = xp.where(nx < 1e-4, 0.0, px), xp.where(nx < 1e-4, vx, py)
    px, py = xp.where((nx >= 1e-4) & (ny < 1e-4), vy, px), xp.where((nx >= 1e-4) & (ny < 1e-4), 0.0, py)
    outside = (alpha <= 0.0) | (alpha >= nx + ny)
    return xp.where(outside, 0.0, px), xp.where(outside, 0.0, py)


def position(c, bc, grid, G=(0.0, 0.0), Z=(0.0, 0.0), hs=None):
    """``G . (x - Z)`` at the interface, in interfacial cells (``NODATA`` elsewhere); Basilisk ``position()``."""
    xp = array_module(c)
    hs = heights(c, bc) if hs is None else hs
    cp1 = pad_bc(c, bc, 1)
    x, y = grid.meshgrid()
    delta = float(grid.h[0])
    # HF position along each axis
    pos = []
    for axis in range(2):
        hh = height(hs[axis], xp)
        o = [x, y]
        o[axis] = o[axis] + hh * delta
        p = (o[0] - Z[0]) * G[0] + (o[1] - Z[1]) * G[1]
        pos.append(xp.where(xp.abs(hh) > 1.0, NODATA, p))
    nx_ = cp1[2:, 1:-1] - cp1[:-2, 1:-1]
    ny_ = cp1[1:-1, 2:] - cp1[1:-1, :-2]
    first_x = ~(xp.abs(nx_) < xp.abs(ny_))  # sorted by decreasing |n|; ties keep x first
    p1 = xp.where(first_x, pos[0], pos[1])
    p2 = xp.where(first_x, pos[1], pos[0])
    hp = xp.where(p1 == NODATA, p2, p1)
    # fall back to the centroid of the PLIC segment
    mx, my = mycs(cp1)
    alpha = line_alpha(c, mx, my)
    cx, cy = line_length_center(mx, my, alpha)
    cent = (x + delta * cx - Z[0]) * G[0] + (y + delta * cy - Z[1]) * G[1]
    hp = xp.where(hp == NODATA, cent, hp)
    return xp.where(interfacial(cp1), hp, NODATA)

"""Interface curvature from volume fractions (Basilisk ``curvature.h`` ``curvature()``, ``parabola.h``), 2-D.

In interfacial cells, in this order (Popinet 2009):

1. height-function curvature ``h_xx / (1 + h_x^2)^{3/2}`` along the direction where ``|grad c|``
   is largest (then the other), limited to ``1/Delta``;
2. a parabola fitted to the "mixed" heights of both directions (if at least 3 independent
   points) and the interface centroid (weight 0.1);
3. the average of the curvatures computed in the 3x3 neighbourhood;
4. a parabola fitted to the interface centroids of the 3x3 neighbourhood.

Steps 1 and the selection are array operations; 2-4 concern few cells and run per cell on the
host, in Basilisk's operation order.
"""

from __future__ import annotations

import math

import numpy as np

from ..backend import array_module, to_numpy
from .fractions import mycs
from .geometry import line_alpha
from .heights import HSHIFT, NODATA, heights, interfacial, pad_bc

PARABOLA_FIT_CENTER_WEIGHT = 0.1


def _orientation(H, xp):
    return xp.abs(H) > HSHIFT / 2.0


def _pad_heights(h, bc, axis, xp):
    """One ghost layer for the height component ``axis``: antisymmetric across the walls normal to
    ``axis``, symmetric across the others, periodic wrap (Basilisk's vector default)."""
    out = h
    for ax in range(2):
        parts = []
        for side in (0, 1):
            cond = bc.get(ax, side)
            idx = [slice(None)] * 2
            if cond.periodic:
                idx[ax] = slice(-1, None) if side == 0 else slice(0, 1)
                parts.append(out[tuple(idx)])
            else:
                idx[ax] = slice(0, 1) if side == 0 else slice(-1, None)
                v = out[tuple(idx)]
                parts.append(-v if ax == axis else v)
        out = xp.concatenate([parts[0], out, parts[1]], axis=ax)
    return out


def _kappa_dir(hp, axis, delta, xp):
    """Basilisk ``kappa_x`` (``axis = 0``: heights h.x along y) or ``kappa_y`` on the padded heights."""
    def nb(k):
        if axis == 0:
            return hp[1:-1, 1 + k: hp.shape[1] - 1 + k]
        return hp[1 + k: hp.shape[0] - 1 + k, 1:-1]
    hm, h0, hpl = nb(-1), nb(0), nb(1)
    ori = _orientation(h0, xp)
    ok = (h0 != NODATA) & (hm != NODATA) & (hpl != NODATA)
    ok &= (_orientation(hm, xp) == ori) & (_orientation(hpl, xp) == ori)
    one = xp.ones_like(h0)
    hm_, h0_, hp_ = xp.where(ok, hm, one), xp.where(ok, h0, one), xp.where(ok, hpl, one)
    hx = (hp_ - hm_) / 2.0
    hxx = (hp_ + hm_ - 2.0 * h0_) / delta
    k = hxx / (1.0 + hx * hx) ** 1.5
    return xp.where(ok, k, NODATA)


def _height_curvature(cp, hs, bc, delta, xp):
    kx = _kappa_dir(_pad_heights(hs[0], bc, 0, xp), 0, delta, xp)
    ky = _kappa_dir(_pad_heights(hs[1], bc, 1, xp), 1, delta, xp)
    nx = cp[2:, 1:-1] - cp[:-2, 1:-1]
    ny = cp[1:-1, 2:] - cp[1:-1, :-2]
    swap = xp.abs(nx) < xp.abs(ny)
    k1, n1 = xp.where(swap, ky, kx), xp.where(swap, ny, nx)
    k2, n2 = xp.where(swap, kx, ky), xp.where(swap, nx, ny)
    k = xp.where(k1 != NODATA, xp.where(n1 < 0.0, -k1, k1), xp.where(k2 != NODATA, xp.where(n2 < 0.0, -k2, k2), NODATA))
    lim = xp.abs(k) > 1.0 / delta
    sign = xp.where(k > 0.0, 1.0, -1.0)
    return xp.where((k != NODATA) & lim, sign / delta, k)


# -- scalar helpers (host, Basilisk's order) ------------------------------------------------------
def _line_alpha1(c, nx, ny):
    return float(line_alpha(np.asarray(c), np.asarray(nx), np.asarray(ny)))


def _line_length_center(mx, my, alpha):
    """Basilisk ``line_length_center``: ``(length, (px, py))``."""
    alpha += (mx + my) / 2.0
    nx, ny = mx, my
    if nx < 0.0:
        alpha -= nx
        nx = -nx
    if ny < 0.0:
        alpha -= ny
        ny = -ny
    if alpha <= 0.0 or alpha >= nx + ny:
        return 0.0, (0.0, 0.0)
    if nx < 1e-4:
        return 1.0, (0.0, (1.0 - alpha if my < 0.0 else alpha) - 0.5)
    if ny < 1e-4:
        return 1.0, ((1.0 - alpha if mx < 0.0 else alpha) - 0.5, 0.0)
    px = py = 0.0
    if alpha >= nx:
        px += 1.0
        py += (alpha - nx) / ny
    else:
        px += alpha / nx
    ax, ay = px, py
    if alpha >= ny:
        py += 1.0
        ay -= 1.0
        px += (alpha - ny) / nx
        ax -= (alpha - ny) / nx
    else:
        py += alpha / ny
        ay -= alpha / ny
    out = []
    for p, m in ((px, mx), (py, my)):
        p /= 2.0
        p = min(max(p, 0.0), 1.0)
        if m < 0.0:
            p = 1.0 - p
        out.append(p - 0.5)
    return math.sqrt(ax * ax + ay * ay), (out[0], out[1])


def _smatrix_inverse(m, pivmin):
    """Basilisk ``smatrix_inverse`` (Gauss--Jordan with full pivoting), in place; returns the minimal pivot or 0."""
    n = len(m)
    indxc, indxr, ipiv = [0] * n, [0] * n, [-1] * n
    icol = irow = 0
    minpiv = float(np.float32(1e30))
    for i in range(n):
        big = 0.0
        for j in range(n):
            if ipiv[j] != 0:
                for k in range(n):
                    if ipiv[k] == -1 and abs(m[j][k]) >= big:
                        big = abs(m[j][k])
                        irow, icol = j, k
        ipiv[icol] += 1
        if irow != icol:
            m[irow], m[icol] = m[icol], m[irow]
        indxr[i], indxc[i] = irow, icol
        if abs(m[icol][icol]) < minpiv:
            minpiv = abs(m[icol][icol])
        if minpiv < pivmin:
            break
        pivinv = 1.0 / m[icol][icol]
        m[icol][icol] = 1.0
        for l in range(n):
            m[icol][l] *= pivinv
        for ll in range(n):
            if ll != icol:
                dum = m[ll][icol]
                m[ll][icol] = 0.0
                for l in range(n):
                    m[ll][l] -= m[icol][l] * dum
    if minpiv >= pivmin:
        for l in range(n - 1, -1, -1):
            if indxr[l] != indxc[l]:
                for k in range(n):
                    m[k][indxr[l]], m[k][indxc[l]] = m[k][indxc[l]], m[k][indxr[l]]
    return 0.0 if minpiv < pivmin else minpiv


class _ParabolaFit:
    """Basilisk ``parabola.h`` in 2-D: ``y = a0 x^2 + a1 x + a2`` in the frame of the normal."""

    def __init__(self, o, m):
        self.o = o
        norm = math.sqrt(m[0] * m[0] + m[1] * m[1])
        self.m = (m[0] / norm, m[1] / norm)
        self.M = [[0.0] * 3 for _ in range(3)]
        self.rhs = [0.0] * 3
        self.a = [0.0] * 3

    def add(self, p, w):
        x1, y1 = p[0] - self.o[0], p[1] - self.o[1]
        x = self.m[1] * x1 - self.m[0] * y1
        y = self.m[0] * x1 + self.m[1] * y1
        x2 = w * x * x
        x3 = x2 * x
        x4 = x3 * x
        M = self.M
        M[0][0] += x4
        M[1][0] += x3
        M[1][1] += x2
        M[2][1] += w * x
        M[2][2] += w
        self.rhs[0] += x2 * y
        self.rhs[1] += w * x * y
        self.rhs[2] += w * y

    def solve(self):
        M = self.M
        M[0][1] = M[1][0]
        M[0][2] = M[2][0] = M[1][1]
        M[1][2] = M[2][1]
        piv = _smatrix_inverse(M, 1e-10)
        if piv:
            self.a[0] = M[0][0] * self.rhs[0] + M[0][1] * self.rhs[1] + M[0][2] * self.rhs[2]
            self.a[1] = M[1][0] * self.rhs[0] + M[1][1] * self.rhs[1] + M[1][2] * self.rhs[2]
        else:
            self.a[0] = self.a[1] = 0.0
        return piv

    def curvature(self, kappamax):
        dnm = 1.0 + self.a[1] * self.a[1]
        kappa = -2.0 * self.a[0] / dnm ** 1.5
        if abs(kappa) > kappamax:
            return kappamax if kappa > 0.0 else -kappamax
        return kappa


def _independents(pts):
    if len(pts) < 2:
        return len(pts)
    ni = 1
    for j in range(1, len(pts)):
        depends = False
        for i in range(j):
            d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d2 < 0.25:
                depends = True
                break
        ni += not depends
    return ni


def _height(H):
    return H - HSHIFT if H > HSHIFT / 2.0 else (H + HSHIFT if H < -HSHIFT / 2.0 else H)


def _ori(H):
    return abs(H) > HSHIFT / 2.0


def _fit(cp, hp, mxy, i, j, delta):
    """``height_curvature_fit`` at interior cell ``(i, j)`` (padded arrays shifted by one)."""
    pts = []
    for axis in (0, 1):  # foreach_dimension: x (h.y along x) then y (h.x along y), Basilisk's rotation
        hh = hp[1]  if axis == 0 else hp[0]
        def H(k):
            return hh[i + 1 + k, j + 1] if axis == 0 else hh[i + 1, j + 1 + k]
        n1 = n2 = 0
        for k in (-1, 0, 1):
            if H(k) != NODATA:
                if _ori(H(k)):
                    n1 += 1
                else:
                    n2 += 1
        ori = n1 > n2
        for k in (-1, 0, 1):
            if H(k) != NODATA and _ori(H(k)) == ori:
                pts.append((float(k), _height(H(k))) if axis == 0 else (_height(H(k)), float(k)))
    if _independents(pts) < 3:
        return NODATA
    c = cp[i + 1, j + 1]
    m = (mxy[0][i, j], mxy[1][i, j])
    alpha = _line_alpha1(c, m[0], m[1])
    _, fc = _line_length_center(m[0], m[1], alpha)
    fit = _ParabolaFit(fc, m)
    fit.add(fc, PARABOLA_FIT_CENTER_WEIGHT)
    for p in pts:
        fit.add(p, 1.0)
    fit.solve()
    return fit.curvature(2.0) / delta


def _centroids_fit(cp, mxp, i, j, delta):
    """``centroids_curvature_fit`` at interior cell ``(i, j)``; ``mxp`` normals on the padded cells."""
    c = cp[i + 1, j + 1]
    m = (mxp[0][i + 1, j + 1], mxp[1][i + 1, j + 1])
    alpha = _line_alpha1(c, m[0], m[1])
    _, fc = _line_length_center(m[0], m[1], alpha)
    fit = _ParabolaFit(fc, m)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cn = cp[i + 1 + di, j + 1 + dj]
            if 0.0 < cn < 1.0:
                mn = (mxp[0][i + 1 + di, j + 1 + dj], mxp[1][i + 1 + di, j + 1 + dj])
                an = _line_alpha1(cn, mn[0], mn[1])
                area, fcn = _line_length_center(mn[0], mn[1], an)
                fit.add((fcn[0] + di, fcn[1] + dj), area)
    fit.solve()
    return fit.curvature(2.0) / delta


def curvature(c, bc, grid, sigma: float = 1.0, hs=None, stats: dict | None = None):
    """``sigma * kappa`` in interfacial cells, ``NODATA`` elsewhere (Basilisk ``curvature()``)."""
    xp = array_module(c)
    delta = float(grid.h[0])
    hs = heights(c, bc) if hs is None else hs
    cp = pad_bc(c, bc, 1)
    inter = interfacial(cp)
    k = xp.where(inter, _height_curvature(cp, hs, bc, delta, xp), NODATA)
    n_h = int(((k != NODATA) & inter).sum())
    # cells left for the fits / averages: few, on the host
    cph = to_numpy(cp)
    kh = to_numpy(k).copy()
    interh = to_numpy(inter)
    todo = np.argwhere(interh & (kh == NODATA))
    n_f = n_a = n_c = 0
    if todo.size:
        hp = [to_numpy(_pad_heights(hs[a], bc, a, xp)) for a in range(2)]
        mx, my = (to_numpy(v) for v in mycs(cp))
        for i, j in todo:
            v = _fit(cph, hp, (mx, my), i, j, delta)
            kh[i, j] = v
            n_f += int(v != NODATA)
    out = kh.copy()
    rest = np.argwhere(interh & (kh == NODATA))
    if rest.size:
        kp = to_numpy(pad_bc(xp.asarray(kh), bc, 1))
        cpp = pad_bc(cp, bc, 1)  # two layers for the centroids of the neighbours' normals
        mxp = [to_numpy(v) for v in mycs(cpp)]
        for i, j in rest:
            sk = a = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    v = kp[i + 1 + di, j + 1 + dj]
                    if v < NODATA:
                        sk += v
                        a += 1
            if a > 0.0:
                out[i, j] = sk / a
                n_a += 1
            else:
                out[i, j] = _centroids_fit(cph, mxp, i, j, delta)
                n_c += 1
    if stats is not None:
        stats.update(h=n_h, f=n_f, a=n_a, c=n_c)
    res = xp.asarray(out)
    return xp.where(res == NODATA, NODATA, sigma * res)

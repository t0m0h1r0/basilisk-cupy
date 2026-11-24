"""Height-function curvature of the CLS field ``psi``, with CCD or 2nd-order derivatives along the interface.

For a cell whose ``psi`` gradient is mostly along axis ``a`` (``t`` the other axis), the heights of
the columns of its row are the positions, along ``a``, of the one ``psi = 1/2`` crossing in the
window of ``2m + 1`` cells centred on the row (``heights="crossing"``, the default): the crossing
between two cells is the root of the cubic Lagrange interpolant of ``phi_map = eps ln(psi/(1 - psi))``
through four cells. ``heights="sum"`` takes the column sum ``sum psi Delta`` instead, exact for a
planar symmetric profile but shifted by ``O(eps^2 kappa)`` where the interface is curved and
oblique (1-3 % in the curvature, N = 128; docs/notes/25). On each run of consecutive columns with
a height, ``eta'`` and ``eta''`` come from one CCD solve along ``t`` (``method="ccd"``, compact
closures at the run ends, whose ``trim`` cells on each side are not used: with them the capillary
wave diverged at the walls) or from central differences (``method="fd2"``, Basilisk's choice), and

    kappa = s eta'' / (1 + eta'^2)^(3/2),   s = +1 if psi = 1 lies towards +a,

positive for a drop with ``psi = 1`` inside. Cells of the band ``psi_min < psi < 1 - psi_min`` that
no run covers take the curvature of ``psi`` itself (:func:`.cls_curvature.psi_curvature`).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..backend import array_module, to_numpy
from ..discretization.ccd import ccd_matrices
from ..fields import CellField
from .cls_curvature import PSI_MIN, psi_curvature


@lru_cache(maxsize=None)
def _ccd(n: int, h: float, periodic: bool):
    kind = "periodic" if periodic else "extrapolate"
    d1, d2 = ccd_matrices(n, h, kind, kind)[:2]
    return d1, d2


def _runs(mask):
    """``(start, stop)`` of the runs of True in a 1-D boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _crossing_positions(Q, eps, h):
    """Position along the column (from the centre of cell 0, length units) of the ``psi = 1/2``
    crossing between cells ``k`` and ``k + 1``, ``nan`` where there is none: ``phi_map =
    eps ln(psi/(1 - psi))`` of cells ``k - 1 .. k + 2``, cubic Lagrange, Newton from the linear root."""
    qc = np.clip(Q, 1e-12, 1.0 - 1e-12)
    f = eps * np.log(qc / (1.0 - qc))
    nt, na = Q.shape
    out = np.full((nt, na), np.nan)
    fp = np.pad(f, ((0, 0), (1, 2)), mode="edge")
    for k in range(na - 1):
        f0, f1 = f[:, k], f[:, k + 1]
        cut = (f0 * f1 <= 0) & (f0 != f1)
        if not cut.any():
            continue
        fm, f2 = fp[:, k], fp[:, k + 3]
        s = np.where(cut, f0 / np.where(cut, f0 - f1, 1.0), 0.0)
        for _ in range(4):
            L = (-s * (s - 1) * (s - 2) / 6 * fm + (s + 1) * (s - 1) * (s - 2) / 2 * f0
                 - (s + 1) * s * (s - 2) / 2 * f1 + (s + 1) * s * (s - 1) / 6 * f2)
            dL = (-(3 * s * s - 6 * s + 2) / 6 * fm + (3 * s * s - 4 * s - 1) / 2 * f0
                  - (3 * s * s - 2 * s - 2) / 2 * f1 + (3 * s * s - 1) / 6 * f2)
            s = np.where(cut, np.clip(s - L / np.where(np.abs(dL) > 1e-300, dL, 1.0), 0.0, 1.0), s)
        out[:, k] = np.where(cut, (k + s) * h, np.nan)
    return out


def height_curvature(psi: CellField, *, m: int = 4, delta: float = 0.5, method: str = "ccd",
                     psi_min: float = PSI_MIN, heights: str = "crossing", eps: float | None = None, trim: int = 2):
    grid = psi.grid
    if grid.ndim != 2 or not grid.is_uniform:
        raise NotImplementedError("2-D uniform grids")
    xp = grid.xp
    q = to_numpy(psi.interior)
    n = q.shape
    h = [float(v) for v in grid.h]
    periodic = [psi.bc.is_periodic(ax) for ax in range(2)]
    gx = np.gradient(q, axis=0)
    gy = np.gradient(q, axis=1)
    band = (q > psi_min) & (q < 1.0 - psi_min)
    kappa = np.full(n, np.nan)
    for a in range(2):
        t = 1 - a
        Q = np.moveaxis(q, a, 1)  # [t index, a index]
        nt, na = Q.shape
        mode = "wrap" if periodic[a] else "edge"
        P = np.pad(Q, ((0, 0), (m, m)), mode=mode)
        c = np.concatenate([np.zeros((nt, 1)), np.cumsum(P, axis=1)], axis=1)
        S = (c[:, 2 * m + 1:] - c[:, :na]) * h[a]  # window sums, [t, j]
        lo, hi = P[:, :na], P[:, 2 * m:]
        s = np.sign(hi - lo)
        valid = (np.minimum(lo, hi) < delta) & (np.maximum(lo, hi) > 1.0 - delta)
        eta = -s * S  # height up to a constant per row j
        if heights == "crossing":  # the one crossing within the window, by phi_map interpolation
            cp = _crossing_positions(Q, eps, h[a])
            cnt = np.pad(~np.isnan(cp), ((0, 0), (m, m)))
            val = np.pad(np.nan_to_num(cp), ((0, 0), (m, m)))
            cc = np.concatenate([np.zeros((nt, 1)), np.cumsum(cnt, axis=1)], axis=1)
            cv = np.concatenate([np.zeros((nt, 1)), np.cumsum(val, axis=1)], axis=1)
            count = cc[:, 2 * m:2 * m + na] - cc[:, :na]  # crossings k in [j - m, j + m - 1]
            eta = cv[:, 2 * m:2 * m + na] - cv[:, :na]
            valid = valid & (count == 1)
        dom = np.moveaxis(np.abs(gx if a == 0 else gy) >= np.abs(gy if a == 0 else gx), a, 1)
        want = np.moveaxis(band, a, 1) & dom
        K = np.full((nt, na), np.nan)
        for j in np.nonzero(want.any(axis=0))[0]:
            for sign in (1.0, -1.0):
                ok = valid[:, j] & (s[:, j] == sign)
                if not ok.any():
                    continue
                if periodic[t] and ok.all():
                    runs, per = [(0, nt)], True
                else:
                    runs, per = _runs(ok), False
                for i0, i1 in runs:
                    L = i1 - i0
                    e = eta[i0:i1, j]
                    if method == "ccd" and (L >= 6 + 2 * trim or (per and L >= 3)):
                        d1, d2 = _ccd(L, h[t], per)
                        e1, e2 = d1 @ e, d2 @ e
                        sl = slice(0, L) if per else slice(trim, L - trim)
                        e1, e2 = e1[sl], e2[sl]
                    elif L >= 3:
                        if per:
                            e1 = (np.roll(e, -1) - np.roll(e, 1)) / (2 * h[t])
                            e2 = (np.roll(e, -1) - 2 * e + np.roll(e, 1)) / h[t] ** 2
                            sl = slice(0, L)
                        else:
                            e1 = (e[2:] - e[:-2]) / (2 * h[t])
                            e2 = (e[2:] - 2 * e[1:-1] + e[:-2]) / h[t] ** 2
                            sl = slice(1, L - 1)
                    else:
                        continue
                    k = sign * e2 / (1.0 + e1 * e1) ** 1.5
                    rows = np.arange(i0, i1)[sl]
                    take = want[rows, j]
                    K[rows[take], j] = k[take]
        Km = np.moveaxis(K, 1, a)
        kappa = np.where(np.isnan(kappa) & ~np.isnan(Km), Km, kappa)
    fallback = to_numpy(psi_curvature(psi, psi_min=psi_min))
    kappa = np.where(np.isnan(kappa), fallback, kappa)
    kappa = np.where(band, kappa, 0.0)
    return xp.asarray(kappa)

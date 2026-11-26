"""Curvature of a VOF fraction from Basilisk's heights differentiated by CCD along the interface (docs/notes/66).

tension.h differentiates the heights with the three-point differences of ``kappa_x`` / ``kappa_y``. Here, for
each axis ``a`` and orientation, every column (the line along ``a`` at a fixed index ``t`` of the other axis)
gets the absolute position ``P[t] = i_a + h_a`` (cells) of its interface from the cell of the column closest to
it (Basilisk's heights, ``heights.h``). Runs of consecutive columns whose positions differ by at most ``jump``
cells are one smooth interface; on each run of at least ``min_run`` columns (or a whole periodic line),

* the positions are column averages of the interface over the cell width (column sums of f): they are turned
  into point values ``P - P''/24`` (cells) with ``P''`` from the same CCD solve (``averages=True``);
* ``P'`` and ``P''`` come from one CCD solve along the run (compact closures at the ends, ``trim`` end columns
  not used), ``kappa = P''/Delta / (1 + P'^2)^(3/2)``, with Basilisk's sign (flipped when the fraction decreases
  along ``a``).

An interfacial cell takes the value of its column on the axis of its larger normal component (``mycs``-free:
the centred differences of ``c``, as tension.h); the other interfacial cells keep tension.h's value.
"""

from __future__ import annotations

import numpy as np

from ..backend import to_numpy
from .cls_heights import _ccd
from .curvature import curvature as basilisk_curvature
from .heights import HSHIFT, NODATA, heights, pad_bc


def _runs(ok, P, jump):
    runs, i0 = [], None
    for i, v in enumerate(ok):
        if v and i0 is not None and abs(P[i] - P[i - 1]) > jump:
            runs.append((i0, i))
            i0 = i
        elif v and i0 is None:
            i0 = i
        elif not v and i0 is not None:
            runs.append((i0, i))
            i0 = None
    if i0 is not None:
        runs.append((i0, len(ok)))
    return runs


def ccd_curvature(f, sigma: float = 1.0, *, min_run: int = 8, trim: int = 1, jump: float = 1.5,
                  averages: bool = True, choose: str = "normal", max_slope: float = np.inf,
                  stats: dict | None = None):
    """``sigma kappa`` in the interfacial cells of the VOF field ``f`` (``CellField``), ``NODATA`` elsewhere.
    ``choose``: which axis's run gives a cell its value: ``"normal"`` (the axis of the larger normal component) or
    ``"interior"`` (the run in which the cell's column lies farther from the run's ends). ``max_slope``: the CCD
    value is used only where ``|P'|`` is at most this (the column parametrisation is singular where the interface
    turns vertical, R (1 - 1/sqrt 2) away from 45 degrees on a circle); tension.h's value elsewhere."""
    grid = f.grid
    if grid.ndim != 2 or not grid.is_uniform:
        raise NotImplementedError("2-D uniform grids")
    xp = grid.xp
    c = f.interior
    hs_dev = heights(c, f.bc)
    base = to_numpy(basilisk_curvature(c, f.bc, grid, 1.0, hs=hs_dev))
    inter = base != NODATA
    hs = [to_numpy(v) for v in hs_dev]
    q = to_numpy(c)
    n = q.shape
    delta = float(grid.h[0])
    cp = to_numpy(pad_bc(c, f.bc, 1))
    gn = [cp[2:, 1:-1] - cp[:-2, 1:-1], cp[1:-1, 2:] - cp[1:-1, :-2]]
    kappa = np.full(n, np.nan)
    best = np.full(n, -1.0)  # distance (columns) of the value's column from its run's ends
    for a in range(2):
        b = 1 - a
        H = np.moveaxis(hs[a], a, 1)  # [t, i_a]
        G = np.moveaxis(gn[a], a, 1)
        mine = np.moveaxis(inter & ((np.abs(gn[a]) >= np.abs(gn[b])) | (choose == "interior")), a, 1)
        nt, na = H.shape
        periodic = f.bc.is_periodic(b)
        for up in (False, True):
            valid = (H != NODATA) & ((np.abs(H) > HSHIFT / 2.0) == up)
            hv = np.where(valid, np.where(H > HSHIFT / 2.0, H - HSHIFT, np.where(H < -HSHIFT / 2.0, H + HSHIFT, H)),
                          np.inf)
            k = np.argmin(np.abs(hv), axis=1)
            ok = np.isfinite(hv[np.arange(nt), k])
            P = np.where(ok, k + hv[np.arange(nt), k], np.nan)
            if periodic and ok.all() and np.all(np.abs(np.diff(np.r_[P, P[0]])) <= jump):
                runs, per = [(0, nt)], True
            else:
                runs, per = _runs(ok, P, jump), False
            for i0, i1 in runs:
                L = i1 - i0
                if not per and L < min_run:
                    continue
                e = P[i0:i1]
                d1, d2 = _ccd(L, 1.0, per)
                if averages:
                    e = e - (d2 @ e) / 24.0
                e1, e2 = d1 @ e, d2 @ e
                kk = e2 / delta / (1.0 + e1 * e1) ** 1.5
                sl = slice(0, L) if per else slice(trim, L - trim)
                ts = np.arange(i0, i1)[sl]
                kv, sv = kk[sl], e1[sl]
                keep = np.abs(sv) <= max_slope
                ts, kv = ts[keep], kv[keep]
                dist = np.full(ts.shape, float(L)) if per else np.minimum(ts - i0, i1 - 1 - ts).astype(float)
                # the interfacial cells of these columns whose own orientation is the column's
                cells = mine[ts] & ((G[ts] > 0) == up)
                val = np.where(G[ts] < 0, -kv[:, None], kv[:, None])
                tt, ia = np.nonzero(cells)
                if tt.size == 0:
                    continue
                idx = (ts[tt], ia) if a == 1 else (ia, ts[tt])
                cur, cb = kappa[idx], best[idx]
                take = np.isnan(cur) | ((choose == "interior") & (dist[tt] > cb))
                vals = val[tt, ia]
                kappa[idx[0][take], idx[1][take]] = vals[take]
                best[idx[0][take], idx[1][take]] = dist[tt][take]
    if stats is not None:
        stats["ccd"] = int(np.sum(inter & ~np.isnan(kappa)))
        stats["interfacial"] = int(inter.sum())
    out = np.where(inter, np.where(np.isnan(kappa), base, kappa), NODATA)
    return xp.asarray(np.where(out != NODATA, sigma * out, NODATA))

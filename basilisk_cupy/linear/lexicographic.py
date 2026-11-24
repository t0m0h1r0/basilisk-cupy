"""Lexicographic Gauss--Seidel relaxation, as Basilisk's CPU build ("re-use as soon as computed").

Basilisk relaxes in place, cell after cell, ``x`` outer and ``y`` inner, unless
``GAUSS_SEIDEL`` is defined or it runs on a GPU (red/black). Its CPU reference results
(``*.ref``) are obtained that way; at a loose tolerance the result depends on the sweep, so
comparing with them needs the same order. A cell ``(i, j)`` reads ``(i - 1, *)`` and
``(i, j - 1)`` already updated and the others not: on the fronts ``2 i + j = const`` no cell
reads another of the same front, also for the 9-point viscous stencil, so updating front
after front reproduces the sequential sweep exactly. 2-D, uniform grids. The arithmetic follows
Basilisk's (``n = -Delta^2 b + sum alpha a``, ``c = n / sum alpha``).

``set_relaxation("lexicographic")`` switches the Poisson and viscosity multigrid of uniform
2-D grids to it (default ``"redblack"``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

RELAXATION = {"kind": "redblack"}


def set_relaxation(kind: str) -> None:
    if kind not in ("redblack", "lexicographic"):
        raise ValueError("relaxation is 'redblack' or 'lexicographic'")
    RELAXATION["kind"] = kind


def lexicographic() -> bool:
    return RELAXATION["kind"] == "lexicographic"


@lru_cache(maxsize=64)
def _fronts(nx: int, ny: int):
    """Host index arrays ``(I, J)`` of the cells on each front ``2 i + j = d``, d increasing."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    d = (2 * i + j).ravel()
    order = np.argsort(d, kind="stable")
    d_sorted = d[order]
    cuts = np.flatnonzero(np.diff(d_sorted)) + 1
    groups = np.split(order, cuts)
    return [(g // ny, g % ny) for g in groups]


def _dev_fronts(xp, nx, ny):
    return [(xp.asarray(a), xp.asarray(b)) for a, b in _fronts(nx, ny)]


_DEV: dict = {}


def fronts(xp, nx: int, ny: int):
    key = (xp.__name__, nx, ny)
    if key not in _DEV:
        _DEV[key] = _dev_fronts(xp, nx, ny)
    return _DEV[key]


def relax_poisson(xp, a, b, alpha, lam, n, h: float, g: int) -> None:
    """One sweep on the padded array ``a`` (``g`` ghost layers), ``b`` interior, ``alpha`` faces."""
    ax, ay = alpha
    d2 = h * h
    for I, J in fronts(xp, n[0], n[1]):
        nn = -d2 * b[I, J]
        dd = -lam[I, J] * d2 if lam is not None else 0.0
        nn = nn + (ax[I + 1, J] * a[I + 1 + g, J + g] + ax[I, J] * a[I - 1 + g, J + g])
        dd = dd + (ax[I + 1, J] + ax[I, J])
        nn = nn + (ay[I, J + 1] * a[I + g, J + 1 + g] + ay[I, J] * a[I + g, J - 1 + g])
        dd = dd + (ay[I, J + 1] + ay[I, J])
        a[I + g, J + g] = xp.where(dd != 0, nn / xp.where(dd != 0, dd, 1.0), 0.0)


def relax_viscosity(xp, u, r, mu, rho, dt: float, n, h: float, g: int) -> None:
    """One sweep of Basilisk's ``relax_viscosity`` (CPU, ``ua = u``) on the padded components ``u``."""
    ux, uy = u
    mx, my = mu
    d2 = h * h
    for I, J in fronts(xp, n[0], n[1]):
        rr = rho[I, J] if hasattr(rho, "shape") else rho
        X, Y = I + g, J + g
        # x component
        num = (2.0 * mx[I + 1, J] * ux[X + 1, Y] + 2.0 * mx[I, J] * ux[X - 1, Y]
               + my[I, J + 1] * (ux[X, Y + 1] + (uy[X + 1, Y] + uy[X + 1, Y + 1]) / 4.0
                                 - (uy[X - 1, Y] + uy[X - 1, Y + 1]) / 4.0)
               - my[I, J] * (-ux[X, Y - 1] + (uy[X + 1, Y - 1] + uy[X + 1, Y]) / 4.0
                             - (uy[X - 1, Y - 1] + uy[X - 1, Y]) / 4.0))
        den = 2.0 * mx[I + 1, J] + 2.0 * mx[I, J] + my[I, J + 1] + my[I, J]
        ux[X, Y] = (r[0][I, J] * d2 + dt / rr * num) / (d2 + dt / rr * den)
        # y component (Basilisk's foreach_dimension rotation of the above)
        num = (2.0 * my[I, J + 1] * uy[X, Y + 1] + 2.0 * my[I, J] * uy[X, Y - 1]
               + mx[I + 1, J] * (uy[X + 1, Y] + (ux[X, Y + 1] + ux[X + 1, Y + 1]) / 4.0
                                 - (ux[X, Y - 1] + ux[X + 1, Y - 1]) / 4.0)
               - mx[I, J] * (-uy[X - 1, Y] + (ux[X - 1, Y + 1] + ux[X, Y + 1]) / 4.0
                             - (ux[X - 1, Y - 1] + ux[X, Y - 1]) / 4.0))
        den = 2.0 * my[I, J + 1] + 2.0 * my[I, J] + mx[I + 1, J] + mx[I, J]
        uy[X, Y] = (r[1][I, J] * d2 + dt / rr * num) / (d2 + dt / rr * den)

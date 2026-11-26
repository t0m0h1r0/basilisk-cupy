"""Curvature of a VOF circle (R = 0.25, kappa = 4) from tension.h's heights against the CCD column heights
(:func:`basilisk_cupy.interface.cls_heights.vof_height_curvature`, docs/notes/66), with random centre offsets;
the fractions from Basilisk's ``fraction()`` (linear interpolation of the level function on the cell edges) or
exact.

    python -m cases.vof_curvature_ccd
"""

from __future__ import annotations

import numpy as np


def circle_fractions(n: int, cx: float, cy: float, R: float, origin=(-0.5, -0.5), length: float = 1.0):
    """Exact area fractions of the disc ``|x - c| < R`` in the cells of an n x n grid (Gauss--Legendre between
    the breakpoints of the integrand, which is smooth in between)."""
    h = length / n
    xg, wg = np.polynomial.legendre.leggauss(20)
    f = np.zeros((n, n))
    for i in range(n):
        x0, x1 = origin[0] + i * h - cx, origin[0] + (i + 1) * h - cx
        if x1 <= -R or x0 >= R:
            continue
        for j in range(n):
            y0, y1 = origin[1] + j * h - cy, origin[1] + (j + 1) * h - cy
            if min(abs(y0), abs(y1)) > R and y0 * y1 > 0:
                continue
            pts = [max(x0, -R), min(x1, R)]
            for yy in (y0, y1):
                if abs(yy) < R:
                    xs = np.sqrt(R * R - yy * yy)
                    pts += [xs, -xs]
            pts = sorted(p for p in set(pts) if max(x0, -R) <= p <= min(x1, R))
            a = 0.0
            for p0, p1 in zip(pts[:-1], pts[1:]):
                x = 0.5 * (p1 - p0) * xg + 0.5 * (p1 + p0)
                sq = np.sqrt(np.maximum(R * R - x * x, 0.0))
                a += 0.5 * (p1 - p0) * np.sum(wg * np.maximum(np.minimum(y1, sq) - np.maximum(y0, -sq), 0.0))
            f[i, j] = a / (h * h)
    f = np.where(f > 1.0 - 1e-12, 1.0, np.where(f < 1e-12, 0.0, f))  # quadrature round-off in full/empty cells
    return np.clip(f, 0.0, 1.0)


def errors(n: int, method: str, offsets: int = 8, R: float = 0.25, init: str = "exact"):
    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.boundary import BoundaryConditions
    from basilisk_cupy.fields import CellField
    from basilisk_cupy.interface.cls_heights import vof_height_curvature
    from basilisk_cupy.interface.curvature import curvature
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.heights import NODATA

    rng = np.random.default_rng(1)
    e2, emax = [], []
    for _ in range(offsets):
        cx, cy = rng.uniform(-0.5, 0.5, 2) / n
        g = CartesianGrid((n, n), origin=(-0.5, -0.5))
        if init == "exact":
            f0 = circle_fractions(n, cx, cy, R)
        else:  # vertex values of the level function, linearly interpolated (Basilisk's fraction())
            f0 = fraction(g, lambda x, y: R * R - (x - cx) ** 2 - (y - cy) ** 2)
        f = CellField.from_interior(g, f0, BoundaryConditions())
        if method == "columns":
            from basilisk_cupy.interface.vof_curvature import ccd_curvature

            k = ccd_curvature(f, 1.0)
        elif method == "basilisk":
            k = curvature(f.interior, f.bc, g, 1.0)
        else:  # "ccd": point values from the column averages; "ccd-avg": the averages differentiated as they are
            k = vof_height_curvature(f, 1.0, averages=method == "ccd")
        k = np.asarray(k)
        v = k[k != NODATA]
        e2.append(np.sqrt(np.mean((v - 1 / R) ** 2)) * R)
        emax.append(np.max(np.abs(v - 1 / R)) * R)
    return float(np.mean(e2)), float(np.max(emax))


def main() -> None:
    for init in ("exact", "fraction"):
        for method in ("basilisk", "ccd", "columns"):
            for n in (32, 64, 128, 256):
                e2, em = errors(n, method, init=init)
                print(f"{init:8s} {method:8s} N {n:4d}: rms {e2:.2e} max {em:.2e}", flush=True)


if __name__ == "__main__":
    main()

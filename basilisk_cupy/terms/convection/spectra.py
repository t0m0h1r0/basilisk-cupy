"""Spectral extent of the semi-discrete convection operators (linear advection, periodic, speed 1).

``lambda h`` of each operator lies in ``{|Im| <= imag, -real <= Re <= 0}``; an explicit
integrator with imaginary-axis limit ``I`` and real-axis limit ``R`` is stable (on the
axes; complex eigenvalues are checked empirically) for
``dt <= min(I / (imag sum_k |u_k|/h_k), R / (real sum_k |u_k|/h_k))``.

The extents are computed once from the operators themselves (1-D periodic matrices,
64 cells), so they stay consistent with the implementation:

* fd2 central flux ``D_f(P_f q)``: imag 1.000 (``sin theta``);
* FCCD central flux: imag 1.378 (``sin theta (1 + omega_2^2/8)``);
* CCD ``D1`` (UCCD6 advective part): imag 2.126 (at ``theta = 0.75 pi``);
* UCCD6 hyperviscosity ``sigma h^7 D2^4``: real ``sigma omega_2(pi)^8 = 8493.5 sigma``;
* FUCCD_r dissipation: real ``sigma mu_r(pi)`` (``mu_4(pi) = 4954.5``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _matrix(apply, n: int = 64):
    from ...boundary import BoundaryConditions  # noqa: PLC0415
    from ...fields import CellField  # noqa: PLC0415
    from ...grid.cartesian import CartesianGrid  # noqa: PLC0415

    grid = CartesianGrid((n,), length=(1.0,))
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        f = CellField.from_interior(grid, e, BoundaryConditions.periodic(1))
        f.fill_ghosts()
        cols.append(np.asarray(apply(grid, f)))
    return np.array(cols).T * grid.h[0]  # lambda h


@lru_cache(maxsize=None)
def imag_extent(kind: str) -> float:
    """``max |Im(lambda h)|`` of the advective part: ``fd2``, ``fccd`` (central fluxes) or ``ccd`` (D1)."""
    from ...discretization.ccd import CCDCellDerivative, FCCDFaceComplex  # noqa: PLC0415
    from ...discretization.fd2 import FD2FaceComplex  # noqa: PLC0415
    from ...fields import FaceField  # noqa: PLC0415

    if kind == "ccd":
        ccd = CCDCellDerivative()
        apply = lambda g, f: -ccd.d1(f, 0)  # noqa: E731
    else:
        ops = FD2FaceComplex() if kind == "fd2" else FCCDFaceComplex()
        apply = lambda g, f: -ops.div_f(FaceField(g, [ops.interp_f(f, 0)]))  # noqa: E731
    return float(np.abs(np.linalg.eigvals(_matrix(apply)).imag).max())


def advective_rate(u, uf, kind: str) -> float:
    """``imag_extent * sum_k max|u_k| / h_k`` (1/time)."""
    grid = u.grid
    xp = grid.xp
    speeds = [float(xp.max(xp.abs(uf[k] if uf is not None else u[k].interior))) for k in range(grid.ndim)]
    return imag_extent(kind) * sum(s / _min_width(grid, k) for k, s in enumerate(speeds))


def _min_width(grid, axis: int) -> float:
    if grid.axis_uniform(axis):
        return grid.h[axis]
    return float(grid.metric(axis).width.min())

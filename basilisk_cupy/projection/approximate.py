"""Projection of a face velocity field (Basilisk ``project()`` in ``poisson.h``).

Solves ``div(alpha grad p) = div(uf) / dt`` and corrects
``uf <- uf - dt alpha grad p`` so that ``div(uf) = 0`` up to the solver
tolerance. Divergence and face gradient come from the shared
:class:`~basilisk_cupy.discretization.FaceComplex`, which is what keeps the
projection consistent with the face forces added elsewhere.
"""

from __future__ import annotations

from ..discretization.base import FaceComplex
from ..discretization.fd2 import FD2FaceComplex
from ..fields import CellField, FaceField
from ..linear.multigrid import TOLERANCE, MGStats
from ..linear.poisson import poisson


def project(uf: FaceField, p: CellField, alpha: FaceField | None = None, dt: float = 1.0, *,
            ops: FaceComplex | None = None, nrelax: int = 4, tolerance: float = TOLERANCE,
            basilisk_tolerance: bool = False) -> MGStats:
    """``basilisk_tolerance``: solve to ``tolerance / dt^2`` as Basilisk's ``project()`` does, i.e.
    ``tolerance`` bounds the relative volume change of a cell in one step, ``|div u| dt``.
    Otherwise ``tolerance`` bounds the residual ``div u / dt`` itself (the IMEX steppers)."""
    ops = ops or FD2FaceComplex()
    div = ops.div_f(uf, periodic=p.bc.periodic_axes(uf.grid.ndim)) / dt
    tol = tolerance / (dt * dt) if basilisk_tolerance else tolerance
    stats = poisson(p, div, alpha, ops=ops, tolerance=tol, nrelax=nrelax)
    p.fill_ghosts()
    for ax in range(uf.grid.ndim):
        coeff = alpha[ax] if alpha is not None else 1.0
        uf[ax] = uf[ax] - dt * coeff * ops.grad_f(p, ax)
    return stats

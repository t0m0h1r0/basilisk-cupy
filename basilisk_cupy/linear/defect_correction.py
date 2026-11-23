"""Defect correction for ``div(alpha grad a) + lambda a = b`` with a high-order operator.

``L_H`` is the operator the problem is discretised with (e.g. FCCD), ``L_L``
the compact 2nd-order one that multigrid can relax. Plain defect correction

    a <- a + L_L^{-1} (b - L_H a)

contracts by ``max |1 - lambda|`` over the spectrum of ``L_L^{-1} L_H``. For
FCCD on the cell-centred grid that spectrum is real and lies in
``[1, 1.39]`` with Neumann walls but in ``[1, 1.81]`` with Dirichlet walls
(the one-sided FCCD wall gradient is ~1.8 times stiffer than fd2's), so the
plain iteration needs ~100 corrections there.

This module follows TwoPhaseFlow's ``PPESolverDefectCorrection``: every
correction ``z ~ L_L^{-1} r`` is scaled by the minimal-residual step
``alpha = <r, L_H z> / <L_H z, L_H z>``. Keeping the previous directions and
orthogonalising against them turns this into flexible GCR (Eisenstat et
al. 1983); ``memory = 1`` is exactly the line-search defect correction,
larger ``memory`` the Krylov extension. ``L_L^{-1}`` is approximated by a few
cycles of the fd2 multigrid, which GCR tolerates since it is flexible.
"""

from __future__ import annotations

from ..backend import to_float
from ..discretization.base import FaceComplex
from ..discretization.fd2 import FD2FaceComplex
from ..fields import CellField, FaceField
from .multigrid import NITERMAX, TOLERANCE, MGStats, mg_solve
from .poisson import HelmholtzProblem


def _dot(xp, u, v) -> float:
    return to_float(xp.sum(u * v))


def defect_correction(a: CellField, b, alpha: FaceField | None = None, lam=None, *,
                      ops: FaceComplex, low: FaceComplex | None = None,
                      tolerance: float = TOLERANCE, maxiter: int = NITERMAX, memory: int = 10,
                      inner_cycles: int = 2, nrelax: int = 4, min_cells: int = 2,
                      history: list | None = None) -> MGStats:
    """Solve ``L_H(a) = b`` in place; convergence is ``max|b - L_H a| <= tolerance`` (as Basilisk).

    Returns :class:`MGStats` with ``i`` the number of corrections and
    ``nrelax`` the multigrid cycles spent in the ``L_L`` solves.
    """
    grid = a.grid
    xp = grid.xp
    # The work fields stay with ``a`` between calls, so that on the GPU the problems
    # (and the multigrid cycles recorded as CUDA graphs) are reused; their contents
    # are overwritten before use.
    ws = getattr(a, "_dc_workspace", None)
    if ws is None or ws[0].grid != grid or ws[0].bc != a.bc.homogeneous():
        ws = a._dc_workspace = (CellField(grid, bc=a.bc.homogeneous(), name="dc_work"),
                                CellField(grid, bc=a.bc.homogeneous(), name="dc_z"))
    work, z = ws
    high = HelmholtzProblem.reuse(a, alpha, lam, ops=ops, min_cells=min_cells)
    high0 = HelmholtzProblem.reuse(work, alpha, lam, ops=ops, min_cells=min_cells)
    low_problem = HelmholtzProblem.reuse(z, alpha, lam, ops=low or FD2FaceComplex(), min_cells=min_cells)
    zero = xp.zeros(grid.n, dtype=grid.dtype)
    res = xp.zeros(grid.n, dtype=grid.dtype)

    def true_residual():
        a.fill_ghosts()
        return high.residual([a], [b], [res]), res.copy()

    def apply_high0(v):
        work.interior = v
        work.fill_ghosts()
        out = xp.zeros(grid.n, dtype=grid.dtype)
        high0.residual([work], [zero], [out])
        return -out

    def precondition(r):
        z.data[...] = 0.0
        mg_solve(z, r, low_problem, nrelax=nrelax, tolerance=0.0, nitermax=inner_cycles,
                 nitermin=inner_cycles, warn=False)
        return z.interior.copy()

    stats = MGStats(sum=to_float(xp.sum(b)), nrelax=0, minlevel=low_problem.levels[0].n[0], tolerance=tolerance)
    rmax, r = true_residual()
    stats.resb = stats.resa = rmax
    if history is not None:
        history.append(rmax)
    dirs: list = []  # (z, L_H z, |L_H z|^2)
    while stats.i < maxiter and (stats.resa > tolerance or stats.i < 1):
        zk = precondition(r)
        stats.nrelax += inner_cycles
        wk = apply_high0(zk)
        for zj, wj, nj in dirs:
            beta = _dot(xp, wk, wj) / nj
            wk = wk - beta * wj
            zk = zk - beta * zj
        nk = _dot(xp, wk, wk)
        if nk == 0.0:
            break
        step = _dot(xp, r, wk) / nk
        a.interior = a.interior + step * zk
        r = r - step * wk
        dirs.append((zk, wk, nk))
        stats.i += 1
        stats.resa = to_float(xp.max(xp.abs(r)))
        if len(dirs) >= memory or stats.resa <= tolerance:
            # restart from the true residual (the recurrence drifts by round-off);
            # stop only when the true residual is below tolerance
            dirs.clear()
            stats.resa, r = true_residual()
        if history is not None:
            history.append(stats.resa)
    stats.resa, _ = true_residual()
    return stats

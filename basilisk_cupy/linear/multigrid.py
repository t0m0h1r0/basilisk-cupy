"""Generic multigrid solver (port of ``mg_cycle`` / ``mg_solve`` in Basilisk ``poisson.h``).

The solver is written, like Basilisk's, against two user callbacks:

* ``residual(a, b, res) -> max|res|`` evaluates ``res = b - L_high(a)`` on the
  finest level. ``L_high`` is whatever operator the problem is discretised
  with (fd2 today, CCD/FCCD later).
* ``relax(da, res, level)`` performs one smoothing sweep of the *low-order*
  correction problem ``L_low(da) = res`` on a given level.

When ``L_high`` and ``L_low`` coincide this is plain multigrid; when they differ
it is a defect-correction iteration whose inner solver is multigrid. This is
the single place where the two are separated (design rule 4).

Cycle (Basilisk ``mg_cycle``): restrict the residual to every level, then go
from the coarsest level up to the finest; the coarsest correction starts from
zero, every finer one from the bilinear prolongation of the coarser one, and
``nrelax`` sweeps are applied on each level. The finest correction is then
added to the solution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Protocol

from ..backend import to_float
from ..boundary import BoundaryConditions
from ..fields import CellField
from ..grid.cartesian import CartesianGrid

# Basilisk defaults (poisson.h).
NITERMAX = 100
NITERMIN = 1
TOLERANCE = 1e-3


@dataclass
class MGStats:
    """Convergence statistics (Basilisk ``mgstats``)."""

    i: int = 0          # number of cycles
    resb: float = 0.0   # max residual before the cycles
    resa: float = 0.0   # max residual after the cycles
    sum: float = 0.0    # sum of the right-hand side
    nrelax: int = 0     # number of relaxations (after tuning)
    minlevel: int = 0   # cells per axis on the coarsest level used
    tolerance: float = TOLERANCE

    @property
    def converged(self) -> bool:
        return self.resa <= self.tolerance


class MultigridProblem(Protocol):
    levels: list[CartesianGrid]            # coarsest .. finest
    correction_bc: BoundaryConditions      # homogeneous conditions for corrections

    def residual(self, a: CellField, b, res) -> float: ...

    def relax(self, da: CellField, res, level: int) -> None: ...


# -- transfer operators --------------------------------------------------------
def restrict_cell(fine):
    """Average of the 2^d children (interior arrays)."""
    shape = []
    for v in fine.shape:
        shape += [v // 2, 2]
    reduced = fine.reshape(shape)
    return reduced.mean(axis=tuple(range(1, 2 * fine.ndim, 2)))


def restrict_face(fine, axis: int):
    """Coarse face value = average of the fine faces it covers."""
    ndim = fine.ndim
    sl = [slice(None)] * ndim
    sl[axis] = slice(None, None, 2)
    picked = fine[tuple(sl)]
    shape, mean_axes = [], []
    for a, v in enumerate(picked.shape):
        if a == axis:
            shape.append(v)
        else:
            mean_axes.append(len(shape) + 1)
            shape += [v // 2, 2]
    if not mean_axes:
        return picked.copy()
    return picked.reshape(shape).mean(axis=tuple(mean_axes))


def prolong_bilinear(coarse: CellField):
    """Bilinear (trilinear) prolongation of a coarse cell field with filled ghosts.

    Fine children sit at +-1/4 coarse cell; along each axis
    ``fine = 3/4 c[0] + 1/4 c[+-1]``. Applying this axis by axis gives
    Basilisk's ``bilinear()`` weights (9/16, 3/16, 3/16, 1/16 in 2D).
    """
    grid = coarse.grid
    xp = grid.xp
    g = grid.ghosts
    arr = coarse.data
    for axis in range(grid.ndim):
        n = grid.n[axis]

        def take(start: int, stop: int):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(start, stop)
            return arr[tuple(sl)]

        centre = take(g, g + n)
        lower = take(g - 1, g + n - 1)
        upper = take(g + 1, g + n + 1)
        even = 0.75 * centre + 0.25 * lower
        odd = 0.75 * centre + 0.25 * upper
        stacked = xp.stack([even, odd], axis=axis + 1)
        new_shape = list(centre.shape)
        new_shape[axis] = 2 * n
        arr = stacked.reshape(new_shape)
    # Each axis was refined from its own ghost-padded neighbours while the
    # axes not yet processed kept their padding, so ``arr`` now has exactly
    # the fine interior shape.
    return arr


# -- solver ----------------------------------------------------------------------
def mg_cycle(a: CellField, res_levels: list, da_levels: list[CellField], problem: MultigridProblem,
             nrelax: int) -> None:
    nlev = len(problem.levels)
    for lev in range(nlev - 2, -1, -1):
        res_levels[lev] = restrict_cell(res_levels[lev + 1])
    for lev in range(nlev):
        da = da_levels[lev]
        if lev == 0:
            da.data[...] = 0.0
        else:
            coarse = da_levels[lev - 1]
            coarse.fill_ghosts()
            da.interior = prolong_bilinear(coarse)
        for _ in range(nrelax):
            da.fill_ghosts()
            problem.relax(da, res_levels[lev], lev)
    a.interior = a.interior + da_levels[-1].interior


def mg_solve(a: CellField, b, problem: MultigridProblem, *, nrelax: int = 4,
             tolerance: float = TOLERANCE, nitermax: int = NITERMAX, nitermin: int = NITERMIN,
             on_cycle: Callable[[int, float], None] | None = None) -> MGStats:
    """Solve ``L(a) = b`` by repeated multigrid cycles (Basilisk ``mg_solve``)."""
    xp = a.grid.xp
    levels = problem.levels
    da_levels = [CellField(g, bc=problem.correction_bc, name="da") for g in levels]
    res_levels: list = [None] * len(levels)
    res_levels[-1] = xp.zeros(a.grid.n, dtype=a.grid.dtype)

    stats = MGStats(sum=to_float(xp.sum(b)), nrelax=nrelax if nrelax > 0 else 4,
                    minlevel=levels[0].n[0], tolerance=tolerance)
    a.fill_ghosts()
    resb = stats.resb = stats.resa = problem.residual(a, b, res_levels[-1])

    stats.i = 0
    while stats.i < nitermax and (stats.i < nitermin or stats.resa > tolerance):
        mg_cycle(a, res_levels, da_levels, problem, stats.nrelax)
        a.fill_ghosts()
        stats.resa = problem.residual(a, b, res_levels[-1])
        # Tune the number of relaxations so that each cycle reduces the
        # residual by a factor between 2 and 20 (Basilisk heuristic).
        if stats.resa > tolerance:
            if resb / max(stats.resa, 1e-300) < 1.2 and stats.nrelax < 100:
                stats.nrelax += 1
            elif resb / max(stats.resa, 1e-300) > 10 and stats.nrelax > 2:
                stats.nrelax -= 1
        resb = stats.resa
        stats.i += 1
        if on_cycle is not None:
            on_cycle(stats.i, stats.resa)

    if stats.resa > tolerance:
        warnings.warn(
            f"multigrid: convergence for {a.name or 'field'} not reached after {stats.i} iterations "
            f"(res {stats.resa:g}, sum {stats.sum:g}, nrelax {stats.nrelax}, tolerance {tolerance:g})",
            RuntimeWarning, stacklevel=2,
        )
    return stats

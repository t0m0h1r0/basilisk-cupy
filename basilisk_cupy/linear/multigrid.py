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

import itertools

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
    levels: list[CartesianGrid]                  # coarsest .. finest
    correction_bcs: list[BoundaryConditions]     # homogeneous conditions, one per unknown

    def residual(self, a: list[CellField], b: list, res: list) -> float: ...

    def relax(self, da: list[CellField], res: list, level: int) -> None: ...


# -- transfer operators --------------------------------------------------------
def restrict_cell(fine, out=None):
    """Average of the 2^d children (interior arrays).

    Summed one child after another in C order and divided by 2^d, as Basilisk's
    ``restriction_average`` (``foreach_child() sum += s[]; s[] = sum/(1 << dimension)``),
    so the result does not depend on the reduction order of the array library.
    """
    if type(fine).__module__.startswith("cupy") and fine.dtype.name == "float64":
        from ..kernels.transfer_gpu import restrict  # noqa: PLC0415

        return restrict(fine, out)
    total = None
    for child in itertools.product((0, 1), repeat=fine.ndim):
        part = fine[tuple(slice(c, None, 2) for c in child)]
        total = part.copy() if total is None else total + part
    total = total / float(2 ** fine.ndim)
    if out is None:
        return total
    out[...] = total
    return out


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
    if xp.__name__ == "cupy" and arr.dtype == xp.float64 and arr.flags.c_contiguous:
        from ..kernels.transfer_gpu import prolong  # noqa: PLC0415

        return prolong(arr, grid.n, g)
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
def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def mg_cycle(a: list[CellField], res_levels: list[list], da_levels: list[list[CellField]],
             problem: MultigridProblem, nrelax: int) -> None:
    """One Basilisk cycle for the unknowns ``a`` (a list: scalar or vector problems)."""
    nlev = len(problem.levels)
    for lev in range(nlev - 2, -1, -1):
        res_levels[lev] = [restrict_cell(r) for r in res_levels[lev + 1]]
    for lev in range(nlev):
        das = da_levels[lev]
        if lev == 0:
            for da in das:
                da.data[...] = 0.0
        else:
            for da, coarse in zip(das, da_levels[lev - 1]):
                coarse.fill_ghosts()
                da.interior = prolong_bilinear(coarse)
        for _ in range(nrelax):
            for da in das:
                da.fill_ghosts()
            problem.relax(das, res_levels[lev], lev)
    for s, ds in zip(a, da_levels[-1]):
        s.interior = s.interior + ds.interior


def mg_solve(a, b, problem: MultigridProblem, *, nrelax: int = 4,
             tolerance: float = TOLERANCE, nitermax: int = NITERMAX, nitermin: int = NITERMIN,
             on_cycle: Callable[[int, float], None] | None = None, warn: bool = True) -> MGStats:
    """Solve ``L(a) = b`` by repeated multigrid cycles (Basilisk ``mg_solve``).

    ``a`` is a :class:`CellField` or a list of them (e.g. velocity components),
    ``b`` the matching interior array(s). The problem's ``residual`` and
    ``relax`` callbacks receive lists.
    """
    a, b = _as_list(a), _as_list(b)
    grid = a[0].grid
    xp = grid.xp
    levels = problem.levels
    bcs = problem.correction_bcs
    graphs = None
    if xp.__name__ == "cupy":
        from . import cycle_graph  # noqa: PLC0415

        graphs = cycle_graph.cached(problem, a)  # same problem and unknowns: reuse buffers and graphs
    if graphs is not None:
        da_levels, res_levels = graphs.da_levels, graphs.res_levels
    else:
        da_levels = [[CellField(g, bc=bc, name="da") for bc in bcs] for g in levels]
        res_levels: list = [None] * len(levels)
        res_levels[-1] = [xp.zeros(grid.n, dtype=grid.dtype) for _ in a]
        if xp.__name__ == "cupy" and cycle_graph.supported(a, problem, da_levels):
            graphs = cycle_graph.CycleGraphs(a, problem, da_levels, res_levels)

    stats = MGStats(sum=to_float(xp.sum(b[0])), nrelax=nrelax if nrelax > 0 else 4,
                    minlevel=levels[0].n[0], tolerance=tolerance)
    for s in a:
        s.fill_ghosts()
    resb = stats.resb = stats.resa = problem.residual(a, b, res_levels[-1])

    stats.i = 0
    while stats.i < nitermax and (stats.i < nitermin or stats.resa > tolerance):
        if graphs is not None:
            graphs.run(stats.nrelax)
        else:
            mg_cycle(a, res_levels, da_levels, problem, stats.nrelax)
        for s in a:
            s.fill_ghosts()
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

    if warn and stats.resa > tolerance:
        warnings.warn(
            f"multigrid: convergence for {a[0].name or 'field'} not reached after {stats.i} iterations "
            f"(res {stats.resa:g}, sum {stats.sum:g}, nrelax {stats.nrelax}, tolerance {tolerance:g})",
            RuntimeWarning, stacklevel=2,
        )
    return stats

"""Multigrid level hierarchy built by repeated coarsening."""

from __future__ import annotations

from .cartesian import CartesianGrid


def build_levels(grid: CartesianGrid, min_cells: int = 2) -> list[CartesianGrid]:
    """Return ``[coarsest, ..., finest]``.

    Coarsening stops when a further halving would leave an odd count or fewer
    than ``min_cells`` cells on some axis. A mapped grid keeps its map on every
    level (coarse faces are fine faces). Basilisk's ``poisson()`` uses
    ``minlevel = 1`` (two cells per axis); ``min_cells=2`` matches that for
    power-of-two grids.
    """
    levels = [grid]
    g = grid
    while all(v % 2 == 0 and v // 2 >= min_cells for v in g.n):
        g = CartesianGrid(
            n=tuple(v // 2 for v in g.n), length=g.length, origin=g.origin,
            ghosts=g.ghosts, backend=g.backend, dtype=g.dtype, mapping=g.mapping,
        )
        levels.append(g)
    return levels[::-1]

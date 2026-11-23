"""Cell-centred Cartesian grid (Basilisk ``grid/cartesian.h`` / ``grid/multigrid.h``).

Cells are centred, walls sit on cell faces (Basilisk convention, chosen on
2026-11-23 for CCD as well). Each axis keeps its own spacing so that the
non-uniform, interface-fitted grids of TwoPhaseFlow can be added later through
``grid/mapping.py``; for now only uniform spacing is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backend import get_xp


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform cell-centred grid.

    Parameters
    ----------
    n:
        Number of cells per axis, e.g. ``(64, 64)``.
    length:
        Domain length per axis. Defaults to 1 on every axis.
    origin:
        Coordinate of the lower corner. Defaults to 0 on every axis.
    ghosts:
        Ghost-cell width of cell fields. Basilisk uses 1 (``GHOSTS 1``); compact
        schemes with wider closures may ask for more.
    backend:
        "numpy" or "cupy".
    """

    n: tuple[int, ...]
    length: tuple[float, ...] | None = None
    origin: tuple[float, ...] | None = None
    ghosts: int = 1
    backend: str = "numpy"
    dtype: Any = "float64"
    xp: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = tuple(int(v) for v in self.n)
        if not 1 <= len(n) <= 3:
            raise ValueError("only 1D, 2D and 3D grids are supported")
        if any(v < 1 for v in n):
            raise ValueError(f"invalid cell counts {n}")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "length", tuple(float(v) for v in (self.length or (1.0,) * len(n))))
        object.__setattr__(self, "origin", tuple(float(v) for v in (self.origin or (0.0,) * len(n))))
        if len(self.length) != len(n) or len(self.origin) != len(n):
            raise ValueError("length and origin must have one entry per axis")
        if self.ghosts < 1:
            raise ValueError("at least one ghost layer is required")
        object.__setattr__(self, "xp", get_xp(self.backend))

    # -- geometry ---------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.n)

    @property
    def is_uniform(self) -> bool:
        return True

    @property
    def h(self) -> tuple[float, ...]:
        """Cell width per axis."""
        return tuple(L / n for L, n in zip(self.length, self.n))

    @property
    def shape(self) -> tuple[int, ...]:
        """Interior shape (no ghosts)."""
        return self.n

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Shape of a cell field including ghost layers."""
        return tuple(v + 2 * self.ghosts for v in self.n)

    def face_shape(self, axis: int) -> tuple[int, ...]:
        """Shape of the face array normal to ``axis`` (n+1 faces along it)."""
        return tuple(v + 1 if a == axis else v for a, v in enumerate(self.n))

    def centers(self, axis: int, with_ghosts: bool = False):
        """Cell-centre coordinates along ``axis``."""
        xp = self.xp
        g = self.ghosts if with_ghosts else 0
        idx = xp.arange(-g, self.n[axis] + g, dtype=self.dtype)
        return self.origin[axis] + (idx + 0.5) * self.h[axis]

    def faces(self, axis: int):
        """Face coordinates along ``axis`` (n+1 values)."""
        xp = self.xp
        idx = xp.arange(self.n[axis] + 1, dtype=self.dtype)
        return self.origin[axis] + idx * self.h[axis]

    def meshgrid(self, with_ghosts: bool = False):
        """Cell-centre coordinates as ``ndim`` broadcastable arrays (``indexing='ij'``)."""
        axes = [self.centers(a, with_ghosts) for a in range(self.ndim)]
        return self.xp.meshgrid(*axes, indexing="ij")

    def face_meshgrid(self, axis: int):
        """Coordinates of the faces normal to ``axis``."""
        axes = [self.faces(a) if a == axis else self.centers(a) for a in range(self.ndim)]
        return self.xp.meshgrid(*axes, indexing="ij")

    def cell_volume(self) -> float:
        vol = 1.0
        for h in self.h:
            vol *= h
        return vol

    # -- allocation --------------------------------------------------------
    def zeros_cell(self):
        return self.xp.zeros(self.padded_shape, dtype=self.dtype)

    def zeros_face(self, axis: int):
        return self.xp.zeros(self.face_shape(axis), dtype=self.dtype)

    # -- multigrid ----------------------------------------------------------
    def can_coarsen(self) -> bool:
        return all(v % 2 == 0 and v >= 4 for v in self.n)

    def coarsen(self) -> CartesianGrid:
        if not self.can_coarsen():
            raise ValueError(f"cannot coarsen grid with {self.n} cells")
        return CartesianGrid(
            n=tuple(v // 2 for v in self.n),
            length=self.length,
            origin=self.origin,
            ghosts=self.ghosts,
            backend=self.backend,
            dtype=self.dtype,
        )

    # -- slicing helpers ------------------------------------------------------
    def interior(self) -> tuple[slice, ...]:
        g = self.ghosts
        return tuple(slice(g, g + v) for v in self.n)

    def shifted(self, axis: int, offset: int) -> tuple[slice, ...]:
        """Interior slice shifted by ``offset`` cells along ``axis`` (Basilisk ``a[offset]``)."""
        g = self.ghosts
        out = list(self.interior())
        out[axis] = slice(g + offset, g + offset + self.n[axis])
        return tuple(out)

    def face_neighbours(self, axis: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """Slices of the cells left and right of every face normal to ``axis``.

        Face ``i`` lies between cell ``i-1`` and cell ``i``; boundary faces use
        the ghost cells.
        """
        g = self.ghosts
        left = list(self.interior())
        right = list(self.interior())
        left[axis] = slice(g - 1, g + self.n[axis])
        right[axis] = slice(g, g + self.n[axis] + 1)
        return tuple(left), tuple(right)

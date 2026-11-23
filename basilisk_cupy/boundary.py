"""Boundary conditions filled into ghost cells.

Basilisk declares conditions per field and side (``u.n[left] = dirichlet(0)``)
and derives the homogeneous version automatically for multigrid corrections.
The same happens here: every condition has :meth:`homogeneous`.

Walls lie on cell faces (cell-centred grid). With ghost width ``g`` the ghost
cell at distance ``k`` (1..g) from the wall mirrors interior cell ``k``:

* Dirichlet ``a = v`` on the face:  ``ghost_k = 2 v - interior_k``
* Neumann ``da/dn = q`` (outward):  ``ghost_k = interior_k + (2k - 1) h q``
* Periodic: copy from the opposite side.

Ghost layers are filled axis by axis over the full padded extent of the
other axes, so corner ghosts are filled too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

from .grid.cartesian import CartesianGrid

Value = Union[float, Callable[..., object]]

LOWER, UPPER = 0, 1
SIDE_NAMES = {"left": (0, LOWER), "right": (0, UPPER), "bottom": (1, LOWER), "top": (1, UPPER),
              "back": (2, LOWER), "front": (2, UPPER)}


def _side_slice(grid: CartesianGrid, axis: int, index: int) -> tuple[slice, ...]:
    """Padded-array slice selecting one layer ``index`` along ``axis``."""
    out = [slice(None)] * grid.ndim
    out[axis] = slice(index, index + 1)
    return tuple(out)


def _boundary_value(grid: CartesianGrid, value: Value, axis: int, side: int):
    """Evaluate ``value`` on the boundary plane (padded extent of the other axes)."""
    if not callable(value):
        return value
    xp = grid.xp
    coords = []
    for a in range(grid.ndim):
        if a == axis:
            pos = grid.origin[a] + (grid.length[a] if side == UPPER else 0.0)
            coords.append(xp.asarray([pos], dtype=grid.dtype))
        else:
            coords.append(grid.centers(a, with_ghosts=True))
    mesh = xp.meshgrid(*coords, indexing="ij")
    return value(*mesh)


class Condition:
    periodic = False

    def fill(self, grid: CartesianGrid, data, axis: int, side: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def homogeneous(self) -> "Condition":  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class Dirichlet(Condition):
    value: Value = 0.0

    def fill(self, grid, data, axis, side):
        g, n = grid.ghosts, grid.n[axis]
        v = _boundary_value(grid, self.value, axis, side)
        for k in range(1, g + 1):
            if side == LOWER:
                ghost, inner = g - k, g + k - 1
            else:
                ghost, inner = g + n + k - 1, g + n - k
            data[_side_slice(grid, axis, ghost)] = 2.0 * v - data[_side_slice(grid, axis, inner)]

    def homogeneous(self):
        return Dirichlet(0.0)


@dataclass(frozen=True)
class Neumann(Condition):
    """Outward normal derivative ``da/dn`` on the boundary face."""

    gradient: Value = 0.0

    def fill(self, grid, data, axis, side):
        g, n, h = grid.ghosts, grid.n[axis], grid.h[axis]
        q = _boundary_value(grid, self.gradient, axis, side)
        for k in range(1, g + 1):
            if side == LOWER:
                ghost, inner = g - k, g + k - 1
            else:
                ghost, inner = g + n + k - 1, g + n - k
            data[_side_slice(grid, axis, ghost)] = data[_side_slice(grid, axis, inner)] + (2 * k - 1) * h * q

    def homogeneous(self):
        return Neumann(0.0)


@dataclass(frozen=True)
class Periodic(Condition):
    periodic = True

    def fill(self, grid, data, axis, side):
        g, n = grid.ghosts, grid.n[axis]
        for k in range(1, g + 1):
            if side == LOWER:
                data[_side_slice(grid, axis, g - k)] = data[_side_slice(grid, axis, g + n - k)]
            else:
                data[_side_slice(grid, axis, g + n + k - 1)] = data[_side_slice(grid, axis, g + k - 1)]

    def homogeneous(self):
        return self


@dataclass
class BoundaryConditions:
    """Conditions for every (axis, side). Unset sides default to Neumann(0),
    Basilisk's default ("symmetry") condition for scalars."""

    sides: dict = field(default_factory=dict)

    @classmethod
    def periodic(cls, ndim: int) -> "BoundaryConditions":
        p = Periodic()
        return cls({(a, s): p for a in range(ndim) for s in (LOWER, UPPER)})

    @classmethod
    def uniform(cls, ndim: int, condition: Condition) -> "BoundaryConditions":
        return cls({(a, s): condition for a in range(ndim) for s in (LOWER, UPPER)})

    def set(self, side: str | tuple[int, int], condition: Condition) -> "BoundaryConditions":
        key = SIDE_NAMES[side] if isinstance(side, str) else side
        self.sides[key] = condition
        return self

    def get(self, axis: int, side: int) -> Condition:
        return self.sides.get((axis, side), Neumann(0.0))

    def is_periodic(self, axis: int) -> bool:
        lo, hi = self.get(axis, LOWER), self.get(axis, UPPER)
        if lo.periodic != hi.periodic:
            raise ValueError(f"axis {axis}: periodic must be set on both sides")
        return lo.periodic

    def homogeneous(self) -> "BoundaryConditions":
        return BoundaryConditions({k: c.homogeneous() for k, c in self.sides.items()})

    def apply(self, grid: CartesianGrid, data) -> None:
        """Fill the ghost layers of ``data`` (a padded cell array) in place."""
        for axis in range(grid.ndim):
            self.is_periodic(axis)
            for side in (LOWER, UPPER):
                self.get(axis, side).fill(grid, data, axis, side)


def vector_bcs(ndim: int, periodic_axes=(), normal: dict | None = None,
               tangential: dict | None = None) -> list["BoundaryConditions"]:
    """Boundary conditions for the components of a cell-centred vector.

    Basilisk's defaults for a vector are ``antisymmetry`` for the normal
    component (Dirichlet 0 on the wall face: no penetration) and ``symmetry``
    for the tangential ones (Neumann 0: free slip). ``normal`` and
    ``tangential`` override them per side name, e.g.
    ``tangential={"top": Dirichlet(1.0)}`` for a moving lid (Basilisk
    ``u.t[top] = dirichlet(1)``). In 3D a tangential override applies to both
    tangential components.
    """
    normal = normal or {}
    tangential = tangential or {}
    out = []
    for comp in range(ndim):
        bc = BoundaryConditions()
        for axis in range(ndim):
            for side in (LOWER, UPPER):
                if axis in periodic_axes:
                    cond: Condition = Periodic()
                else:
                    name = next(k for k, v in SIDE_NAMES.items() if v == (axis, side))
                    if comp == axis:
                        cond = normal.get(name, Dirichlet(0.0))
                    else:
                        cond = tangential.get(name, Neumann(0.0))
                bc.set((axis, side), cond)
        out.append(bc)
    return out


def gradient_bcs(field_bc: "BoundaryConditions", ndim: int) -> list["BoundaryConditions"]:
    """Conditions Basilisk gives a gradient vector computed from a scalar:
    periodic where the scalar is periodic, otherwise the vector defaults."""
    periodic_axes = [a for a in range(ndim) if field_bc.is_periodic(a)]
    return vector_bcs(ndim, periodic_axes)

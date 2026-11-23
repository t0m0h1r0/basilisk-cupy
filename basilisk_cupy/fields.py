"""Cell- and face-centred fields (Basilisk ``scalar`` and ``face vector``).

* :class:`CellField` stores a padded array (interior plus ghost layers) and
  its boundary conditions.
* :class:`FaceField` stores one array per axis, holding the ``n+1`` faces
  normal to that axis (no ghosts). Boundary faces are part of the array.

Both are thin containers: numerical operations live in ``discretization/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .boundary import BoundaryConditions
from .grid.cartesian import CartesianGrid


@dataclass
class CellField:
    grid: CartesianGrid
    data: object = None
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    name: str = ""

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = self.grid.zeros_cell()
        elif tuple(self.data.shape) != self.grid.padded_shape:
            raise ValueError(f"{self.name}: expected padded shape {self.grid.padded_shape}, got {self.data.shape}")

    @classmethod
    def from_interior(cls, grid: CartesianGrid, values, bc: BoundaryConditions | None = None,
                      name: str = "") -> "CellField":
        f = cls(grid, bc=bc or BoundaryConditions(), name=name)
        f.interior[...] = values
        f.fill_ghosts()
        return f

    @property
    def interior(self):
        return self.data[self.grid.interior()]

    @interior.setter
    def interior(self, values) -> None:
        self.data[self.grid.interior()] = values

    def fill_ghosts(self) -> "CellField":
        self.bc.apply(self.grid, self.data)
        return self

    def copy(self, name: str | None = None) -> "CellField":
        return CellField(self.grid, self.data.copy(), self.bc, name if name is not None else self.name)


@dataclass
class FaceField:
    grid: CartesianGrid
    components: list = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = [self.grid.zeros_face(a) for a in range(self.grid.ndim)]
        for a, c in enumerate(self.components):
            if tuple(c.shape) != self.grid.face_shape(a):
                raise ValueError(f"{self.name}[{a}]: expected {self.grid.face_shape(a)}, got {c.shape}")

    @classmethod
    def constant(cls, grid: CartesianGrid, value: float, name: str = "") -> "FaceField":
        xp = grid.xp
        return cls(grid, [xp.full(grid.face_shape(a), value, dtype=grid.dtype) for a in range(grid.ndim)], name)

    def __getitem__(self, axis: int):
        return self.components[axis]

    def __setitem__(self, axis: int, values) -> None:
        self.components[axis][...] = values

    def copy(self, name: str | None = None) -> "FaceField":
        return FaceField(self.grid, [c.copy() for c in self.components], name if name is not None else self.name)


@dataclass
class VectorField:
    """Cell-centred vector (Basilisk ``vector``): one :class:`CellField` per axis."""

    components: list
    name: str = ""

    @classmethod
    def zeros(cls, grid: CartesianGrid, bcs: list[BoundaryConditions], name: str = "") -> "VectorField":
        return cls([CellField(grid, bc=bc, name=f"{name}.{'xyz'[a]}") for a, bc in enumerate(bcs)], name)

    @property
    def grid(self) -> CartesianGrid:
        return self.components[0].grid

    def __getitem__(self, axis: int) -> CellField:
        return self.components[axis]

    def __iter__(self):
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def fill_ghosts(self) -> "VectorField":
        for c in self.components:
            c.fill_ghosts()
        return self

    def copy(self, name: str | None = None) -> "VectorField":
        return VectorField([c.copy() for c in self.components], name if name is not None else self.name)

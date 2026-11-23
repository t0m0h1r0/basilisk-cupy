"""Cell-centred Cartesian grid (Basilisk ``grid/cartesian.h`` / ``grid/multigrid.h``).

Cells are centred, walls sit on cell faces (Basilisk convention, chosen on
2026-11-23 for CCD as well).

Non-uniform (stretched) axes: an axis may carry a :mod:`~basilisk_cupy.grid.mapping`
``x(xi)`` of the uniform computational coordinate ``xi in [0, L]``. ``h`` is
then the computational spacing ``L/n``; :meth:`centers` and :meth:`faces`
return mapped physical coordinates and :meth:`metric` the metric terms the
operators use (TwoPhaseFlow paper §10, ``10_grid.tex`` §grid_coord,
``10b_ccd_extensions.tex`` §ccd_metric): ``J = dxi/dx`` and ``dJ/dxi`` at the
centres from the CCD derivatives of the centre coordinates (Dirichlet closure
with the wall coordinates as data), ``J`` at the faces from the FCCD face
gradient of the same coordinates, the physical cell widths and centre
distances. Deriving the metric from the coordinates with the same operators
makes ``d x / d x = 1`` exact (geometric conservation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import get_xp
from .mapping import Mapping, UniformMapping


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
    mapping:
        Optional per-axis :class:`~basilisk_cupy.grid.mapping.Mapping` (``None``
        entries or no tuple: uniform).
    """

    n: tuple[int, ...]
    length: tuple[float, ...] | None = None
    origin: tuple[float, ...] | None = None
    ghosts: int = 1
    backend: str = "numpy"
    dtype: Any = "float64"
    mapping: tuple | None = None
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
        if self.mapping is not None:
            maps = tuple(m if m is not None else UniformMapping() for m in self.mapping)
            if len(maps) != len(n) or not all(isinstance(m, Mapping) for m in maps):
                raise ValueError("mapping must give one Mapping (or None) per axis")
            object.__setattr__(self, "mapping", None if all(m.is_uniform for m in maps) else maps)

    # -- geometry ---------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.n)

    @property
    def is_uniform(self) -> bool:
        return self.mapping is None

    def axis_uniform(self, axis: int) -> bool:
        return self.mapping is None or self.mapping[axis].is_uniform

    def _coordinates(self, axis: int, xi):
        """Physical coordinates of computational positions ``xi`` (host array) along ``axis``."""
        L, o = self.length[axis], self.origin[axis]
        if self.axis_uniform(axis):
            return o + np.asarray(xi, dtype=np.float64)
        return o + L * self.mapping[axis].extended(np.asarray(xi, dtype=np.float64) / L)

    def metric(self, axis: int) -> "AxisMetric":
        """Metric terms along ``axis`` (cached per grid)."""
        return self._metrics[axis]

    @cached_property
    def _metrics(self):
        return tuple(AxisMetric(self, a) for a in range(self.ndim))

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
        """Cell-centre coordinates along ``axis`` (mapped on a non-uniform axis; ghosts mirrored)."""
        xp = self.xp
        g = self.ghosts if with_ghosts else 0
        if self.axis_uniform(axis):
            idx = xp.arange(-g, self.n[axis] + g, dtype=self.dtype)
            return self.origin[axis] + (idx + 0.5) * self.h[axis]
        idx = np.arange(-g, self.n[axis] + g, dtype=np.float64)
        return xp.asarray(self._coordinates(axis, (idx + 0.5) * self.h[axis]), dtype=self.dtype)

    def faces(self, axis: int):
        """Face coordinates along ``axis`` (n+1 values)."""
        xp = self.xp
        if self.axis_uniform(axis):
            idx = xp.arange(self.n[axis] + 1, dtype=self.dtype)
            return self.origin[axis] + idx * self.h[axis]
        idx = np.arange(self.n[axis] + 1, dtype=np.float64)
        return xp.asarray(self._coordinates(axis, idx * self.h[axis]), dtype=self.dtype)

    def meshgrid(self, with_ghosts: bool = False):
        """Cell-centre coordinates as ``ndim`` broadcastable arrays (``indexing='ij'``)."""
        axes = [self.centers(a, with_ghosts) for a in range(self.ndim)]
        return self.xp.meshgrid(*axes, indexing="ij")

    def face_meshgrid(self, axis: int):
        """Coordinates of the faces normal to ``axis``."""
        axes = [self.faces(a) if a == axis else self.centers(a) for a in range(self.ndim)]
        return self.xp.meshgrid(*axes, indexing="ij")

    def cell_volume(self) -> float:
        """Volume of a cell of a uniform grid (see :meth:`cell_volumes` otherwise)."""
        if not self.is_uniform:
            raise ValueError("non-uniform grid: cells differ in volume, use cell_volumes()")
        vol = 1.0
        for h in self.h:
            vol *= h
        return vol

    def cell_volumes(self):
        """Physical cell volumes (interior shape; a scalar on a uniform grid)."""
        if self.is_uniform:
            return self.cell_volume()
        vol = None
        for a in range(self.ndim):
            w = self.metric(a).along(self.metric(a).width)
            vol = w if vol is None else vol * w
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
            mapping=self.mapping,
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

    def at(self, offsets: dict[int, int] | None = None) -> tuple[slice, ...]:
        """Interior slice shifted by ``offsets`` (axis -> cells), e.g. Basilisk ``a[1,-1]`` is
        ``at({0: 1, 1: -1})``."""
        g = self.ghosts
        out = list(self.interior())
        for axis, off in (offsets or {}).items():
            out[axis] = slice(g + off, g + off + self.n[axis])
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


class AxisMetric:
    """Metric terms of one axis of a (possibly) non-uniform grid, on the grid's backend.

    Arrays are 1-D along the axis; :meth:`along` reshapes one for broadcasting.

    * ``J`` (n), ``dJ`` (n): ``dxi/dx`` and ``d(dxi/dx)/dxi`` at the centres;
    * ``J_face`` (n+1): ``dxi/dx`` at the faces;
    * ``width`` (n): physical cell widths ``x_{i+1/2} - x_{i-1/2}``;
    * ``spacing`` (n+1): centre distances across each face (ghost centres
      mirrored at walls, translated on a periodic axis);
    * ``x`` (n), ``x_face`` (n+1): coordinates;
    * ``wall_x_xi``: exact ``dx/dxi`` at the two walls (from the map), which
      converts Neumann data ``df/dx`` to the ``df/dxi`` of the closures (the
      FCCD face metric is only 4th order and would cost the CCD wall rows 1-2
      orders).

    On a uniform axis ``J = 1``, ``dJ = 0``, ``width = spacing = h``.
    """

    def __init__(self, grid: CartesianGrid, axis: int) -> None:
        self.grid, self.axis = grid, axis
        n, h = grid.n[axis], grid.h[axis]
        xp = grid.xp
        self.uniform = grid.axis_uniform(axis)
        xc = grid._coordinates(axis, (np.arange(n) + 0.5) * h)
        xf = grid._coordinates(axis, np.arange(n + 1) * h)
        xg = grid._coordinates(axis, (np.arange(-1, n + 1) + 0.5) * h)  # one ghost centre per side
        dev = self._dev = lambda a: xp.asarray(a, dtype=grid.dtype)  # noqa: E731
        self.x, self.x_face = dev(xc), dev(xf)
        self.width = dev(np.diff(xf))
        self.spacing = dev(np.diff(xg))
        # 3-point first/second derivative weights (left, centre, right) with the neighbour distances
        hl, hr = np.diff(xg)[:-1], np.diff(xg)[1:]
        self.d1_weights = tuple(dev(w) for w in (-hr / (hl * (hl + hr)), (hr - hl) / (hl * hr), hl / (hr * (hl + hr))))
        self.d2_weights = tuple(dev(w) for w in (2 / (hl * (hl + hr)), -2 / (hl * hr), 2 / (hr * (hl + hr))))
        self._host = {"x": xc, "x_face": xf, "x_ghost": xg}
        # exact dx/dxi at the two walls (from the map), for Neumann data in the CCD/FCCD wall closures
        self.wall_x_xi = (1.0, 1.0) if self.uniform else tuple(
            float(v) for v in grid.mapping[axis].derivative(np.array([0.0, 1.0])))

    @cached_property
    def _derivatives(self):
        """Host ``dx/dxi``, ``d2x/dxi2`` (centres) and ``dx/dxi`` (faces), computed on first use."""
        grid, axis = self.grid, self.axis
        n = grid.n[axis]
        if self.uniform:
            return np.ones(n), np.zeros(n), np.ones(n + 1)
        periodic = grid.mapping[axis].periodic
        if n >= (3 if periodic else 6):
            return _coordinate_derivatives(grid, axis, self._host["x"], self._host["x_face"])
        # too few cells for a CCD stencil (coarse multigrid levels): the map's exact derivatives
        mp, L, h = grid.mapping[axis], grid.length[axis], grid.h[axis]
        sc, sf = (np.arange(n) + 0.5) * h / L, np.arange(n + 1) * h / L
        eps = 1e-5
        x_xi = mp.derivative(sc)
        x_xixi = (mp.derivative(np.clip(sc + eps, 0, 1)) - mp.derivative(np.clip(sc - eps, 0, 1))) / (2 * eps * L)
        return x_xi, x_xixi, mp.derivative(sf)

    @cached_property
    def J(self):
        return self._dev(1.0 / self._derivatives[0])

    @cached_property
    def dJ(self):
        x_xi, x_xixi, _ = self._derivatives
        return self._dev(-x_xixi / x_xi ** 2)

    @cached_property
    def J_face(self):
        return self._dev(1.0 / self._derivatives[2])

    def J_face_for(self, order: int):
        """``J`` at the faces from the FCCD face gradient of order ``order`` (4: :attr:`J_face`, or 6),
        so that each operator differentiates the coordinates exactly (geometric conservation)."""
        if order == 4:
            return self.J_face
        cache = self.__dict__.setdefault("_J_face_by_order", {})
        if order not in cache:
            grid, axis = self.grid, self.axis
            if self.uniform:
                cache[order] = self.J_face
            elif grid.n[axis] >= (3 if grid.mapping[axis].periodic else 6):
                face = _coordinate_derivatives(grid, axis, self._host["x"], self._host["x_face"], face_order=order)[2]
                cache[order] = self._dev(1.0 / face)
            else:
                cache[order] = self.J_face  # coarse levels: the map's exact derivative (as J_face)
        return cache[order]

    def along(self, arr, ndim: int | None = None):
        """``arr`` reshaped to broadcast along this axis of an ``ndim``-d array."""
        ndim = self.grid.ndim if ndim is None else ndim
        shape = [1] * ndim
        shape[self.axis] = -1
        return arr.reshape(shape)

    def wall_distance(self, side: int, layer: int) -> float:
        """Physical distance between ghost layer ``layer`` (1, 2, ...) and its mirror cell."""
        g = self.grid
        n = g.n[self.axis]
        k = layer - 1
        xi = ((k + 0.5) if side == 0 else (n - k - 0.5)) * g.h[self.axis]
        x_in = float(g._coordinates(self.axis, np.array([xi]))[0])
        x_wall = g.origin[self.axis] + (0.0 if side == 0 else g.length[self.axis])
        return 2.0 * abs(x_in - x_wall)


def _coordinate_derivatives(grid: CartesianGrid, axis: int, xc, xf, face_order: int = 4):
    """``dx/dxi``, ``d2x/dxi2`` at the centres (CCD) and ``dx/dxi`` at the faces (FCCD), host arrays."""
    from ..boundary import BoundaryConditions, Dirichlet, Periodic  # noqa: PLC0415
    from ..discretization.ccd import CCDCellDerivative, FCCDFaceComplex  # noqa: PLC0415
    from ..fields import CellField  # noqa: PLC0415

    n, h, L = grid.n[axis], grid.h[axis], grid.length[axis]
    line = CartesianGrid((n,), length=(L,))  # uniform computational line, host
    xi_c = (np.arange(n) + 0.5) * h
    periodic = grid.mapping[axis].periodic
    if periodic:  # x - xi is periodic
        values = xc - grid.origin[axis] - xi_c
        bc = BoundaryConditions().set((0, 0), Periodic()).set((0, 1), Periodic())
    else:
        values = xc - grid.origin[axis]
        bc = (BoundaryConditions().set((0, 0), Dirichlet(xf[0] - grid.origin[axis]))
              .set((0, 1), Dirichlet(xf[-1] - grid.origin[axis])))
    f = CellField.from_interior(line, values, bc)
    d1, d2 = CCDCellDerivative().derivatives(f, 0)
    face = FCCDFaceComplex(order=face_order).grad_f(f, 0)
    if periodic:
        d1, face = d1 + 1.0, face + 1.0
    if np.any(d1 <= 0.0) or np.any(face <= 0.0):
        raise ValueError("the mapping is not increasing on this grid")
    return np.asarray(d1), np.asarray(d2), np.asarray(face)

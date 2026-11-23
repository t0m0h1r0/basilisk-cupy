"""Interfaces of the spatial-operator layer (layer 1 of the design).

Two families, matching how Basilisk uses its stencils:

* :class:`FaceComplex` maps between cells and faces. Projection, surface
  tension, gravity and interface transport must all use the *same* instance,
  so the pressure gradient and the face forces are discretised identically
  (balanced force). Basilisk achieves this by writing ``(p[] - p[-1])/Delta``
  everywhere; here it is enforced by sharing one object.
* :class:`CellDerivative` gives cell-centred derivatives (BCG slopes,
  curvature, ...).

Inputs are :class:`~basilisk_cupy.fields.CellField` objects whose ghost cells
have already been filled; outputs are plain arrays (face arrays of shape
``grid.face_shape(axis)`` or interior arrays of shape ``grid.n``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FaceComplex(Protocol):
    name: str
    #: Ghost layers the operator reads on each side.
    stencil_ghosts: int

    def grad_f(self, a, axis: int):
        """Normal derivative of cell field ``a`` on the faces normal to ``axis``."""

    def interp_f(self, a, axis: int):
        """Value of cell field ``a`` interpolated to the faces normal to ``axis``."""

    def div_f(self, flux, periodic=None):
        """Cell-centred divergence of a face field (sum over axes of the face difference).

        ``periodic`` (one flag per axis) is needed by operators whose wall rows differ (FCCD).
        """

    def face_to_cell(self, face_values, axis: int, periodic: bool | None = None):
        """Cell-centred value reconstructed from the faces normal to ``axis``.

        ``periodic`` tells whether ``axis`` is periodic; wide stencils need it
        to choose between wrapping and a wall closure.
        """


@runtime_checkable
class CellDerivative(Protocol):
    name: str
    stencil_ghosts: int

    def d1(self, a, axis: int):
        """First derivative at cell centres."""

    def d2(self, a, axis: int):
        """Second derivative at cell centres."""

    def d11(self, a, axis0: int, axis1: int):
        """Mixed derivative at cell centres."""

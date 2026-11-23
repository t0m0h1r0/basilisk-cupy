"""Second-order operators reproducing Basilisk's inline stencils.

=====================  =========================================  ===================================
operator               formula                                    Basilisk source
=====================  =========================================  ===================================
``grad_f``             ``(a[0] - a[-1]) / Delta``                 ``face_gradient_x``, ``centered.h``
``interp_f``           ``(a[0] + a[-1]) / 2``                     ``face_value`` (``centered.h``)
``div_f``              ``sum (F[1] - F[0]) / Delta``              ``project()`` in ``poisson.h``
``face_to_cell``       ``(F[0] + F[1]) / 2``                      ``centered_gradient()``
``d1``                 ``(a[1] - a[-1]) / (2 Delta)``             ``prediction()`` without limiter
``d2``                 ``(a[1] - 2 a[0] + a[-1]) / Delta^2``      ---
``d11``                centred cross difference                   ---
=====================  =========================================  ===================================
"""

from __future__ import annotations

from ..fields import CellField, FaceField


def _shift(field: CellField, axis: int, offset: int):
    return field.data[field.grid.shifted(axis, offset)]


class FD2FaceComplex:
    name = "fd2"
    stencil_ghosts = 1

    def grad_f(self, a: CellField, axis: int):
        left, right = a.grid.face_neighbours(axis)
        return (a.data[right] - a.data[left]) / a.grid.h[axis]

    def interp_f(self, a: CellField, axis: int):
        left, right = a.grid.face_neighbours(axis)
        return 0.5 * (a.data[right] + a.data[left])

    def div_f(self, flux: FaceField):
        grid = flux.grid
        out = None
        for axis in range(grid.ndim):
            f = flux[axis]
            lo = [slice(None)] * grid.ndim
            hi = [slice(None)] * grid.ndim
            lo[axis] = slice(0, grid.n[axis])
            hi[axis] = slice(1, grid.n[axis] + 1)
            term = (f[tuple(hi)] - f[tuple(lo)]) / grid.h[axis]
            out = term if out is None else out + term
        return out

    def face_to_cell(self, face_values, axis: int):
        n = face_values.shape[axis] - 1
        lo = [slice(None)] * face_values.ndim
        hi = [slice(None)] * face_values.ndim
        lo[axis] = slice(0, n)
        hi[axis] = slice(1, n + 1)
        return 0.5 * (face_values[tuple(lo)] + face_values[tuple(hi)])


class FD2CellDerivative:
    name = "fd2"
    stencil_ghosts = 1

    def d1(self, a: CellField, axis: int):
        return (_shift(a, axis, 1) - _shift(a, axis, -1)) / (2.0 * a.grid.h[axis])

    def d2(self, a: CellField, axis: int):
        return (_shift(a, axis, 1) - 2.0 * a.interior + _shift(a, axis, -1)) / a.grid.h[axis] ** 2

    def d11(self, a: CellField, axis0: int, axis1: int):
        if axis0 == axis1:
            return self.d2(a, axis0)
        grid = a.grid

        def corner(s0: int, s1: int):
            sl = list(grid.interior())
            g = grid.ghosts
            sl[axis0] = slice(g + s0, g + s0 + grid.n[axis0])
            sl[axis1] = slice(g + s1, g + s1 + grid.n[axis1])
            return a.data[tuple(sl)]

        return (corner(1, 1) - corner(1, -1) - corner(-1, 1) + corner(-1, -1)) / (
            4.0 * grid.h[axis0] * grid.h[axis1]
        )

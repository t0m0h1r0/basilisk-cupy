"""Second-order operators reproducing Basilisk's inline stencils.

On a non-uniform (mapped) axis the gradient divides by the physical distance of
the two centres, the divergence by the physical cell width (finite volume) and
the cell derivatives use the non-uniform 3-point formulas; values are still
interpolated with equal weights in the computational coordinate, which stays
2nd order for a smooth map. Uniform axes run the original code.

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
        grid = a.grid
        left, right = grid.face_neighbours(axis)
        if grid.axis_uniform(axis):
            return (a.data[right] - a.data[left]) / grid.h[axis]
        m = grid.metric(axis)  # non-uniform: distance between the two centres
        return (a.data[right] - a.data[left]) / m.along(m.spacing)

    def interp_f(self, a: CellField, axis: int):
        # equal weights in the computational coordinate: 2nd order on a smoothly mapped axis too
        left, right = a.grid.face_neighbours(axis)
        return 0.5 * (a.data[right] + a.data[left])

    def div_f(self, flux: FaceField, periodic=None):
        grid = flux.grid
        out = None
        for axis in range(grid.ndim):
            f = flux[axis]
            lo = [slice(None)] * grid.ndim
            hi = [slice(None)] * grid.ndim
            lo[axis] = slice(0, grid.n[axis])
            hi[axis] = slice(1, grid.n[axis] + 1)
            width = grid.h[axis] if grid.axis_uniform(axis) else grid.metric(axis).along(grid.metric(axis).width)
            term = (f[tuple(hi)] - f[tuple(lo)]) / width
            out = term if out is None else out + term
        return out

    def face_to_cell(self, face_values, axis: int, periodic: bool | None = None):
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
        if not a.grid.axis_uniform(axis):
            return self._weighted(a, axis, a.grid.metric(axis).d1_weights)
        return (_shift(a, axis, 1) - _shift(a, axis, -1)) / (2.0 * a.grid.h[axis])

    def d2(self, a: CellField, axis: int):
        if not a.grid.axis_uniform(axis):
            return self._weighted(a, axis, a.grid.metric(axis).d2_weights)
        return (_shift(a, axis, 1) - 2.0 * a.interior + _shift(a, axis, -1)) / a.grid.h[axis] ** 2

    @staticmethod
    def _weighted(a: CellField, axis: int, weights):
        """Non-uniform 3-point formula with the centre distances (TwoPhaseFlow ``_low_order_derivative``)."""
        m = a.grid.metric(axis)
        wl, wc, wr = (m.along(w) for w in weights)
        return wl * _shift(a, axis, -1) + wc * a.interior + wr * _shift(a, axis, 1)

    def d11(self, a: CellField, axis0: int, axis1: int):
        if axis0 == axis1:
            return self.d2(a, axis0)
        grid = a.grid
        if not (grid.axis_uniform(axis0) and grid.axis_uniform(axis1)):
            # tensor product of the two 3-point first-derivative stencils
            g = grid.ghosts
            m0, m1 = grid.metric(axis0), grid.metric(axis1)
            out = 0.0
            for s0, w0 in zip((-1, 0, 1), m0.d1_weights):
                for s1, w1 in zip((-1, 0, 1), m1.d1_weights):
                    sl = list(grid.interior())
                    sl[axis0] = slice(g + s0, g + s0 + grid.n[axis0])
                    sl[axis1] = slice(g + s1, g + s1 + grid.n[axis1])
                    out = out + m0.along(w0) * m1.along(w1) * a.data[tuple(sl)]
            return out

        def corner(s0: int, s1: int):
            sl = list(grid.interior())
            g = grid.ghosts
            sl[axis0] = slice(g + s0, g + s0 + grid.n[axis0])
            sl[axis1] = slice(g + s1, g + s1 + grid.n[axis1])
            return a.data[tuple(sl)]

        return (corner(1, 1) - corner(1, -1) - corner(-1, 1) + corner(-1, -1)) / (
            4.0 * grid.h[axis0] * grid.h[axis1]
        )

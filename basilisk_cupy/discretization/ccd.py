"""CCD and FCCD operators on Basilisk's cell-centred grid.

Interior equations are those of TwoPhaseFlow (``src/twophase/ccd``):

* **CCD** (Chu & Fan 1998), 6th order, solved for ``f'`` and ``f''`` together::

      7/16 (f'_{i-1} + f'_{i+1}) + f'_i - (h/16)(f''_{i+1} - f''_{i-1}) = 15/(16h) (f_{i+1} - f_{i-1})
      -1/8 (f''_{i-1} + f''_{i+1}) + f''_i + 9/(8h)(f'_{i+1} - f'_{i-1}) = 3/h^2 (f_{i-1} - 2 f_i + f_{i+1})

* **FCCD** (SP-C/SP-D), 4th order on faces, using ``q = f''`` from CCD::

      grad_f   = (u_i - u_{i-1})/h - (h/24)(q_i - q_{i-1})
      interp_f = (u_{i-1} + u_i)/2 - (h^2/16)(q_{i-1} + q_i)
      div_f    = conservative face difference (as fd2 and TwoPhaseFlow's ``face_divergence``)

Walls are on faces, half a cell from the first centre (TwoPhaseFlow uses a
nodal grid with the wall on a node). The first-cell CCD rows and the
wall-face FCCD values are new closures that take the wall datum from the
field's boundary condition (Dirichlet value or Neumann gradient), derived in
``tools/derive_ccd_closures.py`` (coefficients in :mod:`.ccd_coefficients`).
The closures are compact: they read the wall datum and the first two cells only
(the width of the interior three-cell stencil) and take their accuracy from the
implicit derivatives, as the interior rows do. The CCD wall-cell ``f'`` is 4th
and ``f''`` 3rd order (6th inside); the wall-face FCCD values read the CCD jet
``(f, f', f'')`` of the first cells and are 4th order like the interior faces.

Because the grid is fixed, the operator along each axis is a dense matrix
``[f'; f''] = D f + b_lo datum_lo + b_hi datum_hi`` built once (block solve on
the host) and applied with a matrix product on the backend, which is also
how TwoPhaseFlow applies CCD on the GPU.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..backend import array_module
from ..boundary import LOWER, UPPER, Dirichlet, Neumann, _boundary_value
from ..fields import CellField, FaceField
from .ccd_coefficients import CCD_WALL, FCCD_WALL
from .fd2 import FD2FaceComplex

# Chu & Fan (1998), Table 1 (TwoPhaseFlow ccd_solver.py)
ALPHA1, A1, B1 = 7.0 / 16.0, 15.0 / 16.0, 1.0 / 16.0
BETA2, A2, B2 = -1.0 / 8.0, 3.0, -9.0 / 8.0

PERIODIC, DIRICHLET, NEUMANN, EXTRAPOLATE = "periodic", "dirichlet", "neumann", "extrapolate"


# -- 1-D operator matrices ----------------------------------------------------------
def _closure_rows(M, R, datum, kind: str, side: int, n: int, h: float) -> None:
    """First-cell rows (or last-cell rows, by reflection) of the CCD system.

    In the reflected coordinate ``s`` (``s = -1`` at the upper wall) the
    closure reads ``f_s^{(m)}_0 + sum c h^{d-m} f_s^{(d)}_k = h^{-m} (B datum_s + sum r f_k)``
    with ``f_s^{(d)} = s^d f^{(d)}``; multiplying by ``s^m`` gives the rows below.
    The datum is the Dirichlet value or the Neumann gradient ``df/dx``;
    ``datum_s = value`` or ``h s df/dx``.
    """
    s = 1 if side == LOWER else -1
    k0 = 0 if side == LOWER else n - 1
    rows = CCD_WALL[kind]
    for main, name in ((1, "I"), (2, "II")):
        c = rows[name]
        implicit = c["terms"]
        row = 2 * k0 + main - 1
        sm = s ** main
        M[row, row] = 1.0
        for (d, k), v in zip(implicit, c["implicit"]):
            M[row, 2 * (k0 + s * k) + d - 1] += float(v) * s ** d * sm * h ** (d - main)
        explicit = [float(v) for v in c["explicit"]]
        if kind != EXTRAPOLATE:
            b, explicit = explicit[0], explicit[1:]
            datum[row] = b * sm / h ** main * (1.0 if kind == DIRICHLET else s * h)
        for k, v in enumerate(explicit):
            R[row, k0 + s * k] += v * sm / h ** main


@lru_cache(maxsize=64)
def ccd_matrices(n: int, h: float, lo: str, hi: str):
    """Host matrices ``(D1, D2, b1_lo, b1_hi, b2_lo, b2_hi)`` of the CCD operator along one axis.

    ``f' = D1 f + b1_lo datum_lo + b1_hi datum_hi`` (same for ``f''``).
    ``lo``/``hi`` are ``periodic`` (both), ``dirichlet``, ``neumann`` or ``extrapolate``.
    """
    if (lo == PERIODIC) != (hi == PERIODIC):
        raise ValueError("periodic must be set on both sides")
    if n < (3 if lo == PERIODIC else 6):
        raise ValueError(f"CCD needs at least {3 if lo == PERIODIC else 6} cells per axis, got {n}")
    M = np.zeros((2 * n, 2 * n))
    R = np.zeros((2 * n, n))
    datum = np.zeros((2 * n, 2))
    periodic = lo == PERIODIC
    for k in range(n) if periodic else range(1, n - 1):
        km, kp = (k - 1) % n, (k + 1) % n
        i, j = 2 * k, 2 * k + 1
        M[i, i] = 1.0
        M[i, 2 * km] += ALPHA1
        M[i, 2 * kp] += ALPHA1
        M[i, 2 * km + 1] += B1 * h
        M[i, 2 * kp + 1] -= B1 * h
        R[i, kp] += A1 / h
        R[i, km] -= A1 / h
        M[j, j] = 1.0
        M[j, 2 * km + 1] += BETA2
        M[j, 2 * kp + 1] += BETA2
        M[j, 2 * km] += B2 / h
        M[j, 2 * kp] -= B2 / h
        R[j, km] += A2 / h ** 2
        R[j, kp] += A2 / h ** 2
        R[j, k] -= 2 * A2 / h ** 2
    if not periodic:
        _closure_rows(M, R, datum[:, 0], lo, LOWER, n, h)
        _closure_rows(M, R, datum[:, 1], hi, UPPER, n, h)
    T = np.linalg.solve(M, np.concatenate([R, datum], axis=1))
    d1, d2 = T[0::2], T[1::2]
    return (d1[:, :n].copy(), d2[:, :n].copy(),
            d1[:, n].copy(), d1[:, n + 1].copy(), d2[:, n].copy(), d2[:, n + 1].copy())


# -- boundary data --------------------------------------------------------------------
def _kind(cond) -> str:
    if cond.periodic:
        return PERIODIC
    if isinstance(cond, Dirichlet):
        return DIRICHLET
    if isinstance(cond, Neumann):
        return NEUMANN
    raise TypeError(f"CCD has no closure for {type(cond).__name__}")


def _wall_datum(f: CellField, axis: int, side: int):
    """Wall datum on the boundary plane (other axes interior, axis dropped).

    Dirichlet: the value. Neumann: ``df/dxi`` along +axis (the condition stores
    the outward physical gradient; on a non-uniform axis it is multiplied by
    ``dx/dxi`` at the wall face, so that the closures work in ``xi``).
    """
    grid = f.grid
    cond = f.bc.get(axis, side)
    raw = _boundary_value(grid, cond.value if isinstance(cond, Dirichlet) else cond.gradient, axis, side)
    if not hasattr(raw, "shape") or raw.ndim == 0:
        val = raw
    else:
        sl = list(grid.interior())
        sl[axis] = 0
        val = raw[tuple(sl)]
    if isinstance(cond, Neumann) and side == LOWER:
        val = -val
    if isinstance(cond, Neumann) and not grid.axis_uniform(axis):
        val = val * grid.metric(axis).wall_x_xi[0 if side == LOWER else 1]
    return val


def _to_physical(grid, axis: int, g1, g2):
    """Chain rule ``f_x = J f_xi``, ``f_xx = J^2 f_xixi + J J_xi f_xi`` (paper eqs. transform_1st/2nd)."""
    if grid.axis_uniform(axis):
        return g1, g2
    m = grid.metric(axis)
    J = m.along(m.J, g1.ndim if g1 is not None else g2.ndim)
    d1 = None if g1 is None else J * g1
    d2 = None if g2 is None else J * J * g2 + J * m.along(m.dJ, g2.ndim) * g1
    return d1, d2


def _moved(arr, axis):
    return arr.swapaxes(axis, -1)


class _OperatorCache:
    """Device copies of the host matrices, keyed by (matrix id, dtype)."""

    def __init__(self) -> None:
        self._store: dict = {}

    def get(self, xp, host, dtype):
        key = (id(host), xp.__name__, str(dtype))
        hit = self._store.get(key)
        if hit is None or hit[0] is not host:
            hit = (host, xp.asarray(host, dtype=dtype))
            self._store[key] = hit
        return hit[1]


# -- CCD cell derivatives ----------------------------------------------------------------
class CCDCellDerivative:
    """Cell-centred CCD derivatives (CellDerivative protocol)."""

    name = "ccd"
    stencil_ghosts = 1  # reads the interior and the boundary conditions only

    def __init__(self) -> None:
        self._dev = _OperatorCache()

    def _kinds(self, f: CellField, axis: int):
        return _kind(f.bc.get(axis, LOWER)), _kind(f.bc.get(axis, UPPER))

    def _apply(self, xp, values, axis, mats, which: int, data=None):
        D = self._dev.get(xp, mats[which], values.dtype)
        out = _moved(_moved(values, axis) @ D.T, axis)
        for side, val in data or ():
            coef = self._dev.get(xp, mats[2 + 2 * which + side], out.dtype)
            shape = [1] * out.ndim
            shape[axis] = -1
            if getattr(val, "ndim", 0) > 0:
                val = xp.expand_dims(xp.asarray(val, dtype=out.dtype), axis)
            out = out + coef.reshape(shape) * val
        return out

    def derivatives(self, f: CellField, axis: int):
        """``(f', f'')`` along ``axis`` (physical; chain rule on a non-uniform axis)."""
        g1, g2 = self.derivatives_xi(f, axis)
        return _to_physical(f.grid, axis, g1, g2)

    def derivatives_xi(self, f: CellField, axis: int):
        """``(f_xi, f_xixi)`` along ``axis`` from one CCD solve in the computational coordinate."""
        grid = f.grid
        lo, hi = self._kinds(f, axis)
        mats = ccd_matrices(grid.n[axis], float(grid.h[axis]), lo, hi)
        data = None if lo == PERIODIC else [(LOWER, _wall_datum(f, axis, LOWER)),
                                            (UPPER, _wall_datum(f, axis, UPPER))]
        u = f.interior
        return (self._apply(grid.xp, u, axis, mats, 0, data), self._apply(grid.xp, u, axis, mats, 1, data))

    def d1(self, a: CellField, axis: int):
        return self.derivatives(a, axis)[0]

    def d2(self, a: CellField, axis: int):
        return self.derivatives(a, axis)[1]

    def d1_array(self, values, grid, axis: int, periodic: bool):
        """CCD first derivative of an interior array without wall data (``extrapolate`` closure)."""
        return self.derivatives_array(values, grid, axis, periodic, which=(0,))[0]

    def d2_array(self, values, grid, axis: int, periodic: bool):
        """CCD second derivative of an interior array without wall data (``extrapolate`` closure)."""
        return self.derivatives_array(values, grid, axis, periodic, which=(1,))[0]

    def derivatives_array(self, values, grid, axis: int, periodic: bool, which=(0, 1)):
        """``(f', f'')`` (or the subset ``which``) of an interior array without wall data.

        Used where no boundary condition exists for the differentiated quantity:
        products (the conservative part of the skew convection form) and the
        repeated second derivatives of the UCCD6/FUCCD hyperviscosity.
        """
        g1, g2 = self.derivatives_array_xi(values, grid, axis, periodic)
        d = _to_physical(grid, axis, g1, g2)
        return tuple(d[w] for w in which)

    def derivatives_array_xi(self, values, grid, axis: int, periodic: bool):
        """``(f_xi, f_xixi)`` of an interior array without wall data."""
        kind = PERIODIC if periodic else EXTRAPOLATE
        mats = ccd_matrices(grid.n[axis], float(grid.h[axis]), kind, kind)
        return tuple(self._apply(grid.xp, values, axis, mats, w) for w in (0, 1))

    def d1_product(self, a: CellField, b: CellField, axis: int):
        """CCD first derivative of the product ``a b`` along ``axis``.

        A wall where both factors have Dirichlet conditions gets the Dirichlet
        closure with the product of the wall values (4th order at the wall
        cell); other walls use the extrapolating closure (3rd order there).
        """
        grid = a.grid
        values = a.interior * b.interior
        if a.bc.is_periodic(axis):
            return self.d1_array(values, grid, axis, True)
        kinds, data = [], []
        for side in (LOWER, UPPER):
            if isinstance(a.bc.get(axis, side), Dirichlet) and isinstance(b.bc.get(axis, side), Dirichlet):
                kinds.append(DIRICHLET)
                data.append((side, _wall_datum(a, axis, side) * _wall_datum(b, axis, side)))
            else:
                kinds.append(EXTRAPOLATE)
        mats = ccd_matrices(grid.n[axis], float(grid.h[axis]), kinds[0], kinds[1])
        return _to_physical(grid, axis, self._apply(grid.xp, values, axis, mats, 0, data), None)[0]

    def d11(self, a: CellField, axis0: int, axis1: int):
        if axis0 == axis1:
            return self.d2(a, axis0)
        first = self.d1(a, axis0)
        return self.d1_array(first, a.grid, axis1, a.bc.is_periodic(axis1))


# -- FCCD face operators ------------------------------------------------------------------
def _take(arr, axis, sl):
    idx = [slice(None)] * arr.ndim
    idx[axis] = sl
    return arr[tuple(idx)]


class FCCDFaceComplex:
    """Face operators of FCCD (FaceComplex protocol), 4th order."""

    name = "fccd"
    stencil_ghosts = 1

    def __init__(self, ccd: CCDCellDerivative | None = None) -> None:
        self.ccd = ccd or CCDCellDerivative()
        self._fd2 = FD2FaceComplex()
        c = FCCD_WALL
        self._grad_w = [float(v) for v in c["gradient"]["coeffs"]]
        self._grad_t = c["gradient"]["terms"]
        self._val_w = [float(v) for v in c["value"]["coeffs"]]
        self._val_t = c["value"]["terms"]
        self._f2c_w = [float(v) for v in c["face_to_cell_wall"]["coeffs"]]
        self._f2c = [float(v) for v in c["face_to_cell"]["coeffs"]]

    def _q(self, a: CellField, axis: int):
        return self.ccd.derivatives_xi(a, axis)[1]

    @staticmethod
    def _physical_gradient(grid, axis: int, g_xi):
        """Face gradient in ``xi`` times ``J`` at the faces (non-uniform axis)."""
        if grid.axis_uniform(axis):
            return g_xi
        m = grid.metric(axis)
        return m.along(m.J_face, g_xi.ndim) * g_xi

    def _faces(self, a: CellField, axis: int, interior_formula, wall):
        grid = a.grid
        xp = grid.xp
        n, h = grid.n[axis], float(grid.h[axis])
        u = a.interior
        if a.bc.is_periodic(axis):
            q = self._q(a, axis)
            um, qm = xp.roll(u, 1, axis=axis), xp.roll(q, 1, axis=axis)
            inner = interior_formula(um, u, qm, q, h)  # face k between k-1 and k, k = 0..n-1
            return xp.concatenate([inner, _take(inner, axis, slice(0, 1))], axis=axis)
        p, q = self.ccd.derivatives_xi(a, axis)
        inner = interior_formula(_take(u, axis, slice(0, n - 1)), _take(u, axis, slice(1, n)),
                                 _take(q, axis, slice(0, n - 1)), _take(q, axis, slice(1, n)), h)
        jet = (u, p, q)
        lo = xp.expand_dims(wall(a, axis, LOWER, jet, h), axis)
        hi = xp.expand_dims(wall(a, axis, UPPER, jet, h), axis)
        return xp.concatenate([lo, inner, hi], axis=axis)

    @staticmethod
    def _jet_sum(coeffs, terms, jet, axis, side, h):
        """``sum c h^d (s^d f^(d))_k`` over the wall cells ``k`` (``s = -1`` reflects the upper wall)."""
        s = 1.0 if side == LOWER else -1.0
        n = jet[0].shape[axis]
        return sum(c * (s * h) ** d * _take(jet[d], axis, k if side == LOWER else n - 1 - k)
                   for c, (d, k) in zip(coeffs, terms))

    @staticmethod
    def _broadcast(val, like):
        return val + 0.0 * like

    def grad_f(self, a: CellField, axis: int):
        def interior(ul, ur, ql, qr, h):
            return (ur - ul) / h - (h / 24.0) * (qr - ql)

        def wall(f, ax, side, jet, h):
            cond = f.bc.get(ax, side)
            datum = _wall_datum(f, ax, side)
            if isinstance(cond, Neumann):
                if not f.grid.axis_uniform(ax):
                    # the face metric applied below is FCCD's (4th order); divide it out so that the wall
                    # face carries the physical datum exactly (Neumann compatibility of the projection)
                    m = f.grid.metric(ax)
                    k = 0 if side == LOWER else 1
                    datum = datum / (m.wall_x_xi[k] * float(m.J_face[0 if side == LOWER else -1]))
                return self._broadcast(datum, _take(jet[0], ax, 0))
            c = self._grad_w
            s = 1.0 if side == LOWER else -1.0
            return s * (c[0] * datum + self._jet_sum(c[1:], self._grad_t, jet, ax, side, h)) / h

        return self._physical_gradient(a.grid, axis, self._faces(a, axis, interior, wall))

    def grad_f_array(self, values, grid, axis: int, periodic: bool):
        """FCCD face gradient of an interior array that has no boundary condition.

        Interior faces use the FCCD formula with ``q`` from the extrapolating CCD
        closure; on a wall axis the two wall faces are set to zero (there is no
        wall datum). Used for the FUCCD dissipation flux, which is multiplied by
        the wall-normal face speed and therefore vanishes on impermeable walls.
        """
        xp = grid.xp
        n, h = grid.n[axis], float(grid.h[axis])
        q = self.ccd.derivatives_array_xi(values, grid, axis, periodic)[1]

        def formula(ul, ur, ql, qr):
            return (ur - ul) / h - (h / 24.0) * (qr - ql)

        if periodic:
            inner = formula(xp.roll(values, 1, axis=axis), values, xp.roll(q, 1, axis=axis), q)
            out = xp.concatenate([inner, _take(inner, axis, slice(0, 1))], axis=axis)
        else:
            inner = formula(_take(values, axis, slice(0, n - 1)), _take(values, axis, slice(1, n)),
                            _take(q, axis, slice(0, n - 1)), _take(q, axis, slice(1, n)))
            zero = xp.zeros_like(_take(values, axis, slice(0, 1)))
            out = xp.concatenate([zero, inner, zero], axis=axis)
        return self._physical_gradient(grid, axis, out)

    def interp_f(self, a: CellField, axis: int):
        def interior(ul, ur, ql, qr, h):
            return 0.5 * (ul + ur) - (h * h / 16.0) * (ql + qr)

        def wall(f, ax, side, jet, h):
            cond = f.bc.get(ax, side)
            datum = _wall_datum(f, ax, side)
            if isinstance(cond, Dirichlet):
                return self._broadcast(datum, _take(jet[0], ax, 0))
            c = self._val_w
            s = 1.0 if side == LOWER else -1.0  # gradient in the reflected coordinate: s df/dx
            return c[0] * h * s * datum + self._jet_sum(c[1:], self._val_t, jet, ax, side, h)

        return self._faces(a, axis, interior, wall)

    def div_f(self, flux: FaceField):
        return self._fd2.div_f(flux)

    def face_to_cell(self, face_values, axis: int, periodic: bool | None = None):
        """4th-order face-to-centre interpolation.

        The closure differs on periodic and wall axes and a face array does not
        say which it is, so ``periodic`` is required here.
        """
        if periodic is None:
            raise ValueError("FCCD face_to_cell needs periodic=True/False for the axis")
        xp = array_module(face_values)
        n = face_values.shape[axis] - 1
        w = self._f2c
        if periodic:
            base = _take(face_values, axis, slice(0, n))
            return sum(wj * xp.roll(base, -j, axis=axis) for wj, j in zip(w, (-1, 0, 1, 2)))
        inner = sum(wj * _take(face_values, axis, slice(j, j + n - 2)) for wj, j in zip(w, range(4)))
        ww = self._f2c_w
        first = sum(wj * _take(face_values, axis, j) for wj, j in zip(ww, range(4)))
        last = sum(wj * _take(face_values, axis, n - j) for wj, j in zip(ww, range(4)))
        return xp.concatenate([xp.expand_dims(first, axis), inner, xp.expand_dims(last, axis)], axis=axis)


"""High-order face-to-cell reconstruction ``R_h`` (cell value from the face-normal values along one axis).

:class:`CompactFaceToCell` (``reconstruction="ccd"``, recommended) is CCD-like: it widens
no stencil but uses derivatives. The faces along an axis form a nodal line with the walls
on its end nodes; a CCD solve on that line (Chu & Fan interior rows, compact wall closures
derived in ``tools/derive_face_ccd.py``) gives the face jet ``(f, f', f'')`` and the cell
value is the quintic Hermite midpoint of its two faces,
``(f_L + f_R)/2 + 5h/32 (f'_L - f'_R) + h^2/64 (f''_L + f''_R)`` (6th order, error
``-h^6 f^(6)/46080``). The wall rows stay on nodes 0..2 (the interior stencil width):
``f''`` 3rd order and, coupled through the implicit ``h f''_0`` of row I, ``f'`` 4th order;
weighted by ``h^2`` and ``h`` both give ``R_h`` 5th order at walls (6th inside); widening row II to node 3 (``R_h`` 6th at walls) changed no
Navier--Stokes result (the floor is the 4th-order face operators) and was dropped.
Measured, N = 128: periodic 3.0e-13 against 6.8e-11 for the 6-point Lagrange stencil;
walls 2.5e-10 (5th order).

:class:`FaceToCell` (the ``p``-point Lagrange stencil) is kept for comparison.

The face-native steppers (``face_native=True``) take the cell velocity as ``R_h u_f``; its
error is the floor of their cell values (FCCD's ``face_to_cell`` is 4th order: N = 64
Taylor--Green, faces 4.5e-7, cells 2.2e-6). :class:`FaceToCell` is the ``p``-point
Lagrange interpolation at the cell centre (``p`` even, order ``p``): the symmetric
stencil in the interior (``p = 6``: ``(3, -25, 150, 150, -25, 3)/256``) and the ``p``
faces nearest to the cell next to a wall (the wall face included). It works in the
computational coordinate, so it applies unchanged to mapped grids.

The weights are exact rationals; the operator is applied as a dense matrix per axis
(as the CCD operators).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np

from ..backend import array_module


def lagrange_weights(nodes, x) -> list:
    """Exact weights of the Lagrange interpolant through ``nodes`` evaluated at ``x``."""
    out = []
    for j, xj in enumerate(nodes):
        w = Fraction(1)
        for k, xk in enumerate(nodes):
            if k != j:
                w *= Fraction(x - xk) / Fraction(xj - xk)
        out.append(w)
    return out


@lru_cache(maxsize=64)
def reconstruction_matrix(n: int, order: int, periodic: bool) -> np.ndarray:
    """Host matrix ``(n, n+1)``: cell values from the ``n + 1`` faces (face ``k`` at ``k``, cell ``i`` at ``i + 1/2``)."""
    if order < 2 or order % 2:
        raise ValueError("the reconstruction order must be even and >= 2")
    if not periodic and n + 1 < order:
        raise ValueError(f"order {order} needs at least {order - 1} cells on a wall axis, got {n}")
    M = np.zeros((n, n + 1))
    half = order // 2
    for i in range(n):
        x = Fraction(2 * i + 1, 2)
        if periodic:
            nodes = list(range(i + 1 - half, i + 1 + half))
            for k, w in zip(nodes, lagrange_weights(nodes, x)):
                M[i, k % n] += float(w)  # face n is face 0
        else:
            start = min(max(i + 1 - half, 0), n + 1 - order)
            nodes = list(range(start, start + order))
            for k, w in zip(nodes, lagrange_weights(nodes, x)):
                M[i, k] += float(w)
    return M


class FaceToCell:
    """``R_h`` of order ``order`` (callable like ``FaceComplex.face_to_cell``)."""

    def __init__(self, order: int = 6) -> None:
        self.order = int(order)
        self._dev: dict = {}

    def __call__(self, face_values, axis: int, periodic: bool | None = None):
        if periodic is None:
            raise ValueError("FaceToCell needs periodic=True/False for the axis")
        xp = array_module(face_values)
        n = face_values.shape[axis] - 1
        key = (xp.__name__, n, periodic, str(face_values.dtype))
        M = self._dev.get(key)
        if M is None:
            M = self._dev[key] = xp.asarray(reconstruction_matrix(n, self.order, bool(periodic)), dtype=face_values.dtype)
        moved = xp.moveaxis(face_values, axis, -1)
        return xp.moveaxis(moved @ M.T, -1, axis)


# -- compact (CCD) reconstruction -------------------------------------------------------------
# Chu & Fan interior rows (as discretization/ccd.py) and the face-line wall closures
# (tools/derive_face_ccd.py, h = 1, derivatives taken away from the wall):
_WALL_I = {"implicit": (Fraction(1, 7), Fraction(8, 7), Fraction(-2, 7)),          # f''_0, f'_1, f''_1
           "explicit": (Fraction(-31, 14), Fraction(16, 7), Fraction(-1, 14))}      # f_0, f_1, f_2
_WALL_II = {"implicit": (Fraction(5), Fraction(-6)),                                # f''_1, f'_1
            "explicit": (Fraction(9), Fraction(-12), Fraction(3))}                    # f_0, f_1, f_2
HERMITE = (Fraction(1, 2), Fraction(5, 32), Fraction(1, 64))


@lru_cache(maxsize=64)
def face_ccd_matrices(n: int, h: float, periodic: bool):
    """Host ``(D1, D2)``: face-line CCD derivatives, ``(m, m)`` with ``m = n`` (periodic) or ``n + 1``."""
    from .ccd import A1, A2, ALPHA1, B1, B2, BETA2  # noqa: PLC0415

    m = n if periodic else n + 1
    M = np.zeros((2 * m, 2 * m))
    R = np.zeros((2 * m, m))
    for k in range(m) if periodic else range(1, m - 1):
        km, kp = (k - 1) % m, (k + 1) % m
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
        for k0, s in ((0, 1), (m - 1, -1)):  # mirrored at the upper wall: f^(d) -> s^d f^(d)
            node = lambda q, k0=k0, s=s: k0 + s * q  # noqa: E731
            i, j = 2 * k0, 2 * k0 + 1
            b0, a1, b1 = (float(v) for v in _WALL_I["implicit"])
            for q, dd, c in ((0, 1, 1.0), (0, 2, b0), (1, 1, a1), (1, 2, b1)):
                M[i, 2 * node(q) + dd - 1] += c * s ** (dd + 1) * h ** (dd - 1)
            for q, c in enumerate(_WALL_I["explicit"]):
                R[i, node(q)] += float(c) * s / h
            a2, g1 = (float(v) for v in _WALL_II["implicit"])
            for q, dd, c in ((0, 2, 1.0), (1, 2, a2), (1, 1, g1)):
                M[j, 2 * node(q) + dd - 1] += c * s ** (dd + 2) * h ** (dd - 2)
            for q, c in enumerate(_WALL_II["explicit"]):
                R[j, node(q)] += float(c) / h ** 2
    T = np.linalg.solve(M, R)
    return T[0::2].copy(), T[1::2].copy()


@lru_cache(maxsize=64)
def compact_reconstruction_matrix(n: int, periodic: bool) -> np.ndarray:
    """Host ``(n, n+1)`` matrix of the compact ``R_h`` (unit spacing: ``R_h`` is scale free)."""
    D1, D2 = face_ccd_matrices(n, 1.0, periodic)
    m = D1.shape[0]
    w0, w1, w2 = (float(v) for v in HERMITE)
    out = np.zeros((n, n + 1))
    for i in range(n):
        lo, hi = i, (i + 1) % m
        row = w1 * (D1[lo] - D1[hi]) + w2 * (D2[lo] + D2[hi])
        row[lo] += w0
        row[hi] += w0
        out[i, :m] = row  # periodic: face n (= face 0) gets no weight
    return out


class CompactFaceToCell(FaceToCell):
    """CCD-type ``R_h``: face-line CCD jet and quintic Hermite midpoint (6th order, two-face stencil)."""

    def __init__(self) -> None:
        super().__init__(6)
        self.name = "ccd"

    def __call__(self, face_values, axis: int, periodic: bool | None = None):
        if periodic is None:
            raise ValueError("CompactFaceToCell needs periodic=True/False for the axis")
        xp = array_module(face_values)
        n = face_values.shape[axis] - 1
        key = (xp.__name__, n, periodic, str(face_values.dtype))
        M = self._dev.get(key)
        if M is None:
            M = self._dev[key] = xp.asarray(compact_reconstruction_matrix(n, bool(periodic)), dtype=face_values.dtype)
        moved = xp.moveaxis(face_values, axis, -1)
        return xp.moveaxis(moved @ M.T, -1, axis)

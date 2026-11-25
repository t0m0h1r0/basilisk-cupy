"""Surface tension as the adjoint of the CLS transport (docs/notes/43).

The plain CLS transport is ``psi_t = -D_f(psi_f uf)`` (``psi_f`` the face interpolation of ``cls.ops``), linear
in ``uf``: ``psi_t = -T(psi) uf``. With the discrete area ``A = sum_c |G psi|_c V_c`` (``G`` the CCD first
derivative along each axis, the closure of psi's conditions) and ``mu = sigma (dA/dpsi) / V``, the face force

    F_f = -psi_f (mu_R - mu_L) / Delta        (zero on wall faces)

satisfies ``sum_f uf_f F_f V_f = -dE/dt`` for every ``uf`` (summation by parts of ``D_f``): the work of the force
on the face velocity is exactly the change of the discrete surface energy that the same ``uf`` causes.
:func:`attach` puts ``alpha_f F_f`` into CenteredNS's face acceleration (no GFM jump, no potentials).
Uniform 2-D grids, plain divergence only.
"""

from __future__ import annotations

import numpy as np

from basilisk_cupy.discretization.ccd import PERIODIC, ccd_matrices, _kind
from basilisk_cupy.boundary import LOWER

EPS_NORM = 1e-12


def _d1(grid, bc, ax):
    lo = _kind(bc.get(ax, LOWER))
    hi = _kind(bc.get(ax, 1))
    if lo != PERIODIC:  # psi has no wall datum here: extrapolating closure, the same matrix for G and G^T
        lo = hi = "extrapolate"
    return ccd_matrices(grid.n[ax], float(grid.h[ax]), lo, hi)[0]


def _along(M, a, ax):
    return np.moveaxis(np.tensordot(M, np.moveaxis(a, ax, 0), axes=(1, 0)), 0, ax)


def area_and_gradient(psi_field):
    """``(A, dA/dpsi)`` of the discrete area ``sum |G psi| V``."""
    grid = psi_field.grid
    q = psi_field.interior
    V = float(grid.h[0] * grid.h[1])
    D = [_d1(grid, psi_field.bc, ax) for ax in range(2)]
    g = [_along(D[ax], q, ax) for ax in range(2)]
    norm = np.sqrt(g[0] ** 2 + g[1] ** 2 + EPS_NORM ** 2)
    A = float(np.sum(norm - EPS_NORM)) * V
    dA = sum(_along(D[ax].T, g[ax] / norm, ax) for ax in range(2)) * V
    return A, dA


def face_force(cls, sigma):
    """``F_f = -psi_f G_f mu`` on the faces of each axis (zero on wall faces) and ``mu``."""
    grid = cls.psi.grid
    psi = cls.psi
    psi.fill_ghosts()
    _, dA = area_and_gradient(psi)
    V = float(grid.h[0] * grid.h[1])
    mu = sigma * dA / V
    out = []
    for ax in range(2):
        psi_f = np.asarray(cls.ops.interp_f(psi, ax))
        n = grid.n[ax]
        pad = [(0, 0), (0, 0)]
        pad[ax] = (1, 1)
        m = np.pad(mu, pad, mode="edge")
        sl_r = [slice(None)] * 2
        sl_l = [slice(None)] * 2
        sl_r[ax], sl_l[ax] = slice(1, n + 2), slice(0, n + 1)
        F = -psi_f * (m[tuple(sl_r)] - m[tuple(sl_l)]) / float(grid.h[ax])
        if not psi.bc.is_periodic(ax):
            w = [slice(None)] * 2
            w[ax] = [0, n]
            F[tuple(w)] = 0.0
        out.append(F)
    return out, mu


def attach(ns, tp, sigma=1.0):
    """Use the adjoint face force instead of the GFM jump / the potentials."""
    tp.potentials.clear()
    tp.pressure_jump = False

    def hook(_ns):
        F, _ = face_force(tp.cls, sigma)
        _ns.a_is_constant = False
        for ax in range(2):
            _ns.a[ax] = _ns.a[ax] + _ns.alpha[ax] * F[ax]

    ns.on("acceleration", hook)

    def dtmax(_ns):  # keep tension.h's explicit capillary bound
        tp._capillary_dtmax(_ns)

    ns.on("stability", dtmax)

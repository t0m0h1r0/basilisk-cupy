"""Energy consistency of the corrected CLS kinematics with the GFM force (docs/notes/45).

For a fixed ``psi`` the transport is linear in the face velocity, ``psi_t = -T(psi) uf`` (no clipping, no mass
closure): ``T`` is assembled column by column (wall faces excluded). With the discrete surface energy
``E = sigma A_h(psi)`` (:func:`cases.adjoint_tension.area_and_gradient`) and ``g = dE/dpsi``,

    dE/dt = -g^T T uf,       F_adj = T^T g / V_f   =>   sum_f uf_f F_adj,f V_f + dE/dt = 0

for every ``uf``. ``F_GFM`` is the sharp GFM face acceleration (unit density). ``Delta F = F_adj - F_GFM`` is
split by the projection into a gradient part (taken by the pressure) and a divergence-free part (drives flow).
"""

from __future__ import annotations

import numpy as np

from basilisk_cupy.fields import FaceField
from cases.adjoint_tension import area_and_gradient


def free_faces(grid, periodic=(False, False)):
    """Index list ``(axis, i, j)`` of the faces that carry velocity (wall faces excluded)."""
    out = []
    for a in range(2):
        shape = grid.face_shape(a)
        for i in range(shape[0]):
            for j in range(shape[1]):
                k = (i, j)[a]
                if not periodic[a] and k in (0, grid.n[a]):
                    continue
                if periodic[a] and k == grid.n[a]:
                    continue
                out.append((a, i, j))
    return out


def to_face_field(grid, vec, faces, periodic=(False, False)):
    arrs = [np.zeros(grid.face_shape(a)) for a in range(2)]
    for v, (a, i, j) in zip(vec, faces):
        arrs[a][i, j] = v
    for a in range(2):
        if periodic[a]:
            sl_last = [slice(None)] * 2
            sl_first = [slice(None)] * 2
            sl_last[a], sl_first[a] = grid.n[a], 0
            arrs[a][tuple(sl_last)] = arrs[a][tuple(sl_first)]
    return FaceField(grid, arrs, "uf")


def from_face_field(F, faces):
    return np.array([F[a][i, j] for (a, i, j) in faces])


def transport_matrix(cls, kinematics: str = "transport"):
    """``T`` with ``psi_t = -T uf`` for the frozen ``psi`` of ``cls``: the library transport (``"transport"``)
    or the band correction of :mod:`cases.velocity_extension` (``"extension"``)."""
    from cases.velocity_extension import advective_rhs, corrected_face_velocity

    grid = cls.psi.grid
    per = tuple(cls.psi.bc.is_periodic(a) for a in range(2))
    faces = free_faces(grid, per)
    q = cls.psi.interior.copy()
    cols = []
    for k in range(len(faces)):
        e = np.zeros(len(faces))
        e[k] = 1.0
        uf = to_face_field(grid, e, faces, per)
        r = cls._rhs(q, uf)
        if kinematics == "extension":
            ufc = corrected_face_velocity(cls, uf)
            delta = FaceField(grid, [ufc[a] - uf[a] for a in range(2)], "delta")
            r = r + advective_rhs(cls, q, delta)
        cols.append(-np.asarray(r).ravel())
    return np.array(cols).T, faces


def gfm_face_force(tp, faces):
    """The sharp GFM face acceleration ``j (H_R - H_L)/Delta`` at unit density (``j`` from the potentials)."""
    from basilisk_cupy.interface.cls import centre_spacing

    xp = np
    phi = None
    for pot in tp.potentials:
        v = pot(tp)
        phi = v if phi is None else phi + v
    arrs = []
    for ax in range(2):
        h = centre_spacing(tp.f.grid, ax)
        hl, hr, cut, theta = tp._cut(ax)
        jl, jr = tp._faces(phi, ax)
        j = (1.0 - theta) * jl + theta * jr
        sign = hr.astype(float) - hl.astype(float)
        arrs.append(xp.where(cut, j * sign / h, 0.0))
    return from_face_field(arrs, faces)


def adjoint_force(cls, T, sigma=1.0):
    """``F_adj = T^T g / V_f`` and ``g`` (cell gradient of ``sigma A_h``)."""
    grid = cls.psi.grid
    _, dA = area_and_gradient(cls.psi)
    g = sigma * np.asarray(dA).ravel()
    Vf = float(grid.h[0] * grid.h[1])
    return T.T @ g / Vf, g


def split(F, grid, faces, periodic=(False, False)):
    """``F = G phi + F_sol``: the projection (fd2, Neumann walls) of the face vector ``F``."""
    from basilisk_cupy import CellField
    from basilisk_cupy.boundary import BoundaryConditions, Neumann, Periodic
    from basilisk_cupy.fields import FaceField as FF
    from basilisk_cupy.projection.approximate import project

    bc = BoundaryConditions.periodic(2) if all(periodic) else BoundaryConditions()
    p = CellField(grid, bc=bc, name="phi")
    uf = to_face_field(grid, F, faces, periodic)
    alpha = FF(grid, [np.ones(grid.face_shape(a)) for a in range(2)], "alpha")
    project(uf, p, alpha, 1.0, tolerance=1e-13)
    sol = from_face_field(uf, faces)
    return F - sol, sol


def crossing_matrix(cls):
    """``C`` with ``d eta_i = sum_c C_ic d psi_c``: the derivative of the psi = 1/2 crossing height of each
    column (graph interface in y, one crossing per column) with respect to psi."""
    grid = cls.psi.grid
    q = cls.psi.interior
    nx, ny = grid.n
    h = float(grid.h[1])
    C = np.zeros((nx, nx * ny))
    for i in range(nx):
        col = q[i]
        k = int(np.nonzero((col[:-1] - 0.5) * (col[1:] - 0.5) <= 0)[0][0])
        pl, ph = col[k], col[k + 1]
        dp = ph - pl
        # y* = y_k + h (1/2 - pl) / dp
        # d y*/d pl = h (-(dp) + (1/2 - pl)) / dp^2 ;  d y*/d ph = -h (1/2 - pl) / dp^2
        C[i, i * ny + k] = h * (-dp + (0.5 - pl)) / dp ** 2
        C[i, i * ny + k + 1] = -h * (0.5 - pl) / dp ** 2
    return C


def sharp_energy_force(cls, T, sigma=1.0):
    """``F = (C T)^T g / V_f`` with the sharp-interface energy ``E = sigma/2 sum (eta_x)^2 dx`` of the crossing
    heights, ``eta_x`` from the cosine series on the walls (the exact linear curvature of the GFM test,
    ``g = sigma dx K eta``, ``K`` the symmetrised ``-d2/dx2``)."""
    grid = cls.psi.grid
    nx = grid.n[0]
    L = float(grid.length[0])
    xc = np.asarray(grid.centers(0))
    m = np.arange(nx)
    Cb = np.cos(np.pi * m[None, :] * xc[:, None] / L)
    Kmat = Cb @ np.diag((np.pi * m / L) ** 2) @ np.linalg.inv(Cb)
    Kmat = 0.5 * (Kmat + Kmat.T)
    C = crossing_matrix(cls)
    eta = np.array([0.0] * nx)
    from cases.capwave_jacobian import crossing_heights

    class _TP:  # crossing_heights reads tp.f.grid and tp.cls.psi
        pass

    tp = _TP()
    tp.f = cls.psi
    tp.cls = cls
    eta = crossing_heights(tp)
    dx = float(grid.h[0])
    g = sigma * dx * (Kmat @ eta)
    Vf = float(grid.h[0] * grid.h[1])
    return (C @ T).T @ g / Vf, g, Kmat

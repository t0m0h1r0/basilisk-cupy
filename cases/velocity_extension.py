"""Velocity extension for the CLS transport (docs/notes/44).

The conservative CLS transport ``psi_t = -D_f(psi_f uf)`` moves the ``psi = 1/2`` crossing with a weighted mean of
the face velocities within about one cell of the interface (docs/notes/41: weights 0.57 at the interface face,
0.23 at +-h). Where the velocity varies on a scale below the cell (the vortex layer of a capillary wave,
delta < h) the interface then moves with a smeared velocity, and the wave is under-damped by ~ (h/delta)^2.

Here the interface is moved by the velocity *at* the interface, extended along the normal:

1. the crossings: on every face where ``H = (psi > 1/2)`` changes, the point ``x_lo + theta (x_hi - x_lo)``,
   ``theta = (psi_lo - 1/2) / (psi_lo - psi_hi)``;
2. the velocity there: the component normal to the cut face is interpolated linearly between the two faces that
   bracket the point along that axis; the other component between the cell-centred averages of the two cells;
   the normal speed ``V_n = v . n`` (``n`` the normalised psi gradient, averaged over the two cells);
3. each cell takes ``V_n`` of its nearest crossing (minimum image on periodic axes): ``w = V_n n``; face values
   are the mean of the two cells' normal components, zero on walls;
4. ``psi_t = -D_f(psi_f w_f) + psi D_f(w_f)`` (= ``-w . grad psi``: ``w`` is not divergence-free), TVD-RK3 with
   ``w`` frozen over the step; the stage-B mass closure restores the mass.

Uniform 2-D grids. With the sharp GFM jump this pairs a sharp kinematic operator with its adjoint
(docs/notes/43). A prototype in ``cases/``: :func:`attach` replaces ``cls.advect`` of one interface; the
library's CLS transport is unchanged.
"""

from __future__ import annotations

import numpy as np

from basilisk_cupy.fields import CellField, FaceField


def _pad(xp, a, bc, ax):
    """One ghost layer along ``ax``: periodic wrap, else edge copy."""
    pad = [(0, 0)] * a.ndim
    pad[ax] = (1, 1)
    return xp.pad(a, pad, mode="wrap" if bc.is_periodic(ax) else "edge")


def crossings(cls):
    """``(points (k, 2), normal speeds placeholder index data)`` for the cut faces of both axes.

    Returns a list of tuples ``(ax, i_lo, j_lo, theta)`` as arrays: for axis 0 the cut lies between cells
    ``(i, j)`` and ``(i + 1, j)``; for axis 1 between ``(i, j)`` and ``(i, j + 1)``."""
    q = cls.psi.interior
    xp = cls.psi.grid.xp
    out = []
    for ax in range(2):
        n = q.shape[ax]
        per = cls.psi.bc.is_periodic(ax)
        lo = q
        hi = xp.roll(q, -1, ax)
        cut = (lo > 0.5) != (hi > 0.5)
        if not per:
            sl = [slice(None)] * 2
            sl[ax] = n - 1
            cut[tuple(sl)] = False
        i, j = xp.nonzero(cut)
        a, b = lo[i, j], hi[i, j]
        theta = (a - 0.5) / (a - b)
        out.append((ax, i, j, theta))
    return out


def extended_face_velocity(cls, uf: FaceField):
    """The extension ``w_f`` (FaceField) of the normal interface speed, and the crossing data."""
    grid = cls.psi.grid
    xp = grid.xp
    h = [float(v) for v in grid.h]
    o = [float(v) for v in grid.origin]
    L = [float(v) for v in grid.length]
    per = [cls.psi.bc.is_periodic(ax) for ax in range(2)]
    psi = cls.psi
    psi.fill_ghosts()
    gx = cls.ccd.d1(psi, 0)
    gy = cls.ccd.d1(psi, 1)
    gn = xp.sqrt(gx * gx + gy * gy) + 1e-30
    nx, ny = gx / gn, gy / gn
    # cell-centred averages of the face velocities (for the component along the cut face)
    uc = [0.5 * (uf[0][1:, :] + uf[0][:-1, :]), 0.5 * (uf[1][:, 1:] + uf[1][:, :-1])]
    pts, speeds = [], []
    for ax, i, j, th in crossings(cls):
        if i.size == 0:
            continue
        b = 1 - ax
        n_ax = grid.n[ax]
        # index of the "hi" cell along ax
        ih = [i, j]
        ih[ax] = (ih[ax] + 1) % n_ax
        # component along ax: the face between lo and hi is at index lo+1 in the face array of ax;
        # the point lies at fraction (theta - 1/2) from that face towards hi (theta > 1/2) or lo (theta < 1/2)
        f_mid = [i, j]
        f_mid[ax] = f_mid[ax] + 1
        f_mid = tuple(f_mid)
        f_next = [i, j]
        f_next[ax] = xp.where(th >= 0.5, f_next[ax] + 2, f_next[ax])
        if per[ax]:
            f_next[ax] = f_next[ax] % n_ax
        else:
            f_next[ax] = xp.clip(f_next[ax], 0, n_ax)
        f_next = tuple(f_next)
        s = xp.abs(th - 0.5)
        v_ax = (1.0 - s) * uf[ax][f_mid] + s * uf[ax][f_next]
        v_b = (1.0 - th) * uc[b][i, j] + th * uc[b][tuple(ih)]
        nax = 0.5 * ((nx, ny)[ax][i, j] + (nx, ny)[ax][tuple(ih)])
        nb = 0.5 * ((nx, ny)[b][i, j] + (nx, ny)[b][tuple(ih)])
        nn = xp.sqrt(nax * nax + nb * nb) + 1e-30
        V = (v_ax * nax + v_b * nb) / nn
        c = [None, None]
        c[ax] = o[ax] + (i if ax == 0 else j) * h[ax] + (0.5 + th) * h[ax]
        c[b] = o[b] + ((j if ax == 0 else i) + 0.5) * h[b]
        pts.append(xp.stack(c, axis=1))
        speeds.append(V)
    if not pts:
        return FaceField(grid, [xp.zeros(grid.face_shape(a)) for a in range(2)], "w"), None
    P = xp.concatenate(pts)
    S = xp.concatenate(speeds)
    x, y = grid.meshgrid()
    X = xp.stack([xp.broadcast_to(x, grid.n).ravel(), xp.broadcast_to(y, grid.n).ravel()], axis=1)
    Vc = xp.empty(X.shape[0])
    for s0 in range(0, X.shape[0], 4096):
        d = X[s0:s0 + 4096, None, :] - P[None, :, :]
        for a in range(2):
            if per[a]:
                d[..., a] = d[..., a] - L[a] * xp.round(d[..., a] / L[a])
        Vc[s0:s0 + 4096] = S[xp.argmin((d * d).sum(axis=2), axis=1)]
    Vc = Vc.reshape(grid.n)
    wc = [Vc * nx, Vc * ny]
    faces = []
    for a in range(2):
        p = _pad(xp, wc[a], cls.psi.bc, a)
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[a], hi[a] = slice(0, grid.n[a] + 1), slice(1, grid.n[a] + 2)
        f = 0.5 * (p[tuple(lo)] + p[tuple(hi)])
        if not per[a]:
            w = [slice(None)] * 2
            w[a] = [0, grid.n[a]]
            f[tuple(w)] = 0.0
        faces.append(f)
    return FaceField(grid, faces, "w"), (P, S)


def _interp_faces(grid, uf, ax, X, Y):
    """Bilinear interpolation of the face component ``uf[ax]`` at points ``(X, Y)`` (clamped to the domain)."""
    xp = grid.xp
    h = [float(v) for v in grid.h]
    o = [float(v) for v in grid.origin]
    a = uf[ax]
    # positions of the samples of uf[ax]: faces along ax, centres along the other axis
    sx = (X - o[0]) / h[0] - (0.0 if ax == 0 else 0.5)
    sy = (Y - o[1]) / h[1] - (0.5 if ax == 0 else 0.0)
    nxs, nys = a.shape
    sx = xp.clip(sx, 0.0, nxs - 1.000001)
    sy = xp.clip(sy, 0.0, nys - 1.000001)
    i0, j0 = xp.floor(sx).astype(int), xp.floor(sy).astype(int)
    tx, ty = sx - i0, sy - j0
    return ((1 - tx) * (1 - ty) * a[i0, j0] + tx * (1 - ty) * a[i0 + 1, j0]
            + (1 - tx) * ty * a[i0, j0 + 1] + tx * ty * a[i0 + 1, j0 + 1])


def projected_cell_velocity(cls, uf: FaceField):
    """``w = V n`` at the cell centres with ``V = n . u(x - phi n)`` (see :func:`projected_face_velocity`)."""
    from basilisk_cupy.interface.cls import phi_from_psi

    grid = cls.psi.grid
    xp = grid.xp
    psi = cls.psi
    psi.fill_ghosts()
    gx, gy = cls.ccd.d1(psi, 0), cls.ccd.d1(psi, 1)
    gn = xp.sqrt(gx * gx + gy * gy) + 1e-30
    nx, ny = gx / gn, gy / gn
    phi = phi_from_psi(psi.interior, cls.eps)
    x, y = grid.meshgrid()
    X = x - phi * nx
    Y = y - phi * ny
    V = nx * _interp_faces(grid, uf, 0, X, Y) + ny * _interp_faces(grid, uf, 1, X, Y)
    return V * nx, V * ny


def gradient_rhs(cls, q, wc):
    """``-w . grad psi`` with the CCD gradient (no divergence of w: w = V n is singular where n is)."""
    f = CellField.from_interior(cls.psi.grid, q, cls.psi.bc)
    f.fill_ghosts()
    return -(wc[0] * cls.ccd.d1(f, 0) + wc[1] * cls.ccd.d1(f, 1))


def corrected_face_velocity(cls, uf: FaceField, band: float = 4.0):
    """``uf* = uf + delta``, built on the faces: on a face normal to ``a``,
    ``delta = chi (V - u . n) n_a`` with ``n``, ``phi`` averaged from the two cells, ``V = n . u`` interpolated at
    the foot point ``x_f - phi n`` on the interface, ``u . n`` from the face value and the other component
    averaged from the four neighbouring faces; ``chi = exp(-(phi / (band eps))^4)``. A uniform velocity gives
    ``delta = 0``."""
    from basilisk_cupy.interface.cls import phi_from_psi

    grid = cls.psi.grid
    xp = grid.xp
    psi = cls.psi
    psi.fill_ghosts()
    gx, gy = cls.ccd.d1(psi, 0), cls.ccd.d1(psi, 1)
    gn = xp.sqrt(gx * gx + gy * gy) + 1e-30
    ncell = [gx / gn, gy / gn]
    phic = phi_from_psi(psi.interior, cls.eps)
    per = [psi.bc.is_periodic(ax) for ax in range(2)]
    faces = []
    for a in range(2):
        b = 1 - a
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[a], hi[a] = slice(0, grid.n[a] + 1), slice(1, grid.n[a] + 2)

        def to_face(c):
            p = _pad(xp, c, psi.bc, a)
            return 0.5 * (p[tuple(lo)] + p[tuple(hi)])

        nf = [to_face(ncell[0]), to_face(ncell[1])]
        nn = xp.sqrt(nf[0] ** 2 + nf[1] ** 2) + 1e-30
        nf = [nf[0] / nn, nf[1] / nn]
        phif = to_face(phic)
        X, Y = (grid.face_meshgrid(a)[k] * xp.ones(grid.face_shape(a)) for k in range(2))
        Xf, Yf = X - phif * nf[0], Y - phif * nf[1]
        V = nf[0] * _interp_faces(grid, uf, 0, Xf, Yf) + nf[1] * _interp_faces(grid, uf, 1, Xf, Yf)
        ub = _interp_faces(grid, uf, b, X, Y)  # the other component at this face
        un = uf[a] * nf[a] + ub * nf[b]
        chi = xp.exp(-(phif / (band * cls.eps)) ** 4)
        f = uf[a] + chi * (V - un) * nf[a]
        if not per[a]:
            w = [slice(None)] * 2
            w[a] = [0, grid.n[a]]
            f[tuple(w)] = uf[a][tuple(w)]
        faces.append(f)
    return FaceField(grid, faces, "uf*")


def projected_face_velocity(cls, uf: FaceField):
    """Smooth extension: every cell takes ``V = n . u(x - phi n)``, the normal velocity at its foot point on
    the interface (``phi`` the logit distance, ``n`` the normalised CCD gradient of psi); ``w = V n``."""
    from basilisk_cupy.interface.cls import phi_from_psi

    grid = cls.psi.grid
    xp = grid.xp
    psi = cls.psi
    psi.fill_ghosts()
    gx, gy = cls.ccd.d1(psi, 0), cls.ccd.d1(psi, 1)
    gn = xp.sqrt(gx * gx + gy * gy) + 1e-30
    nx, ny = gx / gn, gy / gn
    phi = phi_from_psi(psi.interior, cls.eps)
    x, y = grid.meshgrid()
    X = x - phi * nx
    Y = y - phi * ny
    V = nx * _interp_faces(grid, uf, 0, X, Y) + ny * _interp_faces(grid, uf, 1, X, Y)
    per = [psi.bc.is_periodic(ax) for ax in range(2)]
    wc = [V * nx, V * ny]
    faces = []
    for a in range(2):
        p = _pad(xp, wc[a], psi.bc, a)
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[a], hi[a] = slice(0, grid.n[a] + 1), slice(1, grid.n[a] + 2)
        f = 0.5 * (p[tuple(lo)] + p[tuple(hi)])
        if not per[a]:
            w = [slice(None)] * 2
            w[a] = [0, grid.n[a]]
            f[tuple(w)] = 0.0
        faces.append(f)
    return FaceField(grid, faces, "w")


def advective_rhs(cls, q, w: FaceField):
    """``-D_f(psi_f w_f) + psi D_f(w_f)``."""
    grid = cls.psi.grid
    f = CellField.from_interior(grid, q, cls.psi.bc)
    f.fill_ghosts()
    out = None
    for ax in range(2):
        psi_f = cls.ops.interp_f(f, ax)
        n = grid.n[ax]
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
        flux = psi_f * w[ax]
        d = (flux[tuple(hi)] - flux[tuple(lo)] - q * (w[ax][tuple(hi)] - w[ax][tuple(lo)])) / cls._widths[ax]
        out = -d if out is None else out - d
    return out


def attach(cls, method: str = "correction"):
    """Move this CLS interface with the extended normal speed instead of the conservative transport:
    ``method="correction"`` (the default: the conservative transport by ``uf`` plus the band correction
    ``delta = chi (V - u . n) n`` of :func:`corrected_face_velocity` advected without compression,
    ``-D_f(psi_f delta) + psi D_f delta``), ``"correction-conservative"`` (``D_f(psi_f (uf + delta))``: the
    compression by ``div delta`` moves the crossing too), ``"gradient"`` (foot-point speed,
    ``psi_t = -w . grad psi`` with the CCD gradient),
    ``"projection"`` (foot-point speed, face form ``-D_f(psi_f w) + psi D_f w``) or ``"nearest"``
    (nearest-crossing speed, face form)."""
    xp = cls.psi.grid.xp
    from basilisk_cupy.interface.cls import mass_correction

    def advect(uf, dt):
        q0 = cls.psi.interior
        m0 = xp.sum(q0 * cls.dv)
        if method == "correction":  # conservative transport by uf, the band correction advected without compression
            ufc = corrected_face_velocity(cls, uf)
            delta = FaceField(cls.psi.grid, [ufc[a] - uf[a] for a in range(2)], "delta")
            rhs = lambda q: cls._rhs(q, uf) + advective_rhs(cls, q, delta)  # noqa: E731
        elif method == "correction-conservative":
            ufc = corrected_face_velocity(cls, uf)
            rhs = lambda q: cls._rhs(q, ufc)  # noqa: E731
        elif method == "gradient":
            wc = projected_cell_velocity(cls, uf)
            rhs = lambda q: gradient_rhs(cls, q, wc)  # noqa: E731
        else:
            w = projected_face_velocity(cls, uf) if method == "projection" else extended_face_velocity(cls, uf)[0]
            rhs = lambda q: advective_rhs(cls, q, w)  # noqa: E731
        q1 = xp.clip(q0 + dt * rhs(q0), 0.0, 1.0)
        q2 = xp.clip(0.75 * q0 + 0.25 * (q1 + dt * rhs(q1)), 0.0, 1.0)
        q3 = xp.clip(q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs(q2)), 0.0, 1.0)
        if cls.mass_closure:
            q3 = mass_correction(q3, cls.dv, m0)
        cls.psi.interior = q3
        cls.psi.fill_ghosts()

    cls.advect = advect
    return cls

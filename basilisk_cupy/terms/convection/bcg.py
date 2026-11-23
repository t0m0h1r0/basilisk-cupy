"""Bell--Colella--Glaz advection (port of Basilisk ``bcg.h`` and ``prediction()``).

Second-order, unsplit, upwind. A face value is built from the upwind cell
value, a space--time extrapolation with the cell gradient, half of the source
term, and a transverse upwind correction; the cell update is the difference
of the face fluxes.

All face arrays follow :class:`~basilisk_cupy.fields.FaceField` (``n+1``
faces along the normal axis, face ``k`` between cells ``k-1`` and ``k``).
Cell quantities are evaluated on the cells ``-1 .. n`` along the normal axis
(one ghost on each side), which is what Basilisk's 1-ghost stencils read.
"""

from __future__ import annotations

from typing import Callable

from ...boundary import BoundaryConditions, Neumann, Periodic, gradient_bcs
from ...fields import CellField, FaceField, VectorField

Limiter = Callable[[object, object, object, object], object]


# -- limiters (Basilisk utils.h; arguments s0 = a[-1], s1 = a[0], s2 = a[1]) -------------
def _generic_limiter(xp, r, beta):
    v1 = xp.maximum(0.0, xp.minimum(r, beta))
    v2 = xp.minimum(beta * r, 1.0)
    return xp.maximum(v1, v2)


def _limited(beta: float):
    def limiter(xp, s0, s1, s2):
        d = s1 - s0
        safe = xp.where(d == 0, 1.0, d)
        return xp.where(d == 0, 0.0, _generic_limiter(xp, (s2 - s1) / safe, beta) * d)
    return limiter


minmod = _limited(1.0)
superbee = _limited(2.0)
sweby = _limited(1.5)


def gradients(f: CellField, limiter: Limiter | None = None) -> VectorField:
    """Cell-centred gradient of ``f`` (Basilisk ``gradients()``), ghosts filled.

    Without a limiter Basilisk uses centred differences, which is the
    default for the velocity in ``centered.h``.
    """
    grid = f.grid
    xp = grid.xp
    g = VectorField.zeros(grid, gradient_bcs(f.bc, grid.ndim), name=f"grad({f.name})")
    for ax in range(grid.ndim):
        s0, s1, s2 = f.data[grid.at({ax: -1})], f.data[grid.at()], f.data[grid.at({ax: 1})]
        if limiter is None:
            g[ax].interior = (s2 - s0) / (2.0 * grid.h[ax])
        else:
            g[ax].interior = limiter(xp, s0, s1, s2) / grid.h[ax]
    return g.fill_ghosts()


def _cells_along(data, grid, axis: int, extra: dict[int, int] | None = None):
    """View of cells ``-1 .. n`` along ``axis`` (interior elsewhere, shifted by ``extra``)."""
    g = grid.ghosts
    sl = list(grid.at(extra))
    sl[axis] = slice(g - 1, g + grid.n[axis] + 1)
    return data[tuple(sl)]


def _pad_face_along(face, grid, face_axis: int, axis: int, periodic: bool):
    """Extend a face array (normal to ``face_axis``) by one cell on each side of ``axis``.

    Basilisk keeps ghost faces; they are only read for inflow/periodic faces.
    Periodic axes wrap; other axes copy the adjacent face (symmetry).
    """
    xp = grid.xp
    mode = "wrap" if periodic else "edge"
    pad = [(0, 0)] * face.ndim
    pad[axis] = (1, 1)
    return xp.pad(face, pad, mode=mode)


def _upwind_select(xp, un, left, right):
    """Pick the upwind candidate: Basilisk ``s = sign(un)`` with sign(0) = -1."""
    return xp.where(un > 0, left, right)


def _bcg_faces(f: CellField, axis: int, un, grad_axis: CellField, transverse: list, src: CellField | None,
               dt: float):
    """Face values of ``f`` normal to ``axis`` (common core of tracer_fluxes and prediction).

    ``un`` is the CFL number ``dt*u_n/Delta`` on the faces; ``transverse`` lists,
    for every other axis ``t``, a cell-centred transverse velocity on cells
    ``-1 .. n`` along ``axis`` (shape as returned by :func:`_cells_along`).
    """
    grid = f.grid
    xp = grid.xp
    h = grid.h[axis]
    n = grid.n[axis]

    def lo(arr):  # cell k-1 for face k
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, n + 1)
        return arr[tuple(sl)]

    def hi(arr):  # cell k for face k
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(1, n + 2)
        return arr[tuple(sl)]

    fc = _cells_along(f.data, grid, axis)
    gc = _cells_along(grad_axis.data, grid, axis)
    # candidate from the left cell (s = +1) and from the right cell (s = -1)
    left = lo(fc) + (1.0 - un) * lo(gc) * h / 2.0
    right = hi(fc) - (1.0 + un) * hi(gc) * h / 2.0
    for t, vt in transverse:
        ht = grid.h[t]
        fp = _cells_along(f.data, grid, axis, {t: 1})
        fm = _cells_along(f.data, grid, axis, {t: -1})
        fyy = xp.where(vt < 0, fp - fc, fc - fm)
        corr = dt * vt * fyy / (2.0 * ht)
        left = left - lo(corr)
        right = right - hi(corr)
    face = _upwind_select(xp, un, left, right)
    if src is not None:
        sc = _cells_along(src.data, grid, axis)
        face = face + (lo(sc) + hi(sc)) * dt / 4.0
    return face


def tracer_fluxes(f: CellField, uf: FaceField, dt: float, src: CellField | None = None,
                  grad: VectorField | None = None, limiter: Limiter | None = None) -> FaceField:
    """Advective fluxes ``f_face * uf`` (Basilisk ``tracer_fluxes()``, fm = 1)."""
    grid = f.grid
    if not grid.is_uniform:
        raise NotImplementedError("BCG on a non-uniform grid is not implemented yet")
    f.fill_ghosts()
    grad = grad or gradients(f, limiter)
    flux = FaceField(grid, name=f"flux({f.name})")
    for ax in range(grid.ndim):
        un = dt * uf[ax] / grid.h[ax]
        transverse = []
        for t in range(grid.ndim):
            if t == ax:
                continue
            ft = _pad_face_along(uf[t], grid, t, ax, f.bc.is_periodic(ax))
            # cells -1..n along ax, average of the two faces normal to t
            n_t = grid.n[t]
            sl_lo = [slice(None)] * ft.ndim
            sl_hi = [slice(None)] * ft.ndim
            sl_lo[t] = slice(0, n_t)
            sl_hi[t] = slice(1, n_t + 1)
            vn = (ft[tuple(sl_lo)] + ft[tuple(sl_hi)]) / 2.0
            transverse.append((t, vn))
        face = _bcg_faces(f, ax, un, grad[ax], transverse, src, dt)
        flux[ax] = face * uf[ax]
    return flux


def advection(tracers: list[CellField], uf: FaceField, dt: float, src: list[CellField] | None = None,
              ops=None, limiter: Limiter | None = None) -> None:
    """Explicit update ``f += dt * (flux[k] - flux[k+1]) / Delta`` (Basilisk ``advection()``)."""
    from ...discretization.fd2 import FD2FaceComplex  # noqa: PLC0415

    ops = ops or FD2FaceComplex()
    src = src or [None] * len(tracers)
    for f, s in zip(tracers, src):
        flux = tracer_fluxes(f, uf, dt, s, limiter=limiter)
        f.interior = f.interior - dt * ops.div_f(flux, periodic=f.bc.periodic_axes(f.grid.ndim))
        f.fill_ghosts()


def prediction(u: VectorField, g: VectorField, dt: float, limiter: Limiter | None = None) -> FaceField:
    """Face velocity at t + dt/2 from the centred velocity (Basilisk ``prediction()``)."""
    grid = u.grid
    if not grid.is_uniform:
        raise NotImplementedError("BCG on a non-uniform grid is not implemented yet")
    u.fill_ghosts()
    uf = FaceField(grid, name="uf")
    for ax in range(grid.ndim):
        comp = u[ax]
        # du.x: derivative of u.x along x only, with scalar default BCs
        du = gradients(comp, limiter)[ax]
        bc = BoundaryConditions()
        for a in range(grid.ndim):
            for side in (0, 1):
                bc.set((a, side), Periodic() if comp.bc.is_periodic(a) else Neumann(0.0))
        du = CellField(grid, du.data, bc).fill_ghosts()
        left, right = grid.face_neighbours(ax)
        un = dt * (comp.data[left] + comp.data[right]) / (2.0 * grid.h[ax])
        transverse = [(t, _cells_along(u[t].data, grid, ax)) for t in range(grid.ndim) if t != ax]
        uf[ax] = _bcg_faces(comp, ax, un, du, transverse, g[ax], dt)
    return uf

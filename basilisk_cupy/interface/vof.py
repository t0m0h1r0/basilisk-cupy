"""Geometric VOF advection (Basilisk ``vof.h``), 2-D, uniform grid.

Direction-split, conservative, non-diffusive. For each sweep along ``x``::

    c += dt (F_{i-1/2} - F_{i+1/2} + cc (u_{i+1/2} - u_{i-1/2})) / Delta

with ``F = cf u`` and ``cf`` the fraction of the upwind cell that crosses the face during
``dt`` (PLIC: MYC normal, ``rectangle_fraction``). ``cc = (c > 1/2)`` is frozen at the start of
the step (Weymouth & Yue 2010): the multi-dimensional update conserves mass exactly when the
face velocity is discretely divergence-free. Sweeps alternate with the parity of the
iteration index. VOF concentrations (tracers ``t = c f``, or ``(1 - c) f`` for ``inverse``)
are advected with the same upwind volume fraction and a BCG-upwinded ``f``.

The face velocity must be discretely divergence-free for the face difference: with FCCD
operators use ``ops.wall_free_flux(uf, periodic)`` (``docs/notes/03_face_divergence.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fields import CellField, FaceField
from .fractions import reconstruction
from .geometry import rectangle_fraction

SEPS = 0.0  # Basilisk common.h (without embedded boundaries)
CMIN = 0.5  # vof_concentration_gradient: switch to one-sided away from the interface


@dataclass
class VOFTracer:
    """A VOF concentration ``t = c f`` (``inverse``: ``(1 - c) f``) carried with an interface."""

    field: CellField
    inverse: bool = False
    gradient: object = None  # slope limiter g(l, c, r) of f, None for centred differences, "ccd" (docs/notes/72) or
    #                          "ccd2" (with the parabola term, docs/notes/75)
    ccd_cmin: float = CMIN  # the phase fraction from which a cell joins a CCD run (0.99: the interfacial cells keep
    #                         vof.h's gradient, docs/notes/75)


@dataclass
class VOFInterface:
    """A volume-fraction field and its VOF concentrations."""

    c: CellField
    tracers: list = field(default_factory=list)


def _pad(arr, periodic, xp):
    """One ghost layer: periodic wrap, otherwise symmetric (Basilisk's default for derived fields)."""
    out = arr
    for ax in range(2):
        mode = "wrap" if periodic[ax] else "edge"
        width = [(0, 0), (0, 0)]
        width[ax] = (1, 1)
        out = xp.pad(out, width, mode=mode)
    return out


def _with_ghosts(f: CellField):
    """Interior plus one ghost layer (corners included) from the field's boundary conditions."""
    f.fill_ghosts()
    g = f.grid.ghosts
    return f.data[g - 1: f.data.shape[0] - g + 1, g - 1: f.data.shape[1] - g + 1]


def _shift(a, axis, k):
    """Rows ``1 + k .. n + k`` of a padded array along ``axis`` (``k`` in -1, 0, 1)."""
    sl = [slice(1, a.shape[0] - 1), slice(1, a.shape[1] - 1)]
    n = a.shape[axis] - 2
    sl[axis] = slice(1 + k, 1 + k + n)
    return a[tuple(sl)]


def _concentration_gradient(xp, cp, tp, axis, inverse, limiter):
    """Basilisk ``vof_concentration_gradient_x`` (times Delta) on the interior cells."""
    cl, cc, cr = (_shift(cp, axis, k) for k in (-1, 0, 1))
    tl, tc, tr = (_shift(tp, axis, k) for k in (-1, 0, 1))
    if inverse:
        cl, cc, cr = 1.0 - cl, 1.0 - cc, 1.0 - cr
    safe = lambda v: xp.where(v == 0.0, 1.0, v)  # noqa: E731
    fl, fc, fr = tl / safe(cl), tc / safe(cc), tr / safe(cr)
    centred = limiter(fl, fc, fr) if limiter is not None else (fr - fl) / 2.0
    g = xp.where(cl >= CMIN, centred, fr - fc)
    g = xp.where(cr >= CMIN, g, xp.where(cl >= CMIN, fc - fl, 0.0))
    return xp.where(cc >= CMIN, g, 0.0)


MIN_CCD_RUN = 6  # the shortest run of a phase's cells differentiated by CCD (shorter runs keep vof.h's gradient)


def _ccd_concentration_gradient(xp, cp, tp, axis, inverse, base, periodic_axis: bool, second: bool = False,
                                cmin: float = CMIN):
    """``f'`` (times Delta) of the concentration ``f = t / c`` by CCD along runs of the phase's cells (``c >= cmin``,
    or ``1 - c`` for ``inverse``), with extrapolation closures at the run ends (the phase ends there); runs shorter
    than ``MIN_CCD_RUN`` keep ``base`` (vof.h's gradient). The concentration is smooth within its phase, so its CCD
    derivative does not cross the interface. ``second``: also ``f''`` (times Delta^2, zero outside the CCD runs), for
    the parabola term of the flux (docs/notes/75)."""
    import numpy as np  # noqa: PLC0415

    from ..backend import to_numpy  # noqa: PLC0415
    from ..discretization.ccd import ccd_matrices  # noqa: PLC0415

    c = to_numpy(_shift(cp, axis, 0))
    t = to_numpy(_shift(tp, axis, 0))
    if inverse:
        c = 1.0 - c
    f = np.where(c > 0.0, t / np.where(c > 0.0, c, 1.0), 0.0)
    out = to_numpy(base).copy()
    out2 = np.zeros_like(out)
    F, C, O, O2 = (np.moveaxis(a, axis, 0) for a in (f, c, out, out2))
    n, m = F.shape
    for j in range(m):
        ok = C[:, j] >= cmin
        if periodic_axis and ok.all():
            d1, d2 = ccd_matrices(n, 1.0, "periodic", "periodic")[:2]
            O[:, j] = d1 @ F[:, j]
            O2[:, j] = d2 @ F[:, j]
            continue
        k = 0
        while k < n:
            if not ok[k]:
                k += 1
                continue
            e = k
            while e < n and ok[e]:
                e += 1
            if e - k >= MIN_CCD_RUN:
                d1, d2 = ccd_matrices(e - k, 1.0, "extrapolate", "extrapolate")[:2]
                O[k:e, j] = d1 @ F[k:e, j]
                O2[k:e, j] = d2 @ F[k:e, j]
            k = e
    if second:
        return xp.asarray(out), xp.asarray(out2)
    return xp.asarray(out)


def _sweep(interface: VOFInterface, uf: FaceField, dt: float, axis: int, cc, tcs, periodic):
    c = interface.c
    grid = c.grid
    xp = grid.xp
    h = float(grid.h[axis])
    cp = _with_ghosts(c)
    nx, ny, alpha = reconstruction(cp)
    nxp, nyp, alp = (_pad(a, periodic, xp) for a in (nx, ny, alpha))
    # along the sweep axis: normal component "x" is the sweep direction
    nn, nt = (nxp, nyp) if axis == 0 else (nyp, nxp)
    u = uf[axis]  # faces 0..n along axis
    un = u * dt / (h + SEPS)
    s = xp.where(un > 0.0, 1.0, -1.0)
    # upwind cell of face k: cell k - 1 (s = +1) or cell k (s = -1), in padded indexing k or k + 1
    def upwind(a):
        lo = _faces(a, axis, 0)   # padded cell k (= interior cell k - 1)
        hi = _faces(a, axis, 1)   # padded cell k + 1 (= interior cell k)
        return xp.where(s > 0.0, lo, hi)
    cu = upwind(cp)
    a_n, a_t, a_al = upwind(nn), upwind(nt), upwind(alp)
    bx = s * un - 0.5
    # Basilisk's foreach_dimension rotates the coordinates: the sweep direction is always the first one
    frac = rectangle_fraction(-s * a_n, a_t, a_al, -0.5, -0.5, bx, 0.5)
    cf = xp.where((cu <= 0.0) | (cu >= 1.0), cu, frac)
    flux = cf * u
    cfl = float(xp.max(un * s)) if u.size else 0.0
    tfluxes = []
    for tr, tcl in zip(interface.tracers, tcs):
        tp = _with_ghosts(tr.field)
        g2f = None
        if isinstance(tr.gradient, str) and tr.gradient in ("ccd", "ccd2"):
            base = _concentration_gradient(xp, cp, tp, axis, tr.inverse, None)
            if tr.gradient == "ccd2":
                g, g2 = _ccd_concentration_gradient(xp, cp, tp, axis, tr.inverse, base, bool(periodic[axis]), True,
                                                    tr.ccd_cmin)
                g2f = _pad(g2, periodic, xp)
            else:
                g = _ccd_concentration_gradient(xp, cp, tp, axis, tr.inverse, base, bool(periodic[axis]),
                                                cmin=tr.ccd_cmin)
        else:
            g = _concentration_gradient(xp, cp, tp, axis, tr.inverse, tr.gradient)
        gf = _pad(g, periodic, xp)
        ci = 1.0 - cu if tr.inverse else cu
        cf1 = 1.0 - cf if tr.inverse else cf
        ti, gi = upwind(tp), upwind(gf)
        ff = ti / xp.where(ci > 1e-10, ci, 1.0) + s * xp.minimum(1.0, 1.0 - s * un) * gi / 2.0
        if g2f is not None:  # the swept mean of the mean-preserving parabola: (1 - |C|)(1 - 2|C|)/12 f'' h^2
            a = xp.minimum(1.0, s * un)
            ff = ff + (1.0 - a) * (1.0 - 2.0 * a) / 12.0 * upwind(g2f)
        tfluxes.append(xp.where(ci > 1e-10, ff * cf1 * u, 0.0))
    n = grid.n[axis]
    lo, hi = (slice(0, n), slice(1, n + 1))
    div_u = _take(u, axis, hi) - _take(u, axis, lo)
    c.interior = c.interior + dt * (_take(flux, axis, lo) - _take(flux, axis, hi) + cc * div_u) / h
    for tr, tcl, tf in zip(interface.tracers, tcs, tfluxes):
        tr.field.interior = tr.field.interior + dt * (_take(tf, axis, lo) - _take(tf, axis, hi) + tcl * div_u) / h
    return cfl


def _take(a, axis, sl):
    idx = [slice(None), slice(None)]
    idx[axis] = sl
    return a[tuple(idx)]


def _faces(a, axis, k):
    """Padded cells ``k .. k + n`` along ``axis`` (one per face), interior rows across."""
    n = a.shape[axis] - 2
    idx = [slice(1, a.shape[0] - 1), slice(1, a.shape[1] - 1)]
    idx[axis] = slice(k, k + n + 1)
    return a[tuple(idx)]


def vof_advection(interfaces: list, uf: FaceField, dt: float, i: int) -> float:
    """One VOF step (Basilisk ``vof_advection``); returns the largest face CFL ``|u| dt / Delta``."""
    cfl = 0.0
    for itf in interfaces:
        c = itf.c
        if c.grid.ndim != 2 or not c.grid.is_uniform:
            raise NotImplementedError("VOF advection is implemented for uniform 2-D grids")
        periodic = c.bc.periodic_axes(2)
        cc = (c.interior > 0.5).astype(c.interior.dtype)
        tcs = []
        for tr in itf.tracers:
            t = tr.field.interior
            if tr.inverse:
                tcs.append(c.grid.xp.where(c.interior < 0.5, t / c.grid.xp.where(c.interior < 0.5, 1.0 - c.interior, 1.0), 0.0))
            else:
                tcs.append(c.grid.xp.where(c.interior > 0.5, t / c.grid.xp.where(c.interior > 0.5, c.interior, 1.0), 0.0))
        for d in range(2):
            cfl = max(cfl, _sweep(itf, uf, dt, (i + d) % 2, cc, tcs, periodic))
    return cfl

"""Two-phase flows: VOF interface, variable density/viscosity, interfacial forces
(Basilisk ``two-phase.h`` / ``two-phase-generic.h``, ``iforce.h``, ``reduced.h``), 2-D.

Attached to a :class:`~basilisk_cupy.time.basilisk_centered.CenteredNS` through its stage hooks:

* ``vof``: geometric VOF advection of ``f`` with the projected face velocity
  (``vof_first=True``: at the start of the step with the previous ``dt``, Basilisk
  ``event vof (i++, first)``);
* ``properties``: ``alpha = 1/rho(f_face)``, ``rho(f)``, ``mu(f_face)`` with arithmetic averages
  of the (optionally vertex-filtered) volume fraction;
* ``acceleration``: interfacial forces ``alpha phi grad f`` (``iforce.h``) from potentials
  ``phi`` registered by modules such as reduced gravity ``phi = [rho] G . (x - Z)``
  (``reduced.h``) and surface tension ``phi = sigma kappa`` (``tension.h``, height-function
  curvature), with tension's explicit time-step bound ``sqrt(rho_m Delta^3 / (pi sigma))``.

The VOF scheme caps the CFL number at 1/2.

``pressure_jump=True`` replaces ``iforce.h``'s smeared force and the arithmetic face densities by
the ghost-fluid method on the PLIC interface (as :class:`~.cls_two_phase.TwoPhaseCLS` does on the
``psi = 1/2`` crossing, docs/notes/16 and 23):

* phase of a cell: ``f`` in full cells; in mixed cells the side of the PLIC line
  ``n . x = alpha`` (cell coordinates) its centre is on, ``H = [alpha > 0]``;
* a face is cut when ``H_L != H_R``; the crossing is where the PLIC line of ``L`` or ``R``
  meets the segment between the two centres, ``theta`` its fraction from ``L`` (the line of the
  cell that contains the crossing first, then the other one, then ``(f_L - 1/2)/(f_L - f_R)``);
* face coefficient ``1/(theta rho_L + (1 - theta) rho_R)`` (``rho`` of the phases ``H``), and
  ``1/rho`` of the phase on uncut faces;
* jump ``j`` = the potentials (``sigma kappa`` from the height-function curvature, reduced
  gravity) of the two cells, weighted by ``theta`` (one of them where the other has none),
  entered as the face acceleration ``alpha_f j (H_R - H_L)/Delta``.
"""

from __future__ import annotations

import math

from ..fields import CellField, FaceField
from ..constants import PI
from .curvature import curvature
from .heights import NODATA, pad_bc, position
from .vof import VOFInterface, vof_advection


class TwoPhase:
    def __init__(self, ns, f: CellField, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0,
                 mu2: float = 0.0, vof_first: bool = False, filtered: bool = False,
                 pressure_jump: bool = False) -> None:
        if f.grid.ndim != 2:
            raise NotImplementedError("two-phase flows are implemented in 2-D")
        self.ns, self.f = ns, f
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2
        self.filtered = filtered
        self.pressure_jump = pressure_jump
        self.interface = VOFInterface(f)
        self.potentials = []  # callables (two_phase) -> phi (cell array, NODATA outside the interface)
        ns.cfl = min(ns.cfl, 0.5)
        ns.on("set_dtmax" if vof_first else "vof", self._vof)
        ns.on("properties", self._properties)
        ns.on("acceleration", self._interfacial_force)
        if mu1 or mu2:
            ns.mu = FaceField(f.grid, name="mu")

    # -- properties ---------------------------------------------------------------------
    def rho(self, f):
        xp = self.f.grid.xp
        return xp.minimum(xp.maximum(f, 0.0), 1.0) * (self.rho1 - self.rho2) + self.rho2

    def mu(self, f):
        xp = self.f.grid.xp
        return xp.minimum(xp.maximum(f, 0.0), 1.0) * (self.mu1 - self.mu2) + self.mu2

    def smoothed(self):
        """``sf``: ``f`` itself, or its vertex average (``FILTERED``) with the boundary conditions of ``f``."""
        f = self.f.interior
        if not self.filtered:
            return f
        p = pad_bc(f, self.f.bc, 1)
        return (4.0 * f + 2.0 * (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1])
                + p[:-2, :-2] + p[2:, :-2] + p[2:, 2:] + p[:-2, 2:]) / 16.0

    def _properties(self, ns) -> None:
        if self.pressure_jump:
            return self._gfm_properties(ns)
        sf = self.smoothed()
        sp = pad_bc(sf, self.f.bc, 1)
        n = sf.shape
        for ax in range(2):
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            ff = (sp[tuple(hi)] + sp[tuple(lo)]) / 2.0
            ns.alpha[ax] = 1.0 / self.rho(ff)
            if ns.mu is not None and (self.mu1 or self.mu2):
                ns.mu[ax] = self.mu(ff)
        ns.rho = self.rho(sf)

    # -- VOF ------------------------------------------------------------------------------
    def _vof(self, ns) -> None:
        vof_advection([self.interface], ns.uf, ns.clock.dt, ns.clock.i)

    # -- interfacial forces (iforce.h) ------------------------------------------------------
    def _interfacial_force(self, ns) -> None:
        if self.pressure_jump:
            return self._gfm_force(ns)
        if not self.potentials:
            return
        xp = self.f.grid.xp
        phi = None
        for pot in self.potentials:  # all potentials are computed before f is clamped (reduced.h runs first)
            p = pot(self)
            phi = p if phi is None else xp.where((phi == NODATA) | (p == NODATA), xp.where(phi == NODATA, p, phi), phi + p)
        self.f.interior = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)
        fp = pad_bc(self.f.interior, self.f.bc, 1)
        php = pad_bc(phi, self.f.bc, 1)
        n = phi.shape
        h = float(self.f.grid.h[0])
        ns.a_is_constant = False
        for ax in range(2):
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            fl, fr = fp[tuple(lo)], fp[tuple(hi)]
            pl, pr = php[tuple(lo)], php[tuple(hi)]
            okl, okr = pl < NODATA, pr < NODATA
            phif = xp.where(okl & okr, (pr + pl) / 2.0, xp.where(okr, pr, xp.where(okl, pl, 0.0)))
            add = ns.alpha[ax] / 1.0 * phif * (fr - fl) / h
            ns.a[ax] = ns.a[ax] + xp.where(fr != fl, add, 0.0)

    def add_surface_tension(self, sigma: float) -> None:
        """Surface tension (``tension.h``): ``phi = sigma kappa`` and the capillary time-step bound."""
        self.sigma = float(sigma)
        self.potentials.append(lambda tp: curvature(tp.f.interior, tp.f.bc, tp.f.grid, tp.sigma))
        self.ns.on("stability", self._capillary_dtmax)

    def _capillary_dtmax(self, ns) -> None:
        xp = self.f.grid.xp
        amin = min(float(xp.min(a)) for a in ns.alpha.components)
        amax = max(float(xp.max(a)) for a in ns.alpha.components)
        rhom = (1.0 / amin + 1.0 / amax) / 2.0
        dmin = float(min(self.f.grid.h))
        dt = math.sqrt(rhom * (dmin * dmin * dmin) / (PI * self.sigma))
        if dt < ns.dtmax:
            ns.dtmax = dt

    def add_reduced_gravity(self, G=(0.0, 0.0), Z=(0.0, 0.0)) -> None:
        """Reduced gravity (``reduced.h``): ``phi = (rho2 - rho1) G . (x - Z)`` at the interface."""
        G1 = ((self.rho2 - self.rho1) * G[0], (self.rho2 - self.rho1) * G[1])
        self.potentials.append(lambda tp: position(tp.f.interior, tp.f.bc, tp.f.grid, G1, Z))

    # -- ghost-fluid pressure jump on the PLIC interface ---------------------------------------
    def _plic_cut(self, ax):
        """``(H_L, H_R, cut, theta)`` on the faces normal to ``ax``."""
        from .fractions import reconstruction  # noqa: PLC0415

        xp = self.f.grid.xp
        f = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)
        nx, ny, alpha = reconstruction(pad_bc(f, self.f.bc, 1))
        mixed = (f > 0.0) & (f < 1.0)
        H = xp.where(mixed, alpha > 0.0, f > 0.5)
        n_ax = nx if ax == 0 else ny
        safe = xp.where(xp.abs(n_ax) > 1e-12, n_ax, 1.0)
        xs = xp.where(mixed & (xp.abs(n_ax) > 1e-12), alpha / safe, xp.nan)  # crossing of the centre line
        n = f.shape

        def faces(a):
            ap = pad_bc(a, self.f.bc, 1) if a.dtype != bool else pad_bc(a.astype(f.dtype), self.f.bc, 1) > 0.5
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            return ap[tuple(lo)], ap[tuple(hi)]

        hl, hr = faces(H)
        fl, fr = faces(f)
        xl, xr = faces(xp.nan_to_num(xs, nan=9.0))  # 9: no line
        tl, tr = xl, 1.0 + xr  # the crossing from L's line, from R's line (fraction of the segment)
        okl, okr = (tl >= 0.0) & (tl <= 1.0) & (xl < 8.0), (tr >= 0.0) & (tr <= 1.0) & (xr < 8.0)
        inl, inr = okl & (tl <= 0.5), okr & (tr >= 0.5)
        lin = xp.where(fl != fr, (fl - 0.5) / xp.where(fl != fr, fl - fr, 1.0), 0.5)
        theta = xp.where(inl, tl, xp.where(inr, tr, xp.where(okl, tl, xp.where(okr, tr, lin))))
        theta = xp.minimum(xp.maximum(theta, 0.0), 1.0)
        return hl, hr, hl != hr, theta

    def _gfm_properties(self, ns) -> None:
        xp = self.f.grid.xp
        for ax in range(2):
            hl, hr, cut, theta = self._plic_cut(ax)
            rl = xp.where(hl, self.rho1, self.rho2)
            rr = xp.where(hr, self.rho1, self.rho2)
            ns.alpha[ax] = 1.0 / (theta * rl + (1.0 - theta) * rr)
            if ns.mu is not None and (self.mu1 or self.mu2):
                sp = pad_bc(self.f.interior, self.f.bc, 1)
                n = self.f.interior.shape
                lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
                lo[ax] = slice(0, n[ax] + 1)
                hi = list(lo)
                hi[ax] = slice(1, n[ax] + 2)
                ns.mu[ax] = self.mu((sp[tuple(hi)] + sp[tuple(lo)]) / 2.0)
        ns.rho = self.rho(self.f.interior)

    def _gfm_force(self, ns) -> None:
        if not self.potentials:
            return
        xp = self.f.grid.xp
        phi = None
        for pot in self.potentials:
            p = pot(self)
            phi = p if phi is None else xp.where((phi == NODATA) | (p == NODATA), xp.where(phi == NODATA, p, phi), phi + p)
        n = phi.shape
        h = float(self.f.grid.h[0])
        php = pad_bc(phi, self.f.bc, 1)
        ns.a_is_constant = False
        for ax in range(2):
            hl, hr, cut, theta = self._plic_cut(ax)
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            pl, pr = php[tuple(lo)], php[tuple(hi)]
            okl, okr = pl < NODATA, pr < NODATA
            j = xp.where(okl & okr, (1.0 - theta) * pl + theta * pr, xp.where(okr, pr, xp.where(okl, pl, 0.0)))
            sign = hr.astype(phi.dtype) - hl.astype(phi.dtype)
            ns.a[ax] = ns.a[ax] + xp.where(cut, ns.alpha[ax] * j * sign / h, 0.0)

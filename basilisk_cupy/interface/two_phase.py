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

``conserving=True`` is Basilisk's ``navier-stokes/conserving.h`` (momentum-conserving VOF advection of the velocity):
the centred solver's advection is switched off (``stokes``), the step takes the CFL bound of the face velocity, and
the momentum of each phase, ``q1 = f rho1 u`` and ``q2 = (1 - f) rho2 u`` (the latter carried with ``1 - f``), is
advected as a VOF concentration of ``f`` with the same fluxes; the velocity is then ``u = (q1 + q2)/rho(f)``. The
ghost values of ``q`` follow the definition (``f rho u`` of the ghost cells), as Basilisk's ``boundary_q`` functions.
"""

from __future__ import annotations

import math

from ..fields import CellField, FaceField
from ..constants import PI
from .curvature import curvature
from .heights import NODATA, pad_bc, position
from .vof import VOFInterface, VOFTracer, vof_advection


class _MomentumField(CellField):
    """A phase's momentum component ``q = clamp(f) rho u`` (or ``(1 - clamp(f)) rho u``) whose ghost cells follow
    that definition from the ghosts of ``f`` and ``u`` (Basilisk ``boundary_q1_x`` / ``boundary_q2_x``)."""

    def bind(self, f, u, rho: float, inverse: bool):
        self._f, self._u, self._rho, self._inverse = f, u, rho, inverse
        return self

    def fill_ghosts(self):
        xp = self.grid.xp
        self._f.fill_ghosts()
        self._u.fill_ghosts()
        fc = xp.minimum(xp.maximum(self._f.data, 0.0), 1.0)
        full = ((1.0 - fc) if self._inverse else fc) * self._rho * self._u.data
        inner = self.interior.copy()
        self.data[...] = full
        self.interior = inner
        return self


class TwoPhase:
    def __init__(self, ns, f: CellField, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0,
                 mu2: float = 0.0, vof_first: bool = False, filtered: bool = False, conserving: bool = False,
                 conserving_gradient=None, conserving_ccd_cmin: float | None = None) -> None:
        """``conserving_gradient``: the gradient of the advected momentum concentrations (Basilisk: the velocity's,
        centred by default); ``"ccd"``: CCD along each phase's runs of cells (docs/notes/72); ``"ccd2"``: with the
        swept mean of the mean-preserving parabola from the same CCD solve (docs/notes/75). ``conserving_ccd_cmin``: the
        phase fraction from which a cell joins a CCD run (None: vof.h's CMIN = 0.5; 0.99 leaves the interfacial cells to
        the centred gradient, docs/notes/75)."""
        if f.grid.ndim != 2:
            raise NotImplementedError("two-phase flows are implemented in 2-D")
        self.ns, self.f = ns, f
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2
        self.filtered = filtered
        self.interface = VOFInterface(f)
        self.potentials = []  # callables (two_phase) -> phi (cell array, NODATA outside the interface)
        ns.cfl = min(ns.cfl, 0.5)
        self.conserving = conserving
        self.conserving_gradient = conserving_gradient
        self.conserving_ccd_cmin = conserving_ccd_cmin
        if conserving:
            if vof_first:
                raise NotImplementedError("conserving.h advects the momentum in the vof event")
            ns.stokes = True  # conserving.h switches the centred advection off ...
            ns.on("stability", self._cfl_dtmax)  # ... and takes the CFL bound itself
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
        if not self.conserving:
            vof_advection([self.interface], ns.uf, ns.clock.dt, ns.clock.i)
            return
        xp = self.f.grid.xp
        fc = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)
        tracers = []
        qs = []
        for u in ns.u.components:
            q1 = _MomentumField(self.f.grid, bc=u.bc, name="q1").bind(self.f, u, self.rho1, False)
            q2 = _MomentumField(self.f.grid, bc=u.bc, name="q2").bind(self.f, u, self.rho2, True)
            q1.interior = fc * self.rho1 * u.interior
            q2.interior = (1.0 - fc) * self.rho2 * u.interior
            g = self.conserving_gradient  # u's gradient: centred (Basilisk's default), or "ccd"
            kw = {} if self.conserving_ccd_cmin is None else {"ccd_cmin": float(self.conserving_ccd_cmin)}
            tracers += [VOFTracer(q1, gradient=g, **kw), VOFTracer(q2, inverse=True, gradient=g, **kw)]
            qs.append((q1, q2))
        saved = self.interface.tracers
        self.interface.tracers = list(saved) + tracers
        try:
            vof_advection([self.interface], ns.uf, ns.clock.dt, ns.clock.i)
        finally:
            self.interface.tracers = saved
        rho = self.rho(self.f.interior)
        for u, (q1, q2) in zip(ns.u.components, qs):
            u.interior = (q1.interior + q2.interior) / rho
        ns.u.fill_ghosts()

    def _cfl_dtmax(self, ns) -> None:
        """conserving.h's ``event stability``: ``dtmax = timestep (uf, dtmax)``."""
        from ..time.stepping import timestep  # noqa: PLC0415

        ns.dtmax = timestep(ns.uf, ns.dtmax, ns.cfl, ns.clock)

    # -- interfacial forces (iforce.h) ------------------------------------------------------
    def _interfacial_force(self, ns) -> None:
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

    def add_surface_tension(self, sigma: float, curvature_method: str = "basilisk") -> None:
        """Surface tension (``tension.h``): ``phi = sigma kappa`` and the capillary time-step bound.
        ``curvature_method``: ``"basilisk"`` (tension.h's heights, fd2 along the interface, fits) or ``"ccd"``
        (the interface position of every column from Basilisk's heights, differentiated by CCD along runs of
        columns, :func:`.vof_curvature.ccd_curvature`, docs/notes/66)."""
        self.sigma = float(sigma)
        # CCD's second derivative has 12/5 of the 3-point difference's gain at the grid scale (as the CLS's CCD
        # curvature, cls_two_phase.CCD_CAPILLARY_DT_FACTOR): the capillary step shrinks by sqrt(5/12)
        # (not ``capillary_dt_factor``: TwoPhaseCLS applies that name itself on top of this method)
        self._curvature_dt_factor = 1.0 if curvature_method == "basilisk" else math.sqrt(5.0 / 12.0)
        if curvature_method == "basilisk":
            self.potentials.append(lambda tp: curvature(tp.f.interior, tp.f.bc, tp.f.grid, tp.sigma))
        elif curvature_method == "ccd":
            from .vof_curvature import ccd_curvature  # noqa: PLC0415

            self.potentials.append(lambda tp: ccd_curvature(tp.f, tp.sigma))
        else:
            raise ValueError("curvature_method is 'basilisk' or 'ccd'")
        self.ns.on("stability", self._capillary_dtmax)

    def _capillary_dtmax(self, ns) -> None:
        xp = self.f.grid.xp
        amin = min(float(xp.min(a)) for a in ns.alpha.components)
        amax = max(float(xp.max(a)) for a in ns.alpha.components)
        rhom = (1.0 / amin + 1.0 / amax) / 2.0
        from .cls import min_width  # noqa: PLC0415

        dmin = min_width(self.f.grid)  # the smallest physical cell width (grid.h is the xi width on a mapped axis)
        dt = math.sqrt(rhom * (dmin * dmin * dmin) / (PI * self.sigma)) * getattr(self, "_curvature_dt_factor", 1.0)
        if dt < ns.dtmax:
            ns.dtmax = dt

    def add_reduced_gravity(self, G=(0.0, 0.0), Z=(0.0, 0.0)) -> None:
        """Reduced gravity (``reduced.h``): ``phi = (rho2 - rho1) G . (x - Z)`` at the interface."""
        G1 = ((self.rho2 - self.rho1) * G[0], (self.rho2 - self.rho1) * G[1])
        self.potentials.append(lambda tp: position(tp.f.interior, tp.f.bc, tp.f.grid, G1, Z))

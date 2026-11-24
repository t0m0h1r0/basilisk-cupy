"""Two-phase flow with the CLS interface (TwoPhaseFlow's ``psi``) on Basilisk's centred solver, 2-D.

Same coupling as :class:`~.two_phase.TwoPhase` (Basilisk ``two-phase.h`` + ``iforce.h``) with the
volume fraction replaced by ``psi``, so that only the interface representation differs from the
VOF baseline:

* transport: :meth:`.CLSInterface.advect` with the projected face velocity ``uf`` (stages A, B),
  optionally re-initialised every ``reinit_every`` steps;
* properties: ``rho = rho2 + (rho1 - rho2) psi`` at cells, ``alpha = 1/rho(psi_face)`` and
  ``mu(psi_face)`` with arithmetic face averages (paper 06d: arithmetic averages are the reference
  for the diffuse CLS interface; Basilisk's ``two-phase.h`` does the same with ``f``);
* surface tension: the CSF force ``sigma kappa grad psi`` (paper 02b, 09: balanced force) in
  ``iforce.h``'s face form ``alpha_f sigma kappa_f (psi_R - psi_L)/Delta``, ``kappa`` from
  :meth:`.CLSInterface.curvature` (zero outside the band ``0.01 < psi < 0.99``),
  ``kappa_f`` the face average; tension's time-step bound as in ``tension.h``;
* reduced gravity: ``phi = [rho] G . (x - Z)`` at every cell centre (``grad psi`` localises it;
  ``reduced.h`` evaluates it at the height-function position of the interface cells).

``pressure_jump=True`` replaces the last two by the paper's pressure-jump closure (09b1, 09b3) in
finite-volume form, the ghost-fluid method (Liu, Fedkiw & Kang 2000). The interface is the
``psi = 1/2`` crossing on each face, at the fraction ``theta = (psi_L - 1/2)/(psi_L - psi_R)`` of
the face's centre-to-centre segment:

* jump-corrected face gradient ``G_Gamma(p; j) = G p - B(j)`` with ``B(j) = j (H_R - H_L)/Delta``,
  ``H = [psi > 1/2]``, ``j = sigma kappa + [rho] G . (x - Z)`` interpolated to the crossing; it
  enters Basilisk's projection as the face acceleration ``alpha_f B(j)`` (the same face form as
  ``iforce.h`` with the sharp ``H`` in place of ``f``), so the projection, the velocity
  correction and the centred gradient all use ``G_Gamma``;
* phase-wise constant densities (split PPE): ``alpha_f = 1/rho_k`` on faces inside phase ``k``
  and, on cut faces, the GFM coefficient of ``[alpha d_n p] = 0``,
  ``alpha_f = 1/(theta rho_L + (1 - theta) rho_R)``. One multigrid solve of this operator is the
  two constant-density phase problems coupled by the jump conditions; Basilisk's compact
  two-point face stencils never cross the interface with an unknown jump, so HFE (needed by the
  paper's CCD stencils) has no role here.

Cell densities (momentum, viscosity) and face viscosities stay those of ``psi``.

Capillary time step: tension.h's ``sqrt(rho_m Delta^3 / (pi sigma))`` assumes a curvature whose
response to a grid-scale wiggle is that of 2nd-order differences, ``4/Delta^2``. CCD's second
derivative there is ``48/(5 Delta^2)`` (Nyquist symbol of the Chu--Fan scheme), so the explicit
capillary loop is stiffer by 12/5 and needs ``capillary_dt_factor = sqrt(5/12)``
(:data:`CCD_CAPILLARY_DT_FACTOR`). The default 1 keeps tension.h's bound (unstable at density
ratios 100 and 1000 with equal viscosities, N = 64; docs/notes/19).
"""

from __future__ import annotations

import math

from ..fields import CellField, FaceField
from .cls import CLSInterface
from .two_phase import TwoPhase

#: sqrt(4 / (48/5)): tension.h's bound for a CCD curvature (Nyquist symbol 48/5 against 4).
CCD_CAPILLARY_DT_FACTOR = math.sqrt(5.0 / 12.0)


class TwoPhaseCLS(TwoPhase):
    def __init__(self, ns, cls: CLSInterface, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0,
                 mu2: float = 0.0, vof_first: bool = False, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None,
                 pressure_jump: bool = False, capillary_dt_factor: float = 1.0) -> None:
        f: CellField = cls.psi
        if f.grid.ndim != 2:
            raise NotImplementedError("two-phase flows are implemented in 2-D")
        self.ns, self.f, self.cls = ns, f, cls
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2
        self.filtered = False
        self.reinit_every, self.reinit_method = reinit_every, reinit_method
        self.curvature_kw = curvature_kw or {}
        self.potentials = []
        self.pressure_jump = pressure_jump
        self.capillary_dt_factor = float(capillary_dt_factor)
        self._steps = 0
        ns.cfl = min(ns.cfl, 0.5)
        ns.on("set_dtmax" if vof_first else "vof", self._vof)
        ns.on("properties", self._properties)
        ns.on("acceleration", self._interfacial_force)
        if mu1 or mu2:
            ns.mu = FaceField(f.grid, name="mu")

    def _vof(self, ns) -> None:
        dt = ns.clock.dt
        if not dt:
            return
        self.cls.advect(ns.uf, dt)
        self._steps += 1
        if self.reinit_every and self._steps % self.reinit_every == 0:
            self.cls.reinitialize(method=self.reinit_method)

    def add_surface_tension(self, sigma: float) -> None:
        self.sigma = float(sigma)
        self.potentials.append(lambda tp: tp.sigma * tp.cls.curvature(**tp.curvature_kw))
        self.ns.on("stability", self._capillary_dtmax)

    def _capillary_dtmax(self, ns) -> None:
        saved = ns.dtmax
        ns.dtmax = float("inf")
        super()._capillary_dtmax(ns)
        dt = ns.dtmax * self.capillary_dt_factor
        ns.dtmax = min(saved, dt)

    def add_reduced_gravity(self, G=(0.0, 0.0), Z=(0.0, 0.0)) -> None:
        d = self.rho2 - self.rho1
        grid = self.f.grid

        def pot(tp):
            x, y = grid.meshgrid()
            return d * (G[0] * (x - Z[0]) + G[1] * (y - Z[1]))

        self.potentials.append(pot)

    # -- pressure-jump closure (GFM) -------------------------------------------------------------
    def _faces(self, cell, ax):
        """(left, right) cell values of every face along ``ax`` (ghosts from psi's conditions)."""
        from .heights import pad_bc  # noqa: PLC0415

        cp = pad_bc(cell, self.f.bc, 1)
        n = cell.shape
        lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
        lo[ax] = slice(0, n[ax] + 1)
        hi = list(lo)
        hi[ax] = slice(1, n[ax] + 2)
        return cp[tuple(lo)], cp[tuple(hi)]

    def _cut(self, ax):
        xp = self.f.grid.xp
        pl, pr = self._faces(self.f.interior, ax)
        hl, hr = pl > 0.5, pr > 0.5
        cut = hl != hr
        theta = xp.where(cut, (pl - 0.5) / xp.where(cut, pl - pr, 1.0), 0.0)
        return hl, hr, cut, theta

    def _properties(self, ns) -> None:
        super()._properties(ns)
        if not self.pressure_jump:
            return
        xp = self.f.grid.xp
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            rl = xp.where(hl, self.rho1, self.rho2)
            rr = xp.where(hr, self.rho1, self.rho2)
            ns.alpha[ax] = 1.0 / (theta * rl + (1.0 - theta) * rr)

    def _interfacial_force(self, ns) -> None:
        if not self.pressure_jump:
            return super()._interfacial_force(ns)
        if not self.potentials:
            return
        xp = self.f.grid.xp
        phi = None
        for pot in self.potentials:
            v = pot(self)
            phi = v if phi is None else phi + v
        h = float(self.f.grid.h[0])
        ns.a_is_constant = False
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            jl, jr = self._faces(phi, ax)
            j = (1.0 - theta) * jl + theta * jr
            sign = hr.astype(phi.dtype) - hl.astype(phi.dtype)
            ns.a[ax] = ns.a[ax] + xp.where(cut, ns.alpha[ax] * j * sign / h, 0.0)

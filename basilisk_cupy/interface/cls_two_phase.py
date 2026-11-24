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
"""

from __future__ import annotations

from ..fields import CellField, FaceField
from .cls import CLSInterface
from .two_phase import TwoPhase


class TwoPhaseCLS(TwoPhase):
    def __init__(self, ns, cls: CLSInterface, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0,
                 mu2: float = 0.0, vof_first: bool = False, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None) -> None:
        f: CellField = cls.psi
        if f.grid.ndim != 2:
            raise NotImplementedError("two-phase flows are implemented in 2-D")
        self.ns, self.f, self.cls = ns, f, cls
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2
        self.filtered = False
        self.reinit_every, self.reinit_method = reinit_every, reinit_method
        self.curvature_kw = curvature_kw or {}
        self.potentials = []
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

    def add_reduced_gravity(self, G=(0.0, 0.0), Z=(0.0, 0.0)) -> None:
        d = self.rho2 - self.rho1
        grid = self.f.grid

        def pot(tp):
            x, y = grid.meshgrid()
            return d * (G[0] * (x - Z[0]) + G[1] * (y - Z[1]))

        self.potentials.append(pot)

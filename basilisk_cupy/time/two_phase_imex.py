"""TwoPhaseFlow's standard two-phase step as one combination (paper §11b1, docs/notes/50).

:class:`TwoPhaseIMEX` is :class:`IMEXBDF2NS` with a :class:`TwoPhaseCLS` attached through the hook stages,
with the paper's choices fixed where this port has them:

====  ==========================================  ====================================================
stage paper (§11b1)                               here
====  ==========================================  ====================================================
1     CLS transport, TVD-RK3, projected u_f^n     ``CLSInterface.advect(uf, dt)`` at the ``vof`` stage
                                                  (after the step is chosen), FCCD face values with the
                                                  corrected divergence
2     Ridge--Eikonal + mass closure               ``reinit_every`` / ``reinit_method`` (default: none)
3     rho, mu linear in psi                       ``TwoPhase._properties`` (cut faces: GFM coefficient)
4     curvature from psi by CCD                   ``CLSInterface.curvature`` (``curvature_kw``)
5     EXT2 convection, implicit BDF2 viscosity    ``IMEXBDF2NS`` predictor, :class:`CCDViscosity`
      (CCD + defect correction)
5     buoyancy: gradient part to the pressure,    reduced gravity ``[rho] G.(x - Z)`` in the jump
      residual on faces                           (sharp form of ``Phi grad rho'``)
5a/6  split PPE with the pressure jump, IPC       ``pressure_jump=True``: one multigrid solve with the
                                                  GFM coefficient on cut faces, increment ``dp`` with
                                                  the projection step ``gamma dt``
7     correction with the jump-corrected gradient ``IMEXBDF2NS`` correction (same face acceleration)
====  ==========================================  ====================================================

``faces="fd2"`` (default) keeps Basilisk's 2nd-order face operators, which take any density ratio.
``faces="fccd"`` projects with FCCD on the smooth pressure ``p - H j_ext`` (docs/notes/32); that form
assumes equal densities, so it is refused with a density ratio until the split FCCD PPE exists.

``convection="central"`` (default) is the non-dissipative ``D_f(P_f(u) u_f)``. ``"fuccd"`` (FUCCD_4) and
``"uccd6"`` add the paper's ``sigma |a| h^7 d^8`` upwind dissipation and are kept for comparison only.
"""

from __future__ import annotations

from ..boundary import BoundaryConditions, Neumann, Periodic
from ..discretization.ccd import FCCDFaceComplex
from ..interface.cls import CLSInterface
from ..interface.cls_two_phase import TwoPhaseCLS
from ..terms.viscous import CCDViscosity
from .imex_bdf2 import IMEXBDF2NS

WALLS = ("left", "right", "bottom", "top")
_SIDE_AXIS = {"left": 0, "right": 0, "bottom": 1, "top": 1}


class TwoPhaseIMEX(IMEXBDF2NS):
    def __init__(self, grid, phi0, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0, mu2: float = 0.0,
                 sigma: float = 0.0, gravity=None, periodic: tuple = (), faces: str = "fd2",
                 convection: str = "central", eps: float | None = None, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None, jump_mode: str | None = None,
                 capillary_dt_factor: float | None = None, transport: str = "projected", **kw) -> None:
        """``phi0``: cell-centred signed distance, positive in phase 1 (``psi = 1``). ``gravity``: ``(G, Z)`` of
        the reduced gravity (``G`` the acceleration vector). Walls on the non-periodic sides (``uf`` normal
        zero there). Other keywords go to :class:`IMEXBDF2NS` (``fixed_dt``, ``tolerance``, ``DT``, ...)."""
        if transport not in ("projected", "ab2"):
            raise ValueError("transport is 'projected' or 'ab2'")
        if faces not in ("fd2", "fccd"):
            raise ValueError("faces is 'fd2' or 'fccd'")
        if faces == "fccd" and rho1 != rho2:
            raise NotImplementedError("faces='fccd' solves for p - H j_ext, which assumes equal densities "
                                      "(docs/notes/32); use faces='fd2' with a density ratio")
        periodic = tuple(periodic)
        ops = FCCDFaceComplex() if faces == "fccd" else None
        kw.setdefault("uf_normal", {w: 0.0 for w in WALLS if _SIDE_AXIS[w] not in periodic})
        kw.setdefault("viscous", CCDViscosity())
        super().__init__(grid, periodic=periodic, ops=ops, convection=self._convection(convection), **kw)
        bc = BoundaryConditions()
        for ax in range(grid.ndim):
            for side in (0, 1):
                bc.set((ax, side), Periodic() if ax in periodic else Neumann(0.0))
        self.cls = CLSInterface.from_level_set(grid, phi0, bc, eps=eps, ops=FCCDFaceComplex(order=6),
                                               divergence="full" if faces == "fccd" else "fccd")
        if jump_mode is None:
            jump_mode = "smooth" if faces == "fccd" else "sharp"
        self.tp = TwoPhaseCLS(self, self.cls, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, pressure_jump=True,
                              jump_mode=jump_mode, reinit_every=reinit_every, reinit_method=reinit_method,
                              curvature_kw=curvature_kw, capillary_dt_factor=capillary_dt_factor)
        if sigma:
            self.tp.add_surface_tension(sigma)
        if gravity is not None:
            G, Z = gravity
            self.tp.add_reduced_gravity(G, Z)
        self.faces, self.convection_name = faces, convection
        self.transport = transport
        self._uf_latest = self._uf_transport_prev = None  # u_f^n and u_f^{n-1} at the next transport
        if transport == "ab2":
            # registered after TwoPhaseCLS's transport hook, so it runs first; the second one restores uf
            self.on("vof", self._extrapolate_uf)
            self.on("vof", self._restore_uf, after=True)
            self.on("end_timestep", self._remember_uf, after=True)

    # -- AB2 transport velocity --------------------------------------------------------------------
    def _extrapolate_uf(self, ns) -> None:
        self._uf_saved = None
        if self._uf_transport_prev is None or not self.clock.dt:
            return
        from ..fields import FaceField  # noqa: PLC0415

        w = self.clock.dt / self._dt_prev
        self._uf_saved = self.uf
        self.uf = FaceField(self.grid, [(1.0 + 0.5 * w) * self.uf[ax] - 0.5 * w * self._uf_transport_prev[ax]
                                        for ax in range(self.grid.ndim)], "uf")

    def _restore_uf(self, ns) -> None:
        if self._uf_saved is not None:
            self.uf = self._uf_saved
            self._uf_saved = None

    def _remember_uf(self, ns) -> None:
        self._uf_transport_prev = self._uf_latest
        self._uf_latest = [self.uf[ax].copy() for ax in range(self.grid.ndim)]

    def _convection(self, name: str):
        if name == "central":
            return None  # IMEXBDF2NS's CentralFluxConvection on the shared face operators
        if name == "fuccd":
            from ..terms.convection.fuccd import FUCCDConvection  # noqa: PLC0415

            return FUCCDConvection(r=4)
        if name == "uccd6":
            from ..terms.convection.uccd6 import UCCD6Convection  # noqa: PLC0415

            return UCCD6Convection()
        raise ValueError("convection is 'central', 'fuccd' or 'uccd6'")

    # -- diagnostics -----------------------------------------------------------------------------
    @property
    def psi(self):
        return self.cls.psi.interior

    def pressure(self):
        """The pressure with its jump (``p~ + H j_ext`` in the smooth form)."""
        return self.tp.full_pressure(self)

    def mass(self) -> float:
        """``sum psi dV`` (the diffuse CLS mass the transport conserves)."""
        return float(self.cls.mass())

    def sharp_volume(self) -> float:
        """Volume of ``psi > 1/2`` counted by cells (a coarse sharp-interface volume)."""
        xp = self.grid.xp
        return float(xp.sum(self.psi > 0.5)) * float(self.grid.h[0] * self.grid.h[1])

    def kinetic_energy(self) -> float:
        """``sum rho |u|^2 / 2 dV`` with the cell density."""
        xp = self.grid.xp
        e = sum(xp.sum(self.rho * c.interior ** 2) for c in self.u)
        return 0.5 * float(e) * float(self.grid.h[0] * self.grid.h[1])

    def max_velocity(self) -> float:
        xp = self.grid.xp
        return max(float(xp.max(xp.abs(c.interior))) for c in self.u)

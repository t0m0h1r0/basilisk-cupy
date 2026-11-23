"""FUCCD_r momentum convection: conservative upwind CCD face fluxes.

TwoPhaseFlow (paper §4.7 ``04e1_fuccd.tex``, eq. ``fuccd_flux``; NS form in
``src/twophase/ns_terms/fuccd_convection.py``), for every component ``u_j``::

    -C_j = -sum_k D_f[ a_k P_f u_j + |a_k| sigma_r (-1)^r h_k^(2r-1) G_f (D2_k)^(r-1) u_j ]

``P_f``, ``G_f``, ``D_f`` are the FCCD face value, face gradient and face
divergence of :class:`FCCDFaceComplex` (4th order), ``a_k`` the face-normal
speed. ``r = 4`` (default) is the UCCD6-equivalent hyperviscous flux
(``-sigma |a| h^7 d^8/dx^8`` after the divergence), ``r = 1`` the first-order
donor flux (``sigma_1 = 1/2``: classical upwind).

Face speed: the paper takes the face velocity accepted by the last pressure
projection (``face_velocity="projected"``, the state ``uf``);
TwoPhaseFlow's NS term samples the cell velocity with ``P_f`` instead
(``face_velocity="interpolated"``).

Walls: the dissipative face gradient has no wall datum; its wall faces are set
to zero. The flux there is multiplied by the wall-normal face speed, which is
zero on an impermeable wall, so the wall flux is the FCCD central one, as the
paper's wall policy (``sec:fuccd_wall_closure``: walls are closed by ``a_f = 0``,
no separate dissipative wall stencil). ``r = 1`` uses ``G_f`` of the component
itself, with its boundary condition.

Explicit time stepping: the dissipative part has the symbol
``-sigma_r |a| mu_r(theta) / h``, ``mu_r = 4 sin^2(theta/2) (1 + omega_2^2/24) omega_2^(2(r-1))``
(paper eq. ``fuccd_periodic_symbol``), largest at ``theta = pi``:
``mu_4(pi) = 4954``, so CFL ``<= 0.27`` with ``sigma_4 = 1e-3`` under EXT2/BDF2
(:meth:`max_stable_dt`).
"""

from __future__ import annotations

from ...discretization.ccd import FCCDFaceComplex
from ...fields import FaceField, VectorField
from .uccd6 import EXT2_REAL_LIMIT, OMEGA2_SQ_NYQUIST

DEFAULT_SIGMA = {1: 0.5, 4: 1.0e-3}  # TwoPhaseFlow default_fuccd_sigma


class FUCCDConvection:
    def __init__(self, fccd: FCCDFaceComplex | None = None, r: int = 4, sigma: float | None = None,
                 face_velocity: str = "projected") -> None:
        if int(r) < 1:
            raise ValueError(f"FUCCD order r must be >= 1, got {r}")
        if face_velocity not in ("projected", "interpolated"):
            raise ValueError("face_velocity must be 'projected' or 'interpolated'")
        self.fccd = fccd or FCCDFaceComplex()
        self.r = int(r)
        self.sigma = DEFAULT_SIGMA.get(self.r, 1.0) if sigma is None else float(sigma)
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.face_velocity = face_velocity
        self.name = f"fuccd{self.r}"

    def nyquist_symbol(self) -> float:
        """``mu_r(pi)`` of the dissipative flux."""
        return 4.0 * (1.0 + OMEGA2_SQ_NYQUIST / 24.0) * OMEGA2_SQ_NYQUIST ** (self.r - 1)

    def max_stable_dt(self, u: VectorField, uf: FaceField | None = None) -> float:
        """Largest step for which the explicit (EXT2) dissipative flux stays stable."""
        grid = u.grid
        xp = grid.xp
        speeds = self.face_speeds(u, uf)
        rate = sum(self.sigma * float(xp.max(xp.abs(speeds[k]))) * self.nyquist_symbol() / grid.h[k]
                   for k in range(grid.ndim))
        return EXT2_REAL_LIMIT / rate if rate > 0.0 else float("inf")

    def dissipative_gradient(self, comp, axis: int):
        """``G_f (D2)^(r-1) u_j`` on the faces normal to ``axis``."""
        if self.r == 1:
            return self.fccd.grad_f(comp, axis)
        grid = comp.grid
        ccd = self.fccd.ccd
        periodic = comp.bc.is_periodic(axis)
        d = ccd.d2(comp, axis)
        for _ in range(self.r - 2):
            d = ccd.d2_array(d, grid, axis, periodic)
        return self.fccd.grad_f_array(d, grid, axis, periodic)

    def face_speeds(self, u: VectorField, uf: FaceField | None):
        if self.face_velocity == "projected":
            if uf is None:
                raise ValueError("FUCCD with face_velocity='projected' needs the face velocity uf")
            return [uf[k] for k in range(u.grid.ndim)]
        return [self.fccd.interp_f(u[k], k) for k in range(u.grid.ndim)]

    def tendency(self, u: VectorField, uf: FaceField | None = None) -> list:
        """``-C(u)`` for every component (interior arrays)."""
        grid = u.grid
        xp = grid.xp
        u.fill_ghosts()
        speeds = self.face_speeds(u, uf)
        sign = (-1.0) ** self.r
        out = []
        for comp in u:
            fluxes = []
            for k in range(grid.ndim):
                a = speeds[k]
                coeff = self.sigma * sign * grid.h[k] ** (2 * self.r - 1)
                fluxes.append(a * self.fccd.interp_f(comp, k)
                              + xp.abs(a) * (coeff * self.dissipative_gradient(comp, k)))
            out.append(-self.fccd.div_f(FaceField(grid, fluxes)))
        return out

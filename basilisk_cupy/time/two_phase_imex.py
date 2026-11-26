"""TwoPhaseFlow's standard two-phase step as one combination (paper §11b1, docs/notes/50).

:class:`TwoPhaseIMEX` is :class:`IMEXMultistepNS` (default scheme ``sbdf2`` = :class:`IMEXBDF2NS`) with a
:class:`TwoPhaseCLS` attached through the hook stages,
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
from ..terms.viscous import CCDViscosity, FD2Viscosity
from .imex_multistep import IMEXMultistepNS
from .schemes import EXPLICIT_RK, ab_beta

WALLS = ("left", "right", "bottom", "top")
_SIDE_AXIS = {"left": 0, "right": 0, "bottom": 1, "top": 1}
TRANSPORTS = ("frozen", "ab2", "ab3", "pc")


def _check_faces(faces: str, rho1: float, rho2: float) -> None:
    if faces not in ("fd2", "fccd", "phase-ccd"):
        raise ValueError("faces is 'fd2', 'fccd' or 'phase-ccd'")
    if faces == "fccd" and rho1 != rho2:
        raise NotImplementedError("faces='fccd' solves for p - H j_ext, which assumes equal densities "
                                  "(docs/notes/32); use faces='fd2' or 'phase-ccd' with a density ratio")


def _face_ops(faces: str, taylor: bool = True):
    if faces == "fd2":
        return None
    if faces == "fccd":
        return FCCDFaceComplex()
    from ..discretization.phase_ccd import PhaseCCDFaces  # noqa: PLC0415

    return PhaseCCDFaces(FCCDFaceComplex(), taylor=taylor)


def _convection(name: str):
    if name == "central":
        return None  # IMEXBDF2NS's CentralFluxConvection on the shared face operators
    if name == "fuccd":
        from ..terms.convection.fuccd import FUCCDConvection  # noqa: PLC0415

        return FUCCDConvection(r=4)
    if name == "uccd6":
        from ..terms.convection.uccd6 import UCCD6Convection  # noqa: PLC0415

        return UCCD6Convection()
    raise ValueError("convection is 'central', 'fuccd' or 'uccd6'")


def _attach_two_phase(ns, phi0, *, periodic, faces, rho1, rho2, mu1, mu2, sigma, gravity, eps, reinit_every,
                      reinit_method, curvature_kw, jump_mode, capillary_dt_factor, cls_divergence=None,
                      jump_interp=None, cell_coefficient=None) -> None:
    """CLS (FCCD face values of order 6, corrected divergence) and TwoPhaseCLS with the GFM pressure jump."""
    grid = ns.grid
    bc = BoundaryConditions()
    for ax in range(grid.ndim):
        for side in (0, 1):
            bc.set((ax, side), Periodic() if ax in periodic else Neumann(0.0))
    ns.cls = CLSInterface.from_level_set(grid, phi0, bc, eps=eps, ops=FCCDFaceComplex(order=6),
                                         divergence=cls_divergence or ("fccd" if faces == "fd2" else "full"))
    if jump_mode is None:
        jump_mode = "smooth" if faces == "fccd" else "sharp"
    ns.tp = TwoPhaseCLS(ns, ns.cls, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, pressure_jump=True,
                        jump_mode=jump_mode, reinit_every=reinit_every, reinit_method=reinit_method,
                        curvature_kw=curvature_kw, capillary_dt_factor=capillary_dt_factor,
                        jump_interp=jump_interp or ("hermite" if faces == "phase-ccd" else "linear"),
                        cell_coefficient=cell_coefficient or "gfm")
    if faces == "phase-ccd":
        # the cut faces move with the interface: refresh the operator once the properties are set
        ns.on("properties", lambda _ns: _ns.ops.update(_ns.tp, _ns.alpha), after=True)
    if sigma:
        ns.tp.add_surface_tension(sigma)
    if gravity is not None:
        G, Z = gravity
        ns.tp.add_reduced_gravity(G, Z)


class _VOFState:
    """The volume fraction with the interface the two-phase steppers expect of ``ns.cls`` (``psi``, ``mass``,
    ``sharp_volume``): ``psi`` is the VOF fraction ``f``."""

    def __init__(self, f) -> None:
        self.psi = f
        self.dv = float(f.grid.h[0] * f.grid.h[1])

    def mass(self) -> float:
        return float(self.psi.grid.xp.sum(self.psi.interior)) * self.dv

    def sharp_volume(self) -> float:
        return self.mass()


def _attach_vof(ns, phi0, *, periodic, rho1, rho2, mu1, mu2, sigma, gravity, jump: bool = False,
                crossing: str = "sum", cell_coefficient: str = "mean", refine: int = 1,
                curvature_method: str = "basilisk") -> None:
    """Basilisk's VOF (vof.h, PLIC) with two-phase.h, tension.h (height-function curvature, CSF in iforce.h form)
    and reduced.h; ``phi0`` a level function ``(x, y) -> value`` positive in phase 1 (evaluated on the vertices),
    or cell values of a signed distance (averaged to the vertices)."""
    from ..boundary import BoundaryConditions as BC  # noqa: PLC0415
    from ..fields import CellField  # noqa: PLC0415
    from ..interface.fractions import fraction, fractions  # noqa: PLC0415
    from ..interface.two_phase import TwoPhase  # noqa: PLC0415

    grid = ns.grid
    bc = BC()
    for ax in range(grid.ndim):
        for side in (0, 1):
            bc.set((ax, side), Periodic() if ax in periodic else Neumann(0.0))
    if callable(phi0):
        f = fraction(grid, phi0, refine=refine)
    else:
        xp = grid.xp
        q = xp.asarray(phi0)
        qp = xp.pad(q, 1, mode="wrap" if periodic else "edge")
        vert = 0.25 * (qp[:-1, :-1] + qp[1:, :-1] + qp[:-1, 1:] + qp[1:, 1:])
        f = fractions(vert)
    ns.vof = CellField.from_interior(grid, f, bc)
    ns.cls = _VOFState(ns.vof)
    if jump:
        from ..interface.vof_jump import TwoPhaseVOFJump  # noqa: PLC0415

        ns.tp = TwoPhaseVOFJump(ns, ns.vof, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, crossing=crossing,
                                    cell_coefficient=cell_coefficient)
    else:
        ns.tp = TwoPhase(ns, ns.vof, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2)
    if sigma:
        ns.tp.add_surface_tension(sigma, curvature_method=curvature_method)
    if gravity is not None:
        G, Z = gravity
        ns.tp.add_reduced_gravity(G, Z)


class _TwoPhaseMixin:
    """Diagnostics shared by the two-phase steppers."""

    @property
    def psi(self):
        return self.cls.psi.interior

    def pressure(self):
        """The pressure with its jump (``p~ + H j_ext`` in the smooth form)."""
        full = getattr(self.tp, "full_pressure", None)
        return full(self) if full is not None else self.p.interior

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



class TwoPhaseIMEX(_TwoPhaseMixin, IMEXMultistepNS):
    def __init__(self, grid, phi0, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0, mu2: float = 0.0,
                 sigma: float = 0.0, gravity=None, periodic: tuple = (), faces: str = "fd2",
                 convection: str = "central", eps: float | None = None, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None, jump_mode: str | None = None,
                 capillary_dt_factor: float | None = None, transport: str = "frozen",
                 psi_integrator: str = "ssprk3", scheme="sbdf2", cls_divergence: str | None = None,
                 jump_interp: str | None = None, interface: str = "cls", vof_crossing: str = "sum",
                 cell_coefficient: str | None = None, vof_refine: int = 1, vof_curvature: str = "basilisk",
                 phase_ccd_taylor: bool | None = None, **kw) -> None:
        """``phi0``: cell-centred signed distance, positive in phase 1 (``psi = 1``). ``gravity``: ``(G, Z)`` of
        the reduced gravity (``G`` the acceleration vector). Walls on the non-periodic sides (``uf`` normal
        zero there). ``scheme``: the momentum's multistep IMEX scheme (:class:`~.schemes.MultistepScheme`,
        default ``"sbdf2"``, the paper's BDF2 + EXT2). Other keywords go to :class:`IMEXMultistepNS`
        (``fixed_dt``, ``tolerance``, ``DT``, ...).

        ``transport``: the velocity that moves ``psi`` over the step (frozen through the integrator's stages):

        * ``"frozen"`` (``"projected"``): the paper's ``u_f^n`` (first order in time);
        * ``"ab2"`` / ``"ab3"``: Adams--Bashforth mean of ``u_f^n, u_f^{n-1}, ...`` over the step (variable steps);
        * ``"pc"``: predictor--corrector (Heun on the kinematics): the step is taken with ``u_f^n``, then
          repeated from the same state with ``(u_f^n + u_f^{n+1})/2`` (twice the cost).

        ``psi_integrator``: ``"ssprk3"`` (the paper's TVD-RK3), ``"ssprk2"``, ``"rk4"``, ``"heun3"``, ``"euler"``.

        ``interface``: ``"cls"`` (the conservative level set), ``"vof"`` (Basilisk's VOF with tension.h's CSF) or
        ``"vof-gfm"`` (VOF with the sharp pressure jump, :class:`~..interface.vof_jump.TwoPhaseVOFJump`), docs/notes/64:
        its crossings from ``vof_crossing`` (``"sum"``, the line sum of f, or ``"plic"``, the PLIC line), the
        coefficient of the cell-centred correction on cut faces from ``cell_coefficient`` (``"mean"`` or ``"gfm"``,
        docs/notes/65; default ``"mean"`` for ``"vof-gfm"``, ``"gfm"`` for the CLS). ``vof_refine``: the initial
        fraction of a level function from k x k sub-cells (:func:`~..interface.fractions.fraction`, docs/notes/66).
        ``vof_curvature``: ``"basilisk"`` (tension.h) or ``"ccd"`` (:func:`~..interface.vof_curvature.ccd_curvature`).
        The VOF interfaces take ``faces="fd2"`` (or ``"phase-ccd"`` with ``"vof-gfm"``: vof.h needs a face velocity
        whose fd2 divergence vanishes, so the transport velocity is projected once more with fd2's GFM operator,
        docs/notes/67) and default to the fd2 viscous operator; ``eps``,
        ``phase_ccd_taylor``: the cut faces' Taylor terms of ``faces="phase-ccd"`` (default: with the CLS, without
        for the VOF, docs/notes/67);
        the reinitialisation and the curvature options are the CLS's."""
        if transport == "projected":
            transport = "frozen"
        if transport not in TRANSPORTS:
            raise ValueError(f"transport is one of {TRANSPORTS}")
        if psi_integrator not in EXPLICIT_RK:
            raise ValueError(f"psi_integrator is one of {sorted(EXPLICIT_RK)}")
        periodic = tuple(periodic)
        _check_faces(faces, rho1, rho2)
        kw.setdefault("uf_normal", {w: 0.0 for w in WALLS if _SIDE_AXIS[w] not in periodic})
        # the VOF viscosity jumps at the interface, where CCD's viscous operator grows parasitic currents (notes/64)
        kw.setdefault("viscous", FD2Viscosity() if interface in ("vof", "vof-gfm") else CCDViscosity())
        if phase_ccd_taylor is None:  # the VOF's ghost-fluid jump damps moving interfaces with them (notes/67)
            phase_ccd_taylor = interface == "cls"
        super().__init__(grid, scheme=scheme, periodic=periodic, ops=_face_ops(faces, phase_ccd_taylor),
                         convection=_convection(convection), **kw)
        if interface not in ("cls", "vof", "vof-gfm"):
            raise ValueError("interface is 'cls', 'vof' (tension.h's CSF) or 'vof-gfm' (sharp pressure jump)")
        self.interface = interface
        if interface in ("vof", "vof-gfm"):
            if faces == "fccd" or (faces == "phase-ccd" and interface != "vof-gfm"):
                raise NotImplementedError("the VOF interfaces take fd2 faces, or phase-ccd with 'vof-gfm' (the cut "
                                          "faces of the sharp jump close its CCD segments), docs/notes/67")
            _attach_vof(self, phi0, periodic=periodic, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, sigma=sigma,
                        gravity=gravity, jump=interface == "vof-gfm", crossing=vof_crossing,
                        cell_coefficient=cell_coefficient or "mean", refine=vof_refine,
                        curvature_method=vof_curvature)
            if faces == "phase-ccd":
                # the cut faces move with the interface: refresh the operator once the properties are set
                self.on("properties", lambda _ns: _ns.ops.update(_ns.tp, _ns.alpha), after=True)
        else:
            _attach_two_phase(self, phi0, periodic=periodic, faces=faces, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2,
                              sigma=sigma, gravity=gravity, eps=eps, reinit_every=reinit_every,
                              reinit_method=reinit_method, curvature_kw=curvature_kw, jump_mode=jump_mode,
                              capillary_dt_factor=capillary_dt_factor, cls_divergence=cls_divergence,
                              jump_interp=jump_interp, cell_coefficient=cell_coefficient)
            self.cls.integrator = psi_integrator
        self.faces, self.convection_name = faces, convection
        self.transport, self.psi_integrator = transport, psi_integrator
        self._uf_hist: list = []  # u_f^n, u_f^{n-1}, ... (after the steps taken)
        self._uf_dts: list = []   # dt_n, dt_{n-1}, ...: the steps between them (own history: the momentum
        #                           scheme keeps only as many steps as it needs, e.g. one for sbdf1)
        self._uf_saved = None
        self._pc_velocity = None
        # registered after TwoPhaseCLS's transport hook, so it runs first; the second one restores uf
        self.on("vof", self._transport_velocity)
        self.on("vof", self._restore_uf, after=True)
        self.on("end_timestep", self._remember_uf, after=True)

    # -- transport velocity --------------------------------------------------------------------------
    def _transport_velocity(self, ns) -> None:
        from ..fields import FaceField  # noqa: PLC0415

        self._uf_saved = None
        if not self.clock.dt:
            return
        nd = self.grid.ndim
        if self._pc_velocity is not None:
            faces = self._pc_velocity
        elif self.transport in ("ab2", "ab3") and self._uf_hist:
            k = min(int(self.transport[-1]), len(self._uf_hist))
            steps = [self.clock.dt] + self._uf_dts[:k - 1]
            offsets = [sum(steps[:i + 1]) / steps[0] for i in range(k)]
            w = ab_beta(k, offsets)
            faces = [sum(float(w[j]) * self._uf_hist[j][ax] for j in range(k)) for ax in range(nd)] if k > 1 else None
        else:
            faces = None
        if faces is None and not self._vof_reproject():
            return
        if faces is None:
            faces = [self.uf[ax].copy() for ax in range(nd)]
        uf_t = FaceField(self.grid, faces, "uf")
        if self._vof_reproject():
            self._fd2_project(uf_t)
        self._uf_saved = self.uf
        self.uf = uf_t

    def _vof_reproject(self) -> bool:
        return self.interface != "cls" and self.faces != "fd2"

    def _fd2_project(self, uf) -> None:
        """Project ``uf`` with fd2's GFM operator (its fd2 divergence then vanishes to the solver's tolerance), the
        transport velocity of the VOF with phase-wise CCD faces (docs/notes/67)."""
        from ..fields import CellField  # noqa: PLC0415
        from ..projection.approximate import project  # noqa: PLC0415

        if getattr(self, "_p_vof", None) is None:
            self._p_vof = CellField(self.grid, bc=self.p.bc, name="p_vof")
        self._p_vof.data[...] = 0.0
        self._apply_uf_bc(uf)
        project(uf, self._p_vof, self.alpha, 1.0, ops=None, tolerance=self.tolerance)
        self._apply_uf_bc(uf)

    def _restore_uf(self, ns) -> None:
        if self._uf_saved is not None:
            self.uf = self._uf_saved
            self._uf_saved = None

    def _remember_uf(self, ns) -> None:
        self._uf_hist = ([[self.uf[ax].copy() for ax in range(self.grid.ndim)]] + self._uf_hist)[:3]
        self._uf_dts = ([float(self.clock.dt)] + self._uf_dts)[:3]

    # -- predictor--corrector ------------------------------------------------------------------------
    def step(self) -> None:
        if self.transport != "pc":
            return super().step()
        snap = self._snapshot()
        uf_n = [self.uf[ax].copy() for ax in range(self.grid.ndim)]
        super().step()
        uf_pred = [self.uf[ax].copy() for ax in range(self.grid.ndim)]
        self._restore(snap)
        self._pc_velocity = [0.5 * (uf_n[ax] + uf_pred[ax]) for ax in range(self.grid.ndim)]
        try:
            super().step()
        finally:
            self._pc_velocity = None

    def _snapshot(self) -> dict:
        import copy  # noqa: PLC0415

        nd = self.grid.ndim

        def faces(f):
            return None if f is None else [f[ax].copy() for ax in range(nd)]

        return {
            "cells": [c.data.copy() for c in (*self.u, *self.g, self.p, self.pf, self.dp, self.cls.psi)],
            "uf": faces(self.uf), "alpha": faces(self.alpha), "mu": faces(self.mu), "a": faces(self.a),
            "a_is_constant": self.a_is_constant,
            "rho": self.rho.copy() if hasattr(self.rho, "copy") else self.rho,
            "clock": copy.deepcopy(self.clock), "dtmax": self.dtmax,
            "lists": {k: list(getattr(self, k)) for k in ("_u_hist", "_e_hist", "_v_hist", "_dts", "_uf_hist", "_uf_dts", "_a_hist")},
            "prev": (self._u_prev, self._c_prev, self._dt_prev),
            "tp": (getattr(self.tp, "_steps", None), getattr(self.tp, "j_ext", None)),
        }

    def _restore(self, s: dict) -> None:
        from ..fields import FaceField  # noqa: PLC0415

        nd = self.grid.ndim
        for c, d in zip((*self.u, *self.g, self.p, self.pf, self.dp, self.cls.psi), s["cells"]):
            c.data[...] = d
        self.uf = FaceField(self.grid, [a.copy() for a in s["uf"]], "uf")
        for name in ("alpha", "mu", "a"):
            if s[name] is not None:
                f = getattr(self, name)
                for ax in range(nd):
                    f[ax] = s[name][ax].copy()
        self.a_is_constant = s["a_is_constant"]
        self.rho = s["rho"]
        self.clock = s["clock"]
        self.dtmax = s["dtmax"]
        for k, v in s["lists"].items():
            setattr(self, k, list(v))
        self._u_prev, self._c_prev, self._dt_prev = s["prev"]
        if s["tp"][0] is not None:
            self.tp._steps, self.tp.j_ext = s["tp"]

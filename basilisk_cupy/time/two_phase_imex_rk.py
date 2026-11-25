"""Two-phase IMEX Runge--Kutta stepper: the interface advances inside the stages (docs/notes/52).

:class:`TwoPhaseIMEXRK` is :class:`IMEXRKNS` (ARS-type tableaux: an explicit first stage, one implicit
coefficient ``gamma``) with the CLS interface and the GFM pressure jump of :class:`TwoPhaseIMEX`. Unlike
the multistep stepper, which moves ``psi`` once per step with a frozen velocity, ``psi`` is one more explicit
component of the same tableau. Stage ``i`` (``i >= 1``; stage 0 is ``u^n``, ``psi^n``, ``u_f^n``)::

    psi_i = clip(psi^n + dt sum_{j<i} aE_ij R(psi_j, u_f,j)),   R = -div(psi u_f) (CLS's flux, FCCD values)
    rho, alpha, mu from psi_i                                     ("properties" hooks)
    (I - gamma dt V) u*_i = u^n + c_i dt g^n + dt sum_{j<i} (aE_ij C_j + aI_ij V_j)
    jump acceleration at the psi_i interface                      ("acceleration" hooks)
    IPC projection with the step c_i dt                           (as IMEXRKNS)

then ``psi^{n+1} = psi^n + dt sum_j bE_j R_j`` (the last stage when the explicit part is stiffly accurate)
with the CLS mass closure, and ``u^{n+1}`` the last stage, or ``u^n + dt sum_j (bE_j C_j + bI_j V_j)``
projected over ``p^n`` when a part is not stiffly accurate. Re-initialisation as in :class:`TwoPhaseCLS`
(every ``reinit_every`` steps, after the step).
"""

from __future__ import annotations

from ..interface.cls import mass_correction
from ..terms.viscous import CCDViscosity
from .imex_rk import IMEXRKNS
from .two_phase_imex import _SIDE_AXIS, WALLS, _attach_two_phase, _check_faces, _convection, _face_ops, _TwoPhaseMixin


class TwoPhaseIMEXRK(_TwoPhaseMixin, IMEXRKNS):
    def __init__(self, grid, phi0, *, tableau: str = "ars222", rho1: float = 1.0, rho2: float = 1.0,
                 mu1: float = 0.0, mu2: float = 0.0, sigma: float = 0.0, gravity=None, periodic: tuple = (),
                 faces: str = "fd2", convection: str = "central", eps: float | None = None, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None, jump_mode: str | None = None,
                 capillary_dt_factor: float | None = None, **kw) -> None:
        """Arguments as :class:`TwoPhaseIMEX`; ``tableau``: ``ars111``, ``ars121``, ``ars222`` (default),
        ``ars233``, ``ars343``, ``ars443`` (:data:`~.imex_rk.TABLEAUX`)."""
        if kw.get("face_native"):
            raise NotImplementedError("TwoPhaseIMEXRK has the cell-state form only")
        periodic = tuple(periodic)
        _check_faces(faces, rho1, rho2)
        kw.setdefault("uf_normal", {w: 0.0 for w in WALLS if _SIDE_AXIS[w] not in periodic})
        kw.setdefault("viscous", CCDViscosity())
        super().__init__(grid, tableau=tableau, periodic=periodic,
                         ops=_face_ops(faces), convection=_convection(convection), **kw)
        _attach_two_phase(self, phi0, periodic=periodic, faces=faces, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2,
                          sigma=sigma, gravity=gravity, eps=eps, reinit_every=reinit_every,
                          reinit_method=reinit_method, curvature_kw=curvature_kw, jump_mode=jump_mode,
                          capillary_dt_factor=capillary_dt_factor)
        self.faces, self.convection_name = faces, convection

    def _set_psi(self, q) -> None:
        xp = self.grid.xp
        self.cls.psi.interior = xp.clip(q, 0.0, 1.0)
        self.cls.psi.fill_ghosts()
        self._run_hooks("properties")

    def _new_acceleration(self) -> None:
        if not self.a_is_constant:
            for ax in range(self.grid.ndim):
                self.a[ax] = 0.0
        self._run_hooks("acceleration")

    def step(self) -> None:
        c = self.clock
        self._run_hooks("set_dtmax")
        self.dtmax = self.DT
        self._run_hooks("stability")
        self._stability()
        dt = c.dt
        # no "vof" stage: TwoPhaseCLS would move psi once with a frozen velocity; here psi follows the stages
        for stage in ("tracer_advection", "tracer_diffusion"):
            self._run_hooks(stage)

        t = self.tableau
        nd = self.grid.ndim
        xp = self.grid.xp
        cls = self.cls
        psi_n = cls.psi.interior.copy()
        m0 = xp.sum(psi_n * cls.dv)
        u_n = [comp.interior.copy() for comp in self.u]
        g_n = [comp.interior.copy() for comp in self.g]
        p_n = self.p.interior.copy()
        zero = [0.0] * nd
        gdt = t.gamma * dt
        last = t.stages - 1
        need_last = not (t.explicit_stiffly_accurate and t.implicit_stiffly_accurate)

        self._run_hooks("properties")
        self._run_hooks("advection_term")
        C = [zero if self.stokes else self.convection.tendency(self.u, self.uf)]
        R = [cls._rhs(psi_n, self.uf)]
        V = [zero]
        implicit = self.mu is not None
        for i in range(1, t.stages):
            q = psi_n
            for j in range(i):
                if t.AE[i][j]:
                    q = q + dt * t.AE[i][j] * R[j]
            self._set_psi(q)
            rhs = []
            for ax in range(nd):
                r = u_n[ax] + t.c[i] * dt * g_n[ax]
                for j in range(i):
                    if t.AE[i][j]:
                        r = r + dt * t.AE[i][j] * C[j][ax]
                    if t.AI[i][j]:
                        r = r + dt * t.AI[i][j] * V[j][ax]
                rhs.append(r)
                self.u[ax].interior = r
            self.u.fill_ghosts()
            self._run_hooks("viscous_term")
            self._implicit_solve(gdt, None)
            ustar = [comp.interior.copy() for comp in self.u]
            V.append([(us - r) / gdt for us, r in zip(ustar, rhs)] if implicit else zero)
            self._new_acceleration()
            self._run_hooks("projection")
            self._project_stage(ustar, g_n, p_n, t.c[i] * dt)
            if i < last or need_last:
                C.append(zero if self.stokes else self.convection.tendency(self.u, self.uf))
                R.append(cls._rhs(cls.psi.interior, self.uf))

        if not t.explicit_stiffly_accurate:
            q = psi_n
            for j in range(t.stages):
                if t.bE[j]:
                    q = q + dt * t.bE[j] * R[j]
            q = xp.clip(q, 0.0, 1.0)
        else:
            q = cls.psi.interior
        if cls.mass_closure:
            q = mass_correction(q, cls.dv, m0)
        self._set_psi(q)
        if need_last:
            self._new_acceleration()
            bI = t.implicit_weights
            final = []
            for ax in range(nd):
                v = u_n[ax] + dt * g_n[ax]
                for j in range(t.stages):
                    if t.bE[j] and not self.stokes:
                        v = v + dt * t.bE[j] * C[j][ax]
                    if bI[j]:
                        v = v + dt * bI[j] * V[j][ax]
                final.append(v)
            self._project_stage(final, g_n, p_n, dt)

        tp = self.tp
        tp._steps += 1
        if tp.reinit_every and tp._steps % tp.reinit_every == 0:
            cls.reinitialize(method=tp.reinit_method)
            self._run_hooks("properties")
        if not self.a_is_constant:
            for ax in range(nd):
                self.a[ax] = 0.0
        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

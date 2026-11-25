"""IMEX linear multistep Navier--Stokes stepper with incremental pressure correction (any :class:`MultistepScheme`).

For the scheme ``sum_j alpha_j u^{n+1-j} = dt [sum_j beta_j E^{n-j} + sum_j gamma_j V^{n+1-j}]``
(``E`` the convection tendency, ``V`` the viscous one, docs/notes/51), divided by ``alpha_0``::

    (I - tau V) u* = sum_{j>=1} (-alpha_j/alpha_0) u^{n+1-j}
                     + sigma ( sum_j beta_j E^{n-j} + sum_{j>=1} gamma_j V^{n+1-j} + g^n ),
    sigma = dt / alpha_0,   tau = sigma gamma_0,

followed by the incremental projection of :class:`IMEXBDF2NS` with the projection step ``sigma`` (the
pressure is the Lagrange multiplier of the implicit level): ``u_f* = P_f(u* - sigma g^n) + sigma (a -
alpha G_f p^n)``, ``D_f alpha G_f dp = D_f u_f*/sigma``, ``u^{n+1} = u* + sigma (g^{n+1} - g^n)``.
``sbdf2`` reproduces :class:`IMEXBDF2NS` (same coefficients, variable steps included).

The past viscous tendencies needed by Adams--Moulton / theta / MCNAB2 / CNLF are those of the solved
predictors, ``V^{k} = (u*_k - rhs_k)/tau_k`` (as :class:`IMEXRKNS` does for its stages): the viscous
operator applied to ``u*``, not to the projected ``u^{k}``.

The explicit-convection stability bound of :class:`IMEXBDF2NS` (EXT2's real-axis limit) is kept for every
explicit family; other families have other limits (not applied).
"""

from __future__ import annotations

from ..fields import FaceField
from ..projection.approximate import project
from .imex_bdf2 import IMEXBDF2NS
from .schemes import MultistepScheme


class IMEXMultistepNS(IMEXBDF2NS):
    def __init__(self, grid, *, scheme="sbdf2", **kwargs) -> None:
        if kwargs.get("face_native"):
            raise NotImplementedError("IMEXMultistepNS has the cell-state form only (face_native=False)")
        super().__init__(grid, **kwargs)
        self.scheme = scheme if isinstance(scheme, MultistepScheme) else MultistepScheme(scheme)
        self._u_hist: list = []   # u^n, u^{n-1}, ... (interior arrays, before this step)
        self._e_hist: list = []   # E^n, E^{n-1}, ...
        self._v_hist: list = []   # V^n, V^{n-1}, ...
        self._dts: list = []      # dt_{n}, dt_{n-1}, ... of the steps already taken (newest first)
        self.last_coefficients = None

    def step(self) -> None:
        c = self.clock
        self._run_hooks("set_dtmax")
        self.dtmax = self.DT
        self._run_hooks("stability")
        self._stability()
        dt = c.dt
        for stage in ("vof", "tracer_advection", "tracer_diffusion", "properties"):
            self._run_hooks(stage)

        ndim = self.grid.ndim
        depth = self.scheme.max_depth
        steps = [dt] + self._dts[:depth - 1]
        coef = self.scheme.coefficients(steps, v_levels=len(self._v_hist))
        self.last_coefficients = coef
        a0 = coef.alpha[0]
        sigma = dt / a0
        tau = sigma * coef.gamma[0]

        extra = self._implicit_term()
        self._run_hooks("advection_term")
        conv_n = [0.0] * ndim if self.stokes else self.convection.tendency(self.u, self.uf)
        e_levels = [conv_n] + self._e_hist
        u_n = [comp.interior.copy() for comp in self.u]
        u_levels = [u_n] + self._u_hist
        g_n = [comp.interior.copy() for comp in self.g]
        rhs = []
        for ax in range(ndim):
            r = sigma * g_n[ax]
            for j, a in enumerate(coef.alpha[1:]):
                if a:
                    r = r - (a / a0) * u_levels[j][ax]
            for j, b in enumerate(coef.beta):
                if b and not self.stokes:
                    r = r + sigma * b * e_levels[j][ax]
            for j, gm in enumerate(coef.gamma[1:]):
                if gm and self._v_hist:
                    r = r + sigma * gm * self._v_hist[j][ax]
            rhs.append(r)
            self.u[ax].interior = r
        self.u.fill_ghosts()

        self._run_hooks("viscous_term")
        self._implicit_solve(tau, extra)
        v_new = None
        if self.mu is not None:
            v_new = [(self.u[ax].interior - rhs[ax]) / tau for ax in range(ndim)]
        if not self.a_is_constant:
            for ax in range(ndim):
                self.a[ax] = 0.0

        self._run_hooks("acceleration")
        for ax in range(ndim):
            self.u[ax].interior = self.u[ax].interior - sigma * g_n[ax]
        self.u.fill_ghosts()
        self.uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) + sigma * self._face_pressure_force(self.p, ax)
                                        for ax in range(ndim)], "uf")
        self._apply_uf_bc(self.uf)

        self._run_hooks("projection")
        self.dp.data[...] = 0.0
        self.mgp = project(self.uf, self.dp, self.alpha, sigma, ops=self.ops,
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance)
        self._apply_uf_bc(self.uf)
        self.p.interior = self.p.interior + self.dp.interior
        self.p.fill_ghosts()
        self.centered_gradient()
        self.correction(sigma)

        keep = max(depth - 1, 0)
        self._u_hist = ([u_n] + self._u_hist)[:keep]
        self._e_hist = ([conv_n] + self._e_hist)[:keep]
        self._v_hist = ([v_new] + self._v_hist)[:depth] if v_new is not None else []
        self._dts = ([dt] + self._dts)[:max(depth, 1)]
        # IMEXBDF2NS's bookkeeping (CFL limit, warnings) reads these
        self._u_prev, self._c_prev, self._dt_prev = u_n, conv_n, dt
        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

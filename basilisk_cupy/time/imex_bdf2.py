"""IMEX-BDF2 with incremental pressure correction (TwoPhaseFlow chapter 11, single-phase path).

Predictor (``n >= 1``; the first step is BDF1 with ``gamma = 1``)::

    (I - gamma dt V) u* = 4/3 u^n - 1/3 u^{n-1}
                          + gamma dt ( -C^EXT2 + g^n ),   C^EXT2 = 2 C(u^n) - C(u^{n-1})

with ``g^n`` the cell-centred ``a - alpha G p^n`` (as Basilisk's ``g``) and
``gamma = 2/3``. Projection (IPC): the face velocity carries the face
pressure gradient of ``p^n``, the Poisson problem is solved for the
increment ``dp`` with the projection step ``gamma dt``::

    u_f* = P_f(u* - gamma dt g^n) + gamma dt (a - alpha G_f p^n)
    D_f alpha G_f dp = D_f u_f* / (gamma dt),   u_f = u_f* - gamma dt alpha G_f dp
    p^{n+1} = p^n + dp,   u^{n+1} = u* + gamma dt (g^{n+1} - g^n)

Variable steps use the standard variable-step BDF2/EXT2 coefficients
(``w = dt_n / dt_{n-1}``); with a constant step they reduce to the above.

Measured temporal error: ``O(dt^2) + O(dt h^p)``, ``p = 2`` with fd2 and 4 with
FCCD. The cross term comes from the approximate projection (the cell
velocity is not discretely divergence-free); it vanishes for Stokes flow,
where the scheme is exactly 2nd order. TwoPhaseFlow avoids it by carrying
the projected face velocity as the state, which is a later step here.

Convection terms with an explicit stability bound (UCCD6, FUCCD:
``max_stable_dt``) cap the adaptive step; a fixed step above the bound warns.

Basilisk's own scheme (:class:`CenteredNS`) is kept as the reference. With
the same operators the incremental and the full-pressure projection are
algebraically identical (Basilisk starts its Poisson iteration from p^n), so
the difference between the two steppers is the time integration.
"""

from __future__ import annotations

from ..fields import CellField, FaceField
from ..linear.viscosity import viscosity
from ..projection.approximate import project
from ..terms.convection.central import CentralFluxConvection
from .basilisk_centered import CenteredNS
from .stepping import dtnext


class IMEXBDF2NS(CenteredNS):
    """Same fields, boundary conditions and hook stages as :class:`CenteredNS`."""

    def __init__(self, grid, *, convection=None, fixed_dt: float | None = None, **kwargs) -> None:
        super().__init__(grid, **kwargs)
        self.convection = convection or CentralFluxConvection(self.ops)
        #: Use a constant step instead of Basilisk's adaptive one (``cfl_number`` reports the CFL).
        self.fixed_dt = fixed_dt
        self.cfl_number = 0.0
        self.dp = CellField(grid, bc=self.p.bc, name="dp")
        self._u_prev = None      # u^{n-1}
        self._c_prev = None      # -C(u^{n-1})
        self._dt_prev = None
        self._warned_dt = False

    # -- helpers ---------------------------------------------------------------------
    def _convection_dt_limit(self) -> float:
        """Stability bound of the explicit convection term (``max_stable_dt``, if it has one)."""
        limit = getattr(self.convection, "max_stable_dt", None)
        if self.stokes or limit is None:
            return float("inf")
        return limit(self.u, self.uf)

    def _stability(self) -> None:
        limit = self._convection_dt_limit()
        if self.fixed_dt is None:
            self.dtmax = min(self.dtmax, limit)
            super()._stability()
            return
        if self.fixed_dt > limit and not self._warned_dt:
            import warnings  # noqa: PLC0415

            warnings.warn(f"fixed_dt {self.fixed_dt:g} exceeds the explicit stability bound {limit:g} of "
                          f"the {getattr(self.convection, 'name', 'convection')} term", RuntimeWarning, stacklevel=3)
            self._warned_dt = True
        self.clock.dt = dtnext(self.fixed_dt, self.clock)
        # explicit convection: record the CFL number of the step taken
        xp = self.grid.xp
        self.cfl_number = max(float(xp.max(xp.abs(self.uf[ax]))) * self.clock.dt / self.grid.h[ax]
                              for ax in range(self.grid.ndim))

    def _face_pressure_force(self, p: CellField, ax: int):
        p.fill_ghosts()
        return self.a[ax] - self.alpha[ax] * self.ops.grad_f(p, ax)

    # -- one step ----------------------------------------------------------------------
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
        bdf2 = self._u_prev is not None
        if bdf2:
            w = dt / self._dt_prev
            gamma = (1.0 + w) / (1.0 + 2.0 * w)
            # (1+2w)/(1+w) u* - (1+w) u^n + w^2/(1+w) u^{n-1} = dt RHS, divided by (1+2w)/(1+w)
            c_n, c_m = gamma * (1.0 + w), gamma * w * w / (1.0 + w)
        else:
            w, gamma, c_n, c_m = 0.0, 1.0, 1.0, 0.0

        # explicit convection at t^n (extrapolated)
        self._run_hooks("advection_term")
        conv_n = [0.0] * ndim if self.stokes else self.convection.tendency(self.u, self.uf)
        if bdf2 and not self.stokes:
            conv = [(1.0 + w) * cn - w * cp for cn, cp in zip(conv_n, self._c_prev)]
        else:
            conv = conv_n

        u_n = [comp.interior.copy() for comp in self.u]
        g_n = [comp.interior.copy() for comp in self.g]
        gdt = gamma * dt
        for ax in range(ndim):
            rhs = c_n * u_n[ax] + gdt * (conv[ax] + g_n[ax])
            if bdf2:
                rhs = rhs - c_m * self._u_prev[ax]
            self.u[ax].interior = rhs
        self.u.fill_ghosts()

        self._run_hooks("viscous_term")
        if self.mu is not None:
            self.mgu = viscosity(self.u, self.mu, self.rho, gdt, nrelax=self.mgu.nrelax or 4,
                                 tolerance=self.tolerance)
        if not self.a_is_constant:
            for ax in range(ndim):
                self.a[ax] = 0.0

        # IPC: remove the cell pressure gradient, add the face one, project the increment
        self._run_hooks("acceleration")
        for ax in range(ndim):
            self.u[ax].interior = self.u[ax].interior - gdt * g_n[ax]
        self.u.fill_ghosts()
        self.uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) + gdt * self._face_pressure_force(self.p, ax)
                                        for ax in range(ndim)], "uf")
        self._apply_uf_bc(self.uf)

        self._run_hooks("projection")
        self.dp.data[...] = 0.0
        self.mgp = project(self.uf, self.dp, self.alpha, gdt, ops=self.ops,
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance)
        self._apply_uf_bc(self.uf)
        self.p.interior = self.p.interior + self.dp.interior
        self.p.fill_ghosts()
        self.centered_gradient()
        self.correction(gdt)  # u^{n+1} = u* - gdt g^n + gdt g^{n+1}

        self._u_prev, self._c_prev, self._dt_prev = u_n, conv_n, dt
        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

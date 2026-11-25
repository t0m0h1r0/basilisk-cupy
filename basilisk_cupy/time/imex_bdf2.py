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

Face-native state (``face_native=True``; TwoPhaseFlow paper §11b2: the state
is the face velocity ``f`` with ``D_f f = 0`` and ``u = R_h f``): the projected
face velocity is carried as the state and the cell velocity is always its
reconstruction ``R_h`` (``face_to_cell`` of the shared operators), used only to
evaluate convection and viscosity. The predictor is still solved on cells, but
only its increment moves the faces, ``u_f* = A_f + P_f(u* - R_h A_f - gamma dt g^n)
+ gamma dt (a - alpha G_f p^n)`` with ``A_f = c_n u_f^n - c_m u_f^{n-1}``, so the
non-invertible ``R_h``/``P_f`` round trip never acts on the state itself. The
cell correction ``u* + gamma dt (g^{n+1} - g^n)`` and its ``O(dt h^p)`` error
disappear; the cell values carry the error of ``R_h`` instead: 4th order with
FCCD's ``face_to_cell`` (2nd with fd2), or the order given by ``reconstruction``
(``FaceToCell``: 6 or 8). (TwoPhaseFlow's code has the option behind
``face_native_predictor_state``, off by default; §11b1 stage 7 still writes the
nodal correction.)

Convection terms with an explicit stability bound (UCCD6, FUCCD:
``max_stable_dt``) cap the adaptive step; a fixed step above the bound warns.

Basilisk's own scheme (:class:`CenteredNS`) is kept as the reference. With
the same operators the incremental and the full-pressure projection are
algebraically identical (Basilisk starts its Poisson iteration from p^n), so
the difference between the two steppers is the time integration.
"""

from __future__ import annotations

from ..fields import CellField, FaceField
from ..projection.approximate import project
from ..terms.convection.central import CentralFluxConvection
from ..terms.viscous import BasiliskViscosity
from .basilisk_centered import CenteredNS
from .stepping import dtnext


class IMEXBDF2NS(CenteredNS):
    """Same fields, boundary conditions and hook stages as :class:`CenteredNS`."""

    basilisk_init_dt = False  # fixed or CFL steps from the first step on

    def __init__(self, grid, *, convection=None, viscous=None, fixed_dt: float | None = None,
                 face_native: bool = False, reconstruction=None, volume_tolerance: bool = False, **kwargs) -> None:
        super().__init__(grid, **kwargs)
        #: ``tolerance`` bounds the relative volume change of a cell in one projection step, ``|div u| sigma``
        #: (Basilisk's ``project()``: the Poisson problem is solved to ``tolerance / sigma^2``), instead of the
        #: residual ``div u / sigma`` itself (default). Needed for dimensional problems, where the absolute
        #: residual of ``div u / sigma`` can be orders of magnitude above 1.
        self.volume_tolerance = volume_tolerance
        #: carry the projected face velocity as the state, cells reconstructed from it (see module doc)
        self.face_native = face_native
        #: R_h of the face-native state: None (the operators' face_to_cell), "ccd" (compact face-line CCD
        #: + Hermite, 6th order), an even order (Lagrange stencil, discretization/reconstruction.py) or a
        #: callable ``(faces, axis, periodic=) -> cells``
        if reconstruction == "ccd":
            from ..discretization.reconstruction import CompactFaceToCell  # noqa: PLC0415

            reconstruction = CompactFaceToCell()
        elif isinstance(reconstruction, int):
            from ..discretization.reconstruction import FaceToCell  # noqa: PLC0415

            reconstruction = FaceToCell(reconstruction)
        self.reconstruction = reconstruction
        self._uf_prev = None
        self.convection = convection or CentralFluxConvection(self.ops)
        #: implicit viscous solver (``solve(u, mu, rho, dt)``): Basilisk's fd2 multigrid or ``CCDViscosity``
        self.viscous = viscous or BasiliskViscosity()
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

    # -- implicit part: viscosity and, optionally, the convection term's implicit part ----------
    def _implicit_term(self):
        """The convection term's implicit part frozen at ``u^n`` (UCCD6 ``implicit=True``), or None."""
        if self.stokes or not getattr(self.convection, "implicit", False):
            return None
        return self.convection.implicit_part(self.u)

    def _implicit_solve(self, tau: float, extra) -> None:
        """``(I - tau/rho V + tau E) u = u`` in place (viscosity and the implicit extra term)."""
        if self.mu is None and extra is None:
            return
        from ..terms.viscous import DCViscosity  # noqa: PLC0415

        solver = self.viscous
        if extra is not None and not isinstance(solver, DCViscosity):
            if self.mu is not None:
                raise ValueError("an implicit convection part needs a defect-correction viscous solver "
                                 "(CCDViscosity / FD2Viscosity), not Basilisk's viscosity()")
            from ..terms.viscous import CCDViscosity  # noqa: PLC0415

            solver = self._extra_solver = getattr(self, "_extra_solver", None) or CCDViscosity()
        kwargs = {"extra": extra} if extra is not None else {}
        self.mgu = solver.solve(self.u, self.mu, self.rho, tau, nrelax=self.mgu.nrelax or 4,
                                tolerance=self.tolerance, **kwargs)

    # -- face-native state helpers ------------------------------------------------------
    def _cells_from_faces(self, faces) -> list:
        """``R_h``: cell velocity from face-normal velocities (``face_to_cell`` of the shared operators)."""
        rec = self.reconstruction or self.ops.face_to_cell
        return [rec(faces[ax], ax, periodic=self.u[ax].bc.is_periodic(ax)) for ax in range(self.grid.ndim)]

    def _reconstruct_cells(self) -> None:
        for ax, v in enumerate(self._cells_from_faces(self.uf)):
            self.u[ax].interior = v
        self.u.fill_ghosts()

    def _faces_from_increment(self, inc: list) -> list:
        """``P_f`` of a cell-velocity increment (homogeneous conditions: the increment of a fixed BC)."""
        out = []
        for ax in range(self.grid.ndim):
            f = CellField.from_interior(self.grid, inc[ax], self.u[ax].bc.homogeneous())
            f.fill_ghosts()
            out.append(self.ops.interp_f(f, ax))
        return out

    def _project_faces(self, uf_star: list, p_base, cdt: float) -> None:
        """Project ``uf*`` for the increment over ``p_base``; cells are then ``R_h uf``."""
        self.uf = FaceField(self.grid, uf_star, "uf")
        self._apply_uf_bc(self.uf)
        self.dp.data[...] = 0.0
        self.mgp = project(self.uf, self.dp, self.alpha, cdt, ops=self.ops,
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance,
                           basilisk_tolerance=self.volume_tolerance)
        self._apply_uf_bc(self.uf)
        self.p.interior = p_base + self.dp.interior
        self.p.fill_ghosts()
        self.centered_gradient()
        self._reconstruct_cells()

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

        if self.face_native and self._uf_prev is None and not getattr(self, "_fn_started", False):
            self._reconstruct_cells()  # the state is uf; cells are its reconstruction from the first step on
            self._fn_started = True
        extra = self._implicit_term()  # frozen at u^n
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
        if self.face_native:
            uf_n = [self.uf[ax].copy() for ax in range(ndim)]
            a_face = [c_n * uf_n[ax] - (c_m * self._uf_prev[ax] if bdf2 else 0.0) for ax in range(ndim)]
            a_cell = self._cells_from_faces(a_face)
        for ax in range(ndim):
            if self.face_native:
                rhs = a_cell[ax] + gdt * (conv[ax] + g_n[ax])
            else:
                rhs = c_n * u_n[ax] + gdt * (conv[ax] + g_n[ax])
                if bdf2:
                    rhs = rhs - c_m * self._u_prev[ax]
            self.u[ax].interior = rhs
        self.u.fill_ghosts()

        self._run_hooks("viscous_term")
        self._implicit_solve(gdt, extra)
        if not self.a_is_constant:
            for ax in range(ndim):
                self.a[ax] = 0.0

        if self.face_native:
            # faces advance by the interpolated cell increment (no R_h/P_f round trip of the state)
            self._run_hooks("acceleration")
            inc = [self.u[ax].interior - a_cell[ax] - gdt * g_n[ax] for ax in range(ndim)]
            finc = self._faces_from_increment(inc)
            self._run_hooks("projection")
            self._project_faces([a_face[ax] + finc[ax] + gdt * self._face_pressure_force(self.p, ax)
                                 for ax in range(ndim)], self.p.interior.copy(), gdt)
            self._u_prev, self._c_prev, self._dt_prev = u_n, conv_n, dt
            self._uf_prev = uf_n
            self._run_hooks("end_timestep")
            c.t = c.tnext
            c.i += 1
            return

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
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance,
                           basilisk_tolerance=self.volume_tolerance)
        self._apply_uf_bc(self.uf)
        self.p.interior = self.p.interior + self.dp.interior
        self.p.fill_ghosts()
        self.centered_gradient()
        self.correction(gdt)  # u^{n+1} = u* - gdt g^n + gdt g^{n+1}

        self._u_prev, self._c_prev, self._dt_prev = u_n, conv_n, dt
        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

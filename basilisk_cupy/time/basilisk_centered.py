"""Incompressible variable-density Navier--Stokes, centred formulation
(port of Basilisk ``navier-stokes/centered.h``).

The time step runs Basilisk's events in the same order::

    set_dtmax, stability, vof, tracer_advection, tracer_diffusion,
    properties, advection_term, viscous_term, acceleration, projection,
    end_timestep

Other modules (two-phase, surface tension, gravity, tracers, diagnostics)
extend a step by registering hooks on these stage names with
:meth:`CenteredNS.on`. As with Basilisk's event overloading, hooks run
*before* the solver's own action for that stage, in reverse order of
registration (the most recently added module first).

Variables (as in Basilisk): centred ``u``, ``p``, ``g`` (pressure gradient
plus acceleration, centred), face ``uf``, ``pf`` (pressure of the
half-step projection), face ``alpha`` = 1/rho, face ``mu``, face
acceleration ``a``, cell ``rho``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from ..boundary import BoundaryConditions, Neumann, Periodic, vector_bcs
from ..discretization.base import FaceComplex
from ..discretization.fd2 import FD2FaceComplex
from ..fields import CellField, FaceField, VectorField
from ..grid.cartesian import CartesianGrid
from ..linear.multigrid import TOLERANCE, MGStats
from ..linear.viscosity import viscosity
from ..projection.approximate import project
from ..terms.convection.bcg import advection, prediction
from .stepping import TEPS, Clock, dtnext, timestep

STAGES = ("set_dtmax", "stability", "vof", "tracer_advection", "tracer_diffusion", "properties",
          "advection_term", "viscous_term", "acceleration", "projection", "end_timestep")

Hook = Callable[["CenteredNS"], None]


class CenteredNS:
    #: Reproduce Basilisk's first step (init's ``dtnext`` without event times, see ``start``).
    basilisk_init_dt = True

    def __init__(self, grid: CartesianGrid, *, periodic: tuple[int, ...] = (),
                 normal: dict | None = None, tangential: dict | None = None,
                 uf_normal: dict | None = None,
                 mu: float | FaceField | None = None,
                 cfl: float = 0.8, DT: float = 1e30, stokes: bool = False,
                 ops: FaceComplex | None = None, tolerance: float = TOLERANCE, viscous=None,
                 coupled_iterations: int = 0, coupled_tolerance: float | None = None,
                 coupled_criterion: str = "change", coupled_on_failure: str = "warn",
                 coupled_predictor: str | None = None, coupled_anderson: int = 0,
                 coupled_rtol: float | None = None) -> None:
        self.grid = grid
        self.ops = ops or FD2FaceComplex()
        self.periodic = tuple(periodic)
        self.cfl, self.DT, self.stokes, self.tolerance = cfl, DT, stokes, tolerance
        #: Implicit viscous solver with a ``solve(u, mu, rho, dt, tolerance=)`` method (e.g.
        #: :class:`~basilisk_cupy.terms.viscous.CCDViscosity`); ``None``: Basilisk's ``viscosity()``.
        self.viscous = viscous
        #: 0: Basilisk's splitting (the predictor damps the previous step's g with the viscosity and removes
        #: it; the new acceleration enters with the projection). K >= 1: the new acceleration is computed
        #: first and K outer iterations solve the viscosity on ``u + dt g`` with ``g`` from the new
        #: acceleration and the latest pressure, then project (docs/notes/34).
        self.coupled_iterations = int(coupled_iterations)
        #: With K >= 1: stop the outer iterations once the velocity change they make, ``dt max |g_{k+1} -
        #: g_k|`` (from the second iteration on), falls below this (``coupled_iterations`` is then the
        #: maximum); ``None``: always K.
        self.coupled_tolerance = coupled_tolerance
        #: What ``coupled_tolerance`` bounds: ``"change"``, ``dt max |g_{k+1} - g_k|`` (a fixed-point
        #: increment), or ``"residual"``, the momentum residual ``max |u_adv + dt g - (I - dt/rho L) u|``
        #: of the state the iteration would end with (Basilisk ``viscosity()`` only; docs/notes/36).
        if coupled_criterion not in ("change", "residual"):
            raise ValueError("coupled_criterion is 'change' or 'residual'")
        self.coupled_criterion = coupled_criterion
        #: Relative criterion: stop when the measure falls below ``coupled_rtol`` times its value after the
        #: first iteration (dimensionless; with ``coupled_tolerance`` too, the larger bound applies).
        self.coupled_rtol = coupled_rtol
        #: ``coupled_iterations`` reached without meeting ``coupled_tolerance``: ``"warn"``, ``"raise"``
        #: or ``"ignore"``; ``coupled_failures`` counts such steps, ``coupled_info`` describes the last step.
        if coupled_on_failure not in ("warn", "raise", "ignore"):
            raise ValueError("coupled_on_failure is 'warn', 'raise' or 'ignore'")
        self.coupled_on_failure = coupled_on_failure
        self.coupled_failures = 0
        self.coupled_info: dict = {}
        #: Initial pressure of the outer iterations: ``None``, the previous step's; ``"linear"``,
        #: ``2 p^n - p^{n-1}`` (with ``TwoPhaseCLS(jump_mode="smooth")`` ``p`` is the pressure without
        #: its jump, which the acceleration rebuilds from the moved interface).
        if coupled_predictor not in (None, "linear"):
            raise ValueError("coupled_predictor is None or 'linear'")
        self.coupled_predictor = coupled_predictor
        self._p_prev = None
        #: Anderson acceleration of the pressure iterates with this many previous residuals (0: plain
        #: fixed-point iteration). For an affine iteration it is equivalent to GMRES on ``(I - T) p = c``.
        self.coupled_anderson = int(coupled_anderson)
        self.clock = Clock()

        ubcs = vector_bcs(grid.ndim, self.periodic, normal, tangential)
        self.u = VectorField.zeros(grid, ubcs, name="u")
        self.g = VectorField.zeros(grid, vector_bcs(grid.ndim, self.periodic), name="g")
        pbc = BoundaryConditions()
        for ax in range(grid.ndim):
            for side in (0, 1):
                # Basilisk: p[...] = neumann(a.n * fm / alpha.n); zero while a = 0 on walls.
                pbc.set((ax, side), Periodic() if ax in self.periodic else Neumann(0.0))
        self.p = CellField(grid, bc=pbc, name="p")
        self.pf = CellField(grid, bc=pbc, name="pf")
        self.uf = FaceField(grid, name="uf")
        #: Optional values imposed on boundary faces of uf (Basilisk ``uf.n[left] = 0``).
        self.uf_normal = dict(uf_normal or {})

        self.alpha = FaceField.constant(grid, 1.0, name="alpha")
        self.rho = 1.0
        if mu is None or (not isinstance(mu, FaceField) and float(mu) == 0.0):
            self.mu = None
        elif isinstance(mu, FaceField):
            self.mu = mu
        else:
            self.mu = FaceField.constant(grid, float(mu), name="mu")
        self.a = FaceField(grid, name="a")
        self.a_is_constant = True  # becomes False once a hook writes into a

        self.mgp = self.mgpf = self.mgu = MGStats()
        self.dtmax = DT
        self._hooks: dict[str, list[Hook]] = defaultdict(list)

    # -- extension points ------------------------------------------------------
    def on(self, stage: str, hook: Hook, *, after: bool = False) -> None:
        """Run ``hook`` at ``stage``: before the hooks registered earlier (Basilisk: a later
        definition of an event runs first), or after them with ``after=True``."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
        if after:
            self._hooks[stage].append(hook)
        else:
            self._hooks[stage].insert(0, hook)

    def _run_hooks(self, stage: str) -> None:
        for hook in self._hooks.get(stage, ()):
            hook(self)

    # -- helpers ---------------------------------------------------------------------
    def _apply_uf_bc(self, uf: FaceField) -> None:
        from ..boundary import SIDE_NAMES  # noqa: PLC0415

        for name, value in self.uf_normal.items():
            axis, side = SIDE_NAMES[name]
            sl = [slice(None)] * self.grid.ndim
            sl[axis] = 0 if side == 0 else self.grid.n[axis]
            uf[axis][tuple(sl)] = value

    def face_velocity(self) -> FaceField:
        """``uf = fm * face_value(u)``"""
        self.u.fill_ghosts()
        uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) for ax in range(self.grid.ndim)], "uf")
        self._apply_uf_bc(uf)
        return uf

    def correction(self, dt: float) -> None:
        for ax in range(self.grid.ndim):
            self.u[ax].interior = self.u[ax].interior + dt * self.g[ax].interior
        self.u.fill_ghosts()

    def centered_gradient(self) -> None:
        """g = average to cells of (a - alpha grad p) on faces. ``g_scale`` (set by a two-phase module, else None):
        per-axis face factors applied before the average (``alpha_mean / alpha_gfm`` of
        :class:`~..interface.vof_jump.TwoPhaseVOFJump`: the cells take the face's net force with their own
        density, docs/notes/65)."""
        self.p.fill_ghosts()
        scale = getattr(self, "g_scale", None)
        xp = self.grid.xp
        for ax in range(self.grid.ndim):
            gf = self.a[ax] - self.alpha[ax] * self.ops.grad_f(self.p, ax)
            if not self.p.bc.is_periodic(ax):
                # centered.h's wall pressure: p = neumann (a.n fm.n / alpha.n), whose face gradient cancels the
                # acceleration on the boundary faces; here the pressure keeps Neumann(0) and the boundary faces'
                # net acceleration is set to zero, as it results there (docs/notes/71)
                for side, k in ((0, 0), (1, -1)):
                    if isinstance(self.p.bc.get(ax, side), Neumann):
                        idx = [slice(None)] * self.grid.ndim
                        idx[ax] = slice(k, k + 1) if k == 0 else slice(-1, None)
                        gf = xp.array(gf, copy=True) if gf is self.a[ax] else gf
                        gf[tuple(idx)] = 0.0
            if scale is not None:
                gf = gf * scale[ax]
            self.g[ax].interior = self.ops.face_to_cell(gf, ax, periodic=self.p.bc.is_periodic(ax))
        self.g.fill_ghosts()

    # -- initialisation (Basilisk events "defaults" and "init") --------------------------
    def start(self, events: list[float] | None = None) -> None:
        """Call after setting the initial ``u`` (and optionally ``p``/``g``)."""
        self.clock = Clock(events=sorted(events or []))
        self.u.fill_ghosts()
        self.uf = self.face_velocity()
        self._run_hooks("properties")
        self.dtmax = self.DT
        self._run_hooks("stability")  # Basilisk's init calls event ("stability")
        # Basilisk's init runs dtnext() before any event time is known (tnext = HUGE): the step is
        # kept and tnext = t + dt0, which the first step then also aligns to.
        if self.basilisk_init_dt:
            dt0 = self.dtmax if self.stokes else timestep(self.uf, self.dtmax, self.cfl, self.clock)
            self.clock.dt = dt0
            if dt0 < 1e29:
                self.clock.events = sorted(self.clock.events + [self.clock.t + dt0])
        else:
            self._stability()

    # -- stages ------------------------------------------------------------------------
    def _stability(self) -> None:
        dt = self.dtmax if self.stokes else timestep(self.uf, self.dtmax, self.cfl, self.clock)
        self.clock.dt = dtnext(dt, self.clock)

    def step(self) -> None:
        c = self.clock
        self._run_hooks("set_dtmax")
        self.dtmax = self.DT
        self._run_hooks("stability")  # e.g. surface tension lowers dtmax here
        self._stability()
        dt = c.dt
        for stage in ("vof", "tracer_advection", "tracer_diffusion", "properties"):
            self._run_hooks(stage)

        self._run_hooks("advection_term")
        if not self.stokes:
            self.uf = prediction(self.u, self.g, dt)
            self._apply_uf_bc(self.uf)
            self.mgpf = project(self.uf, self.pf, self.alpha, dt / 2.0, ops=self.ops,
                                nrelax=self.mgpf.nrelax or 4, tolerance=self.tolerance,
                                basilisk_tolerance=True)
            self._apply_uf_bc(self.uf)
            advection(list(self.u), self.uf, dt, list(self.g), ops=self.ops)

        self._run_hooks("viscous_term")
        if self.coupled_iterations > 0:
            self._coupled_step(dt)
            self._run_hooks("end_timestep")
            c.t = c.tnext
            c.i += 1
            return
        if self.mu is not None:
            self.correction(dt)
            if self.viscous is None:
                self.mgu = viscosity(self.u, self.mu, self.rho, dt, nrelax=self.mgu.nrelax or 4,
                                     tolerance=self.tolerance)
            else:
                self.mgu = self.viscous.solve(self.u, self.mu, self.rho, dt, tolerance=self.tolerance)
            self.correction(-dt)
        if not self.a_is_constant:
            for ax in range(self.grid.ndim):
                self.a[ax] = 0.0

        self._run_hooks("acceleration")
        self.u.fill_ghosts()
        self.uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) + dt * self.a[ax]
                                        for ax in range(self.grid.ndim)], "uf")
        self._apply_uf_bc(self.uf)

        self._run_hooks("projection")
        self.mgp = project(self.uf, self.p, self.alpha, dt, ops=self.ops,
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance,
                           basilisk_tolerance=True)
        self._apply_uf_bc(self.uf)
        self.centered_gradient()
        self.correction(dt)

        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

    def _coupled_step(self, dt: float) -> None:
        """Viscosity, acceleration and projection solved together by outer iterations.

        The acceleration ``a`` of the new step (e.g. the GFM jump of the interface just moved) is
        computed before the viscosity; each iteration solves ``(I - dt/rho L) w = u_adv + dt g``,
        ``g = face_to_cell(a - alpha grad p)`` with the latest pressure, sets ``u = w - dt g``,
        projects ``interp(u) + dt a`` and updates ``g``. At convergence
        ``(I - dt/rho L) u^{n+1} = u_adv + dt (a - alpha grad p^{n+1})`` with ``div u^{n+1} = 0``: the
        viscosity acts on the velocity the new force and pressure produce. With ``u_adv = 0`` and ``a``
        balanced by ``grad p`` the iterate stays zero (well-balanced).

        The iteration is a fixed point ``p_{k+1} = T(p_k)`` of the pressure (the projection's solution
        does not depend on its initial guess). Stopped after iteration k, the momentum residual is
        ``-(dt^2/rho) L (g_{k+1} - g_k)``: the change ``dt |g_{k+1} - g_k|`` amplified by the viscous
        operator (by up to ``4 nu dt d/h^2``)."""
        xp = self.grid.xp
        if not self.a_is_constant:
            for ax in range(self.grid.ndim):
                self.a[ax] = 0.0
        self._run_hooks("acceleration")
        p_now = self.p.interior.copy()
        if self.coupled_predictor == "linear" and self._p_prev is not None:
            self.p.interior = 2.0 * p_now - self._p_prev
            self.p.fill_ghosts()
        self._p_prev = p_now
        self.centered_gradient()
        u_adv = [c.interior.copy() for c in self.u]
        #: max |g_{k+1} - g_k| of every outer iteration of the last step (convergence diagnostics)
        self.coupled_history = []
        self.coupled_residuals = []  # momentum residual after each iteration (criterion "residual")
        X, F = [], []  # Anderson: pressure iterates and their residuals T(p) - p
        criterion = self.coupled_tolerance is not None or self.coupled_rtol is not None
        converged = not criterion
        for _ in range(self.coupled_iterations):
            g_prev = [c.interior.copy() for c in self.g]
            p_in = self.p.interior.copy()
            for comp, v in zip(self.u, u_adv):
                comp.interior = v.copy()
            self.u.fill_ghosts()
            if self.mu is not None:
                self.correction(dt)
                if self.viscous is None:
                    self.mgu = viscosity(self.u, self.mu, self.rho, dt, nrelax=self.mgu.nrelax or 4,
                                         tolerance=self.tolerance)
                else:
                    self.mgu = self.viscous.solve(self.u, self.mu, self.rho, dt, tolerance=self.tolerance)
                self.correction(-dt)
            self.u.fill_ghosts()
            self.uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) + dt * self.a[ax]
                                            for ax in range(self.grid.ndim)], "uf")
            self._apply_uf_bc(self.uf)
            self._run_hooks("projection")
            self.mgp = project(self.uf, self.p, self.alpha, dt, ops=self.ops,
                               nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance,
                               basilisk_tolerance=True)
            self._apply_uf_bc(self.uf)
            self.centered_gradient()
            self.coupled_history.append(max(float(xp.max(xp.abs(c.interior - gp))) for c, gp in zip(self.g, g_prev)))
            if self.coupled_criterion == "residual":
                self.coupled_residuals.append(self.momentum_residual(u_adv, dt))
            if criterion and len(self.coupled_history) >= 2:  # the first change is the step's, not the iteration's
                seq = ([dt * c for c in self.coupled_history] if self.coupled_criterion == "change"
                       else self.coupled_residuals)
                bound = max(self.coupled_tolerance or 0.0, (self.coupled_rtol or 0.0) * seq[0])
                if seq[-1] < bound:
                    converged = True
                    break
            if self.coupled_anderson > 0 and len(self.coupled_history) < self.coupled_iterations:
                self._anderson(X, F, p_in)
        self.coupled_info = {"iterations": len(self.coupled_history), "converged": converged,
                             "change": dt * self.coupled_history[-1] if self.coupled_history else 0.0}
        if not converged:
            self.coupled_failures += 1
            msg = (f"coupled iterations: {self.coupled_iterations} iterations without reaching "
                   f"tol {self.coupled_tolerance} / rtol {self.coupled_rtol} ({self.coupled_criterion}) "
                   f"at step {self.clock.i}")
            if self.coupled_on_failure == "raise":
                raise RuntimeError(msg)
            if self.coupled_on_failure == "warn":
                import warnings  # noqa: PLC0415

                warnings.warn(msg, RuntimeWarning, stacklevel=3)
        self.correction(dt)

    def _anderson(self, X, F, p_in) -> None:
        """Replace the projection's pressure ``T(p_in)`` by the Anderson mixture of the last
        ``coupled_anderson + 1`` iterates (type II, no damping), and recompute ``g`` from it. The face
        velocity stays the one projected with ``T(p_in)`` (divergence-free)."""
        xp = self.grid.xp
        Tp = self.p.interior.copy()
        f = Tp - p_in
        f = f - xp.mean(f)  # the gauge
        X.append(Tp)
        F.append(f)
        if len(F) > self.coupled_anderson + 1:
            X.pop(0)
            F.pop(0)
        if len(F) < 2:
            return
        dF = xp.stack([(F[i + 1] - F[i]).ravel() for i in range(len(F) - 1)], axis=1)
        dX = [X[i + 1] - X[i] for i in range(len(X) - 1)]
        gamma = xp.linalg.lstsq(dF, F[-1].ravel(), rcond=None)[0]
        mix = X[-1] - sum(float(gamma[i]) * dX[i] for i in range(len(dX)))
        self.p.interior = mix
        self.p.fill_ghosts()
        self.centered_gradient()

    def momentum_residual(self, u_adv, dt: float) -> float:
        """``max |u_adv + dt g - (I - dt/rho L) u|`` for the velocity ``u = self.u + dt g`` the step would end
        with now (Basilisk ``viscosity()`` operator; ``u - u_adv - dt g`` without viscosity)."""
        xp = self.grid.xp
        v = self.u.copy()
        for ax in range(self.grid.ndim):
            v[ax].interior = self.u[ax].interior + dt * self.g[ax].interior
        v.fill_ghosts()
        b = [xp.ascontiguousarray(u_adv[ax] + dt * self.g[ax].interior) for ax in range(self.grid.ndim)]
        if self.mu is None:
            return max(float(xp.max(xp.abs(b[ax] - v[ax].interior))) for ax in range(self.grid.ndim))
        if self.viscous is not None:
            raise NotImplementedError("the momentum residual uses Basilisk's viscosity() operator")
        from ..linear.viscosity import ViscosityProblem  # noqa: PLC0415

        problem = ViscosityProblem(v, self.mu, self.rho, dt)
        res = [xp.zeros_like(c) for c in b]
        problem.residual(list(v), b, res)
        return max(float(xp.max(xp.abs(r))) for r in res)

    def divergence_residuals(self) -> tuple[float, float]:
        """``dt max |div uf|`` (faces, what the projection enforces) and ``dt max |div interp(u)|`` (the
        centred velocity, which the approximate projection leaves divergent)."""
        xp = self.grid.xp
        per = tuple(ax for ax in range(self.grid.ndim) if ax in self.periodic)
        dt = float(self.clock.dt)
        face = dt * float(xp.max(xp.abs(self.ops.div_f(self.uf, periodic=per))))
        cell = dt * float(xp.max(xp.abs(self.ops.div_f(self.face_velocity(), periodic=per))))
        return face, cell

    def run(self, t_end: float, *, every: Callable[["CenteredNS"], bool | None] | None = None,
            max_steps: int | None = None) -> None:
        """Advance to ``t_end``. ``every`` is called after each step; returning True stops."""
        if t_end not in self.clock.events:
            self.clock.events.append(t_end)
        while self.clock.t < t_end - TEPS * max(1.0, t_end):
            if max_steps is not None and self.clock.i >= max_steps:
                break
            self.step()
            if every is not None and every(self):
                break

    # -- diagnostics ------------------------------------------------------------------
    def kinetic_energy(self) -> float:
        xp = self.grid.xp
        return float(sum(xp.sum(c.interior ** 2) for c in self.u) / 2.0 * self.grid.cell_volume())

    def max_divergence(self) -> float:
        xp = self.grid.xp
        return float(xp.max(xp.abs(self.ops.div_f(self.uf, periodic=self.p.bc.periodic_axes(self.grid.ndim)))))

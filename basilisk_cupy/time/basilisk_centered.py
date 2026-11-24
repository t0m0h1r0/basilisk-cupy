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
    def __init__(self, grid: CartesianGrid, *, periodic: tuple[int, ...] = (),
                 normal: dict | None = None, tangential: dict | None = None,
                 uf_normal: dict | None = None,
                 mu: float | FaceField | None = None,
                 cfl: float = 0.8, DT: float = 1e30, stokes: bool = False,
                 ops: FaceComplex | None = None, tolerance: float = TOLERANCE) -> None:
        self.grid = grid
        self.ops = ops or FD2FaceComplex()
        self.periodic = tuple(periodic)
        self.cfl, self.DT, self.stokes, self.tolerance = cfl, DT, stokes, tolerance
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
        """g = average to cells of (a - alpha grad p) on faces."""
        self.p.fill_ghosts()
        for ax in range(self.grid.ndim):
            gf = self.a[ax] - self.alpha[ax] * self.ops.grad_f(self.p, ax)
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
                                nrelax=self.mgpf.nrelax or 4, tolerance=self.tolerance)
            self._apply_uf_bc(self.uf)
            advection(list(self.u), self.uf, dt, list(self.g), ops=self.ops)

        self._run_hooks("viscous_term")
        if self.mu is not None:
            self.correction(dt)
            self.mgu = viscosity(self.u, self.mu, self.rho, dt, nrelax=self.mgu.nrelax or 4,
                                 tolerance=self.tolerance)
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
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance)
        self._apply_uf_bc(self.uf)
        self.centered_gradient()
        self.correction(dt)

        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

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

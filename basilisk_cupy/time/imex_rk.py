"""IMEX Runge--Kutta Navier--Stokes stepper: explicit convection, implicit viscosity, a projection per stage.

Motivation (measured, see README "Time step"): the EXT2 convection of
:class:`~basilisk_cupy.time.imex_bdf2.IMEXBDF2NS` has no stable interval on the
imaginary axis, so the non-dissipative FCCD/CCD convection is held stable only
by viscosity (inviscid Taylor--Green: CFL <= 0.3 with FCCD), and the UCCD6
hyperviscosity gives the real-axis bound ``sigma omega_2^8 CFL <= 4/3``. The
explicit part of ARS(3,4,3) (Ascher, Ruuth & Spiteri 1997, section 2.7) has
the stability region of classical RK4: imaginary-axis limit 2.83, real-axis
limit -2.79; its implicit part is an L-stable SDIRK with one coefficient
``gamma`` on every stage, so the viscous problem (and its recorded CUDA graphs)
is reused within and across steps.

A stage ``i`` (``i >= 2``; stage 1 is ``u^n``) is, with incremental pressure
correction as in :class:`IMEXBDF2NS` (``g`` = cell ``a - alpha grad p``)::

    (I - gamma dt V) u*_i = u^n + dt sum_{j<i} (aE_ij C_j + aI_ij V_j) + c_i dt g^n
    u_f* = P_f(u*_i - c_i dt g^n) + c_i dt (a - alpha G_f p^n)
    D_f alpha G_f dp_i = D_f u_f* / (c_i dt),   u_f,i = u_f* - c_i dt alpha G_f dp_i
    p_i = p^n + dp_i,   u_i = u*_i + c_i dt (g_i - g^n)

``V_j = (u*_j - rhs_j)/(gamma dt)`` is the viscous tendency of the solved stage
and ``C_j = -C(u_j, u_f,j)`` the convection of the projected one. When the
explicit weights differ from the last explicit row (ARS(3,4,3)) the step ends
with ``u** = u_s + dt sum_j (bE_j - aE_sj) C_j`` and one more projection
(``c = 1``); stiffly accurate tableaux (ARS(4,4,3)) end with the last stage.
The implicit part is stiffly accurate in both, so ``u^{n+1}`` carries the
implicit update exactly.

``face_native=True`` (see :mod:`.imex_bdf2`): every stage moves the faces by the
interpolated cell increment, ``u_f*_i = u_f^n + P_f(u*_i - u^n - c_i dt g^n) + c_i dt (a - alpha G_f p^n)``,
projects them and reconstructs the cells; the final ARS(3,4,3) correction projects
``u_f,s + P_f(dt sum_j (bE_j - aE_sj) C_j)`` over ``p_s``.

Step size: a fixed ``fixed_dt``, or Basilisk's CFL step capped by the
convection term's spectrum (``spectrum(u, uf)``: largest imaginary and real
parts) and the tableau's axis limits times ``safety``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import FaceField
from ..projection.approximate import project
from .imex_bdf2 import IMEXBDF2NS
from .stepping import dtnext


@dataclass(frozen=True)
class IMEXTableau:
    name: str
    AE: tuple
    AI: tuple
    bE: tuple
    c: tuple
    gamma: float
    imag_limit: float  # explicit part, measured from the stability function
    real_limit: float
    bI: tuple | None = None  # implicit weights (None: the last row of AI, stiffly accurate)

    @property
    def stages(self) -> int:
        return len(self.c)

    @property
    def explicit_stiffly_accurate(self) -> bool:
        return np.allclose(self.AE[-1], self.bE)

    @property
    def implicit_weights(self) -> tuple:
        return tuple(self.AI[-1]) if self.bI is None else tuple(self.bI)

    @property
    def implicit_stiffly_accurate(self) -> bool:
        return np.allclose(self.AI[-1], self.implicit_weights)


def axis_limits(AE, bE) -> tuple[float, float]:
    """Imaginary- and real-axis stability limits of an explicit tableau (``|R(z)| <= 1`` from 0 outwards)."""
    A = np.array(AE, float)
    b = np.array(bE, float)
    s = len(b)

    def R(z):
        return 1 + z * b @ np.linalg.solve(np.eye(s) - z * A, np.ones(s))

    out = []
    for direction in (1j, -1.0):
        x, step = 0.0, 1e-3
        while x < 10.0 and abs(R((x + step) * direction)) <= 1.0 + 1e-12:
            x += step
        out.append(round(x, 3))
    return out[0], out[1]


def _make(name, AE, AI, bE, c, gamma, bI=None) -> IMEXTableau:
    imag, real = axis_limits(AE, bE)
    return IMEXTableau(name, AE, AI, bE, c, gamma, imag, real, bI)


def _ars111() -> IMEXTableau:
    """Forward--backward Euler (Ascher, Ruuth & Spiteri 1997, section 2.1)."""
    return _make("ars111", ((0, 0), (1, 0)), ((0, 0), (0, 1)), (1, 0), (0, 1), 1.0)


def _ars121() -> IMEXTableau:
    """Forward Euler with the implicit midpoint rule (ARS 1997, section 2.2): not stiffly accurate."""
    return _make("ars121", ((0, 0), (0.5, 0)), ((0, 0), (0, 0.5)), (0, 1), (0, 0.5), 0.5, bI=(0, 1))


def _ars222() -> IMEXTableau:
    """L-stable SDIRK2 with a 2nd-order explicit part (ARS 1997, section 2.6)."""
    g = 1.0 - 1.0 / np.sqrt(2.0)
    d = 1.0 - 1.0 / (2.0 * g)
    return _make("ars222", ((0, 0, 0), (g, 0, 0), (d, 1 - d, 0)), ((0, 0, 0), (0, g, 0), (0, 1 - g, g)),
                 (d, 1 - d, 0), (0, g, 1), g)


def _ars233() -> IMEXTableau:
    """3rd-order SDIRK, gamma = (3 + sqrt 3)/6 (ARS 1997, section 2.4): not stiffly accurate."""
    g = (3.0 + np.sqrt(3.0)) / 6.0
    return _make("ars233", ((0, 0, 0), (g, 0, 0), (g - 1, 2 * (1 - g), 0)), ((0, 0, 0), (0, g, 0), (0, 1 - 2 * g, g)),
                 (0, 0.5, 0.5), (0, g, 1 - g), g, bI=(0, 0.5, 0.5))


def _ars343() -> IMEXTableau:
    g = 0.4358665215
    b1 = -1.5 * g * g + 4 * g - 0.25
    b2 = 1.5 * g * g - 5 * g + 1.25
    AI = ((0, 0, 0, 0), (0, g, 0, 0), (0, (1 - g) / 2, g, 0), (0, b1, b2, g))
    AE = ((0, 0, 0, 0), (g, 0, 0, 0), (0.3212788860, 0.3966543747, 0, 0),
          (-0.105858296, 0.5529291479, 0.5529291479, 0))
    return IMEXTableau("ars343", AE, AI, (0, b1, b2, g), (0, g, (1 + g) / 2, 1), g, 2.828, 2.785)


def _ars443() -> IMEXTableau:
    AI = ((0, 0, 0, 0, 0), (0, 0.5, 0, 0, 0), (0, 1 / 6, 0.5, 0, 0), (0, -0.5, 0.5, 0.5, 0),
          (0, 1.5, -1.5, 0.5, 0.5))
    AE = ((0, 0, 0, 0, 0), (0.5, 0, 0, 0, 0), (11 / 18, 1 / 18, 0, 0, 0), (5 / 6, -5 / 6, 0.5, 0, 0),
          (0.25, 1.75, 0.75, -1.75, 0))
    return IMEXTableau("ars443", AE, AI, AE[-1], (0, 0.5, 2 / 3, 0.5, 1), 0.5, 1.569, 2.142)


TABLEAUX = {"ars111": _ars111, "ars121": _ars121, "ars222": _ars222, "ars233": _ars233, "ars343": _ars343,
            "ars443": _ars443}


def stability_function(tableau: IMEXTableau, z):
    """Explicit-part stability function ``R(z) = 1 + z bE (I - z AE)^{-1} 1``."""
    A = np.array(tableau.AE, float)
    b = np.array(tableau.bE, float)
    s = len(b)
    return 1 + z * b @ np.linalg.solve(np.eye(s) - z * A, np.ones(s))


class IMEXRKNS(IMEXBDF2NS):
    """Same fields, conditions, operators and hook stages as :class:`IMEXBDF2NS`."""

    def __init__(self, grid, *, tableau: str = "ars343", safety: float = 0.9, **kwargs) -> None:
        super().__init__(grid, **kwargs)
        try:
            self.tableau = TABLEAUX[tableau]()
        except KeyError:
            raise ValueError(f"unknown tableau {tableau!r}; available: {sorted(TABLEAUX)}") from None
        self.safety = safety

    # -- step size -----------------------------------------------------------------------
    def _convection_dt_limit(self) -> float:
        spectrum = getattr(self.convection, "spectrum", None)
        if self.stokes or spectrum is None:
            return float("inf")
        imag, real = spectrum(self.u, self.uf)
        t = self.tableau
        # no stable interval on the imaginary axis (ars111, ars121, ars222): as EXT2 in IMEXBDF2NS, the
        # non-dissipative convection is then held stable by viscosity only and no bound is applied
        limits = [t.imag_limit / imag if imag > 0 and t.imag_limit > 0.01 else float("inf"),
                  t.real_limit / real if real > 0 else float("inf")]
        return self.safety * min(limits)

    # -- one step ------------------------------------------------------------------------
    def _project_stage(self, ustar: list, g_n: list, p_n, cdt: float) -> None:
        """IPC projection of the stage: sets ``u``, ``uf``, ``p``, ``g``."""
        ndim = self.grid.ndim
        for ax in range(ndim):
            self.u[ax].interior = ustar[ax] - cdt * g_n[ax]
        self.u.fill_ghosts()
        self.p.interior = p_n
        self.p.fill_ghosts()
        self.uf = FaceField(self.grid, [self.ops.interp_f(self.u[ax], ax) + cdt * self._face_pressure_force(self.p, ax)
                                        for ax in range(ndim)], "uf")
        self._apply_uf_bc(self.uf)
        self.dp.data[...] = 0.0
        self.mgp = project(self.uf, self.dp, self.alpha, cdt, ops=self.ops,
                           nrelax=self.mgp.nrelax or 4, tolerance=self.tolerance)
        self._apply_uf_bc(self.uf)
        self.p.interior = p_n + self.dp.interior
        self.p.fill_ghosts()
        self.centered_gradient()
        self.correction(cdt)  # u = u* - cdt g^n + cdt g_i

    def step(self) -> None:
        c = self.clock
        self._run_hooks("set_dtmax")
        self.dtmax = self.DT
        self._run_hooks("stability")
        self._stability()
        dt = c.dt
        for stage in ("vof", "tracer_advection", "tracer_diffusion", "properties"):
            self._run_hooks(stage)

        t = self.tableau
        ndim = self.grid.ndim
        if self.face_native and not getattr(self, "_fn_started", False):
            self._reconstruct_cells()  # the state is uf; cells are its reconstruction
            self._fn_started = True
        uf_n = [self.uf[ax].copy() for ax in range(ndim)]
        u_n = [comp.interior.copy() for comp in self.u]
        g_n = [comp.interior.copy() for comp in self.g]
        p_n = self.p.interior.copy()
        zero = [0.0] * ndim
        C = []  # explicit tendencies of the projected stages
        V = []  # viscous tendencies of the solved stages
        gdt = t.gamma * dt

        extra = self._implicit_term()  # implicit hyperviscosity frozen at u^n
        implicit = self.mu is not None or extra is not None
        self._run_hooks("advection_term")
        C.append(zero if self.stokes else self.convection.tendency(self.u, self.uf))
        V.append(zero)
        for i in range(1, t.stages):
            rhs = []
            for ax in range(ndim):
                r = u_n[ax] + t.c[i] * dt * g_n[ax]
                for j in range(i):
                    if t.AE[i][j]:
                        r = r + dt * t.AE[i][j] * C[j][ax]
                    if t.AI[i][j]:
                        r = r + dt * t.AI[i][j] * V[j][ax]
                rhs.append(r)
            for ax in range(ndim):
                self.u[ax].interior = rhs[ax]
            self.u.fill_ghosts()
            self._run_hooks("viscous_term")
            self._implicit_solve(gdt, extra)
            ustar = [comp.interior.copy() for comp in self.u]
            V.append([(us - r) / gdt for us, r in zip(ustar, rhs)] if implicit else zero)
            self._run_hooks("acceleration")
            self._run_hooks("projection")
            cdt = t.c[i] * dt
            if self.face_native:
                finc = self._faces_from_increment([us - un - cdt * gn for us, un, gn in zip(ustar, u_n, g_n)])
                self.p.interior = p_n
                self._project_faces([uf_n[ax] + finc[ax] + cdt * self._face_pressure_force(self.p, ax)
                                     for ax in range(ndim)], p_n, cdt)
            else:
                self._project_stage(ustar, g_n, p_n, cdt)
            if i < t.stages - 1 or not t.explicit_stiffly_accurate or not t.implicit_stiffly_accurate:
                C.append(zero if self.stokes else self.convection.tendency(self.u, self.uf))

        if not t.implicit_stiffly_accurate:
            # u^{n+1} = u^n + dt sum_j (bE_j C_j + bI_j V_j) (+ dt g^n, removed again by the IPC projection)
            if self.face_native:
                raise NotImplementedError("face_native with a tableau whose implicit part is not stiffly accurate")
            bI = t.implicit_weights
            final = []
            for ax in range(ndim):
                v = u_n[ax] + dt * g_n[ax]
                for j in range(t.stages):
                    if t.bE[j] and not self.stokes:
                        v = v + dt * t.bE[j] * C[j][ax]
                    if bI[j]:
                        v = v + dt * bI[j] * V[j][ax]
                final.append(v)
            self._project_stage(final, g_n, p_n, dt)
        elif not t.explicit_stiffly_accurate and not self.stokes and self.face_native:
            s = t.stages - 1
            inc = [sum(dt * (t.bE[j] - t.AE[s][j]) * C[j][ax] for j in range(t.stages) if t.bE[j] != t.AE[s][j])
                   for ax in range(ndim)]
            finc = self._faces_from_increment(inc)
            self._project_faces([self.uf[ax] + finc[ax] for ax in range(ndim)], self.p.interior.copy(), dt)
        elif not t.explicit_stiffly_accurate and not self.stokes:
            s = t.stages - 1
            final = []
            for ax in range(ndim):
                v = self.u[ax].interior.copy()
                for j in range(t.stages):
                    w = t.bE[j] - t.AE[s][j]
                    if w:
                        v = v + dt * w * C[j][ax]
                # u_s carries dt g_s; swap it for dt g^n, which the IPC projection from p^n removes again
                final.append(v - dt * self.g[ax].interior + dt * g_n[ax])
            self._project_stage(final, g_n, p_n, dt)

        if not self.a_is_constant:
            for ax in range(ndim):
                self.a[ax] = 0.0
        self._run_hooks("end_timestep")
        c.t = c.tnext
        c.i += 1

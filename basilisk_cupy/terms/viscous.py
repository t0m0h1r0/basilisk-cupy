"""Implicit viscous step ``(I - dt/rho L) u^{n+1} = u^*`` with a swappable spatial operator ``L``.

``L u = div(mu (grad u + grad u^T))`` (full stress, Basilisk's ``div(2 mu D)``).

* :class:`BasiliskViscosity` — Basilisk ``viscosity()`` (fd2 stencil, coupled
  red/black multigrid; :mod:`basilisk_cupy.linear.viscosity`). The default.
* :class:`CCDViscosity` — TwoPhaseFlow's high-order viscous Helmholtz (paper
  §6.3 ``06d_viscous_3layer.tex`` for ``L``, §7.4 ``07b_defect_viscous.tex``
  for the solver): the residual is evaluated with the high-order operator
  ``A_H = I - dt/rho L_H`` and corrected with the fd2 operator ``A_L`` of
  Basilisk's viscosity multigrid (defect correction, GCR-accelerated as the
  Poisson one in :mod:`basilisk_cupy.linear.defect_correction`). The fixed point
  is ``A_H u = b`` whatever ``A_L``; the stopping test is on the ``A_H``
  residual.

``L_H`` (single phase, no interface band) follows the paper's three layers:

* layer A — velocity gradients ``d_b u_a`` with the cell-centred CCD first
  derivative and the velocity boundary conditions;
* layer B — stress ``tau_ab = mu (d_b u_a + d_a u_b)`` at cell centres
  (shared ``tau_xy``);
* layer C — stress divergence ``sum_b D^C_b tau_ab`` taken once, with the CCD
  first derivative of the cell array (no wall datum for the stress: the
  extrapolating closure at walls).

``form="split"`` (the default) evaluates the same-axis stress derivative with
the CCD second derivative, ``d_b (mu d_b u) = mu D2_b u + (D1_b mu)(D1_b u)``,
and keeps layers A/B/C for the cross terms ``D1_b (mu D1_a u_b)``, ``b != a``.
It is still the full stress (not the ``mu lap u`` shortcut the paper forbids)
and holds for variable ``mu``. The reason: the CCD first-derivative symbol
vanishes at ``theta = pi``, so the layered ``D1 D1`` product does not act on the
grid-scale mode while the fd2 correction operator ``A_L`` damps it most; the
defect correction then stalls as ``kappa = nu dt / h^2`` grows (measured, N = 64,
walls: 24 / 77 corrections at kappa = 0.1 / 1, no convergence in 300 at
kappa >= 10), whereas ``D2`` damps that mode like ``A_L`` (15-26 corrections for
kappa = 0.1 ... 1000). ``form="layered"`` keeps the paper's literal composition
for comparison.

Relation to Basilisk: ``viscosity.h`` has the same structure as ``split`` with
2nd-order differences, a compact face-flux difference ``2 mu (u[1] - u[])/Delta``
for the same-axis term and corner averages for the cross terms; ``split``
replaces each difference by its CCD counterpart. It is a choice made here,
not a scheme of Basilisk or of the TwoPhaseFlow paper (whose literal
composition is ``layered``).
"""

from __future__ import annotations

from ..backend import to_float
from ..discretization.ccd import CCDCellDerivative
from ..fields import CellField, FaceField, VectorField
from ..linear.multigrid import NITERMAX, TOLERANCE, MGStats, mg_solve
from ..linear.viscosity import ViscosityProblem, viscosity


class BasiliskViscosity:
    """Basilisk ``viscosity()``: fd2 full stress, multigrid."""

    name = "basilisk"

    def solve(self, u: VectorField, mu: FaceField, rho, dt: float, *, nrelax: int = 4,
              tolerance: float = TOLERANCE) -> MGStats:
        return viscosity(u, mu, rho, dt, nrelax=nrelax, tolerance=tolerance)


def cell_viscosity(mu: FaceField):
    """Cell-centred viscosity from the face field: mean of the two faces, averaged over the axes
    (exact for a constant viscosity)."""
    grid = mu.grid
    acc = None
    for ax in range(grid.ndim):
        n = grid.n[ax]
        lo = [slice(None)] * grid.ndim
        hi = [slice(None)] * grid.ndim
        lo[ax] = slice(0, n)
        hi[ax] = slice(1, n + 1)
        val = 0.5 * (mu[ax][tuple(lo)] + mu[ax][tuple(hi)])
        acc = val if acc is None else acc + val
    return acc / grid.ndim


class CCDStress:
    """High-order full-stress divergence ``L_H u`` (layers A/B/C of the paper)."""

    def __init__(self, ccd: CCDCellDerivative | None = None, form: str = "split") -> None:
        if form not in ("layered", "split"):
            raise ValueError("form must be 'split' or 'layered'")
        self.ccd = ccd or CCDCellDerivative()
        self.form = form

    def __call__(self, u: VectorField, mu_cell) -> list:
        """``L_H u`` for every component (interior arrays); ``u`` needs filled ghosts only for its BCs."""
        grid = u.grid
        ndim = grid.ndim
        ccd = self.ccd
        if self.form == "split":
            scalar_mu = not hasattr(mu_cell, "shape") or mu_cell.ndim == 0
            dmu = None if scalar_mu else [ccd.d1_array(mu_cell, grid, b, u[0].bc.is_periodic(b)) for b in range(ndim)]
            d1 = [[ccd.derivatives(u[a], b) for b in range(ndim)] for a in range(ndim)]  # (d_b u_a, d_bb u_a)
            out = []
            for a in range(ndim):
                acc = None
                for b in range(ndim):
                    if a == b:  # d_a (2 mu d_a u_a)
                        term = 2.0 * (mu_cell * d1[a][a][1] + (0.0 if scalar_mu else dmu[a] * d1[a][a][0]))
                    else:       # d_b (mu d_b u_a) + d_b (mu d_a u_b)
                        term = mu_cell * d1[a][b][1] + (0.0 if scalar_mu else dmu[b] * d1[a][b][0])
                        term = term + ccd.d1_array(mu_cell * d1[b][a][0], grid, b, u[a].bc.is_periodic(b))
                    acc = term if acc is None else acc + term
                out.append(acc)
            return out
        grads = [[ccd.d1(u[a], b) for b in range(ndim)] for a in range(ndim)]  # layer A: d_b u_a
        out = []
        for a in range(ndim):
            acc = None
            for b in range(ndim):
                tau = mu_cell * (grads[a][b] + grads[b][a])                    # layer B
                term = ccd.d1_array(tau, grid, b, u[a].bc.is_periodic(b))       # layer C
                acc = term if acc is None else acc + term
            out.append(acc)
        return out


class FD2Stress:
    """2nd-order full-stress divergence with Basilisk's structure (finite volume on mapped grids).

    Same-axis terms ``d_b (mu d_b u_a)`` and ``d_a (2 mu d_a u_a)`` as compact face fluxes
    ``D_f (mu_f G_f u)`` (Basilisk ``2 mu (u[1] - u[]) / Delta``); cross terms
    ``d_b (mu d_a u_b)`` with centred cell differences. The high-order operator of
    :class:`FD2Viscosity`, a 2nd-order baseline wherever Basilisk's own multigrid does
    not apply (mapped grids).
    """

    def __init__(self) -> None:
        from ..discretization.fd2 import FD2CellDerivative, FD2FaceComplex  # noqa: PLC0415

        self.face = FD2FaceComplex()
        self.cell = FD2CellDerivative()

    def __call__(self, u: VectorField, mu_cell, mu: FaceField | None = None) -> list:
        grid = u.grid
        ndim = grid.ndim
        if mu is None:
            mu = FaceField.constant(grid, float(mu_cell), name="mu")
        u.fill_ghosts()
        out = []
        for a in range(ndim):
            fluxes = [(2.0 if b == a else 1.0) * mu[b] * self.face.grad_f(u[a], b) for b in range(ndim)]
            acc = self.face.div_f(FaceField(grid, fluxes))
            for b in range(ndim):
                if b != a:  # d_b (mu d_a u_b): centred differences through the velocity ghosts (corners included)
                    acc = acc + _cross_term(grid, u[b].data, mu_cell, a, b)
            out.append(acc)
        return out


def _centred(grid, data, axis: int, extend: int):
    """Centred first derivative along ``axis`` of a padded array, interior along ``axis`` and
    along every other axis except ``extend``, which keeps one ghost layer per side."""
    g = grid.ghosts
    sl = []
    for b in range(grid.ndim):
        n = grid.n[b]
        sl.append(slice(g - 1, g + n + 1) if b == extend else slice(g, g + n))
    def shifted(off):
        s = list(sl)
        s[axis] = slice(g + off, g + off + grid.n[axis])
        return data[tuple(s)]
    if grid.axis_uniform(axis):
        return (shifted(1) - shifted(-1)) / (2.0 * grid.h[axis])
    m = grid.metric(axis)
    wl, wc, wr = (m.along(w) for w in m.d1_weights)
    return wl * shifted(-1) + wc * shifted(0) + wr * shifted(1)


def _cross_term(grid, ub_data, mu_cell, a: int, b: int):
    """``d_b (mu d_a u_b)`` (Basilisk's cross term: corner values of ``u_b``)."""
    xp = grid.xp
    t = _centred(grid, ub_data, a, extend=b)  # d_a u_b on the interior plus one ghost layer along b
    if hasattr(mu_cell, "shape") and mu_cell.ndim > 0:
        pad = [(1, 1) if ax == b else (0, 0) for ax in range(grid.ndim)]
        t = xp.pad(mu_cell, pad, mode="edge") * t
    else:
        t = mu_cell * t
    n = grid.n[b]
    lo = [slice(None)] * grid.ndim
    mid = [slice(None)] * grid.ndim
    hi = [slice(None)] * grid.ndim
    lo[b], mid[b], hi[b] = slice(0, n), slice(1, n + 1), slice(2, n + 2)
    if grid.axis_uniform(b):
        return (t[tuple(hi)] - t[tuple(lo)]) / (2.0 * grid.h[b])
    m = grid.metric(b)
    wl, wc, wr = (m.along(w) for w in m.d1_weights)
    return wl * t[tuple(lo)] + wc * t[tuple(mid)] + wr * t[tuple(hi)]


def _dot(xp, x: list, y: list) -> float:
    return to_float(sum(xp.sum(a * b) for a, b in zip(x, y)))


class DCViscosity:
    """Implicit viscosity by defect correction: residual with ``A_H = I - dt/rho L_H`` (``stress``),
    correction with the fd2 operator ``A_L`` (GCR-accelerated).

    ``A_L``: Basilisk's coupled viscosity multigrid on a uniform grid; on a mapped grid,
    where that stencil does not exist (single Delta), the componentwise scalar Helmholtz
    ``(I - dt/rho div(mu grad)) z_a = r_a`` (fd2 finite volume, line-relaxed multigrid):
    the paper allows a scalar approximation for ``A_L``, which only sets the convergence
    rate, not the solution.
    """

    name = "dc"

    def __init__(self, stress, *, memory: int = 10, inner_cycles: int = 2, maxiter: int = NITERMAX) -> None:
        self.stress = stress
        self.memory = memory
        self.inner_cycles = inner_cycles
        self.maxiter = maxiter

    def high(self, u: VectorField, mu_cell, rho, dt: float, mu: FaceField | None = None, extra=None) -> list:
        """``A_H u = u - dt/rho L_H u (+ dt E u)``, ``E`` an implicit extra term (``apply``)."""
        u.fill_ghosts()
        out = [c.interior for c in u]
        if mu is not None:
            lap = self.stress(u, mu_cell, mu) if isinstance(self.stress, FD2Stress) else self.stress(u, mu_cell)
            out = [o - dt / rho * l for o, l in zip(out, lap)]
        if extra is not None:
            out = [o + dt * e for o, e in zip(out, extra.apply(u))]
        return out

    def _workspace(self, u: VectorField):
        """Correction fields with homogeneous conditions, kept with ``u`` (reuse of problems/graphs)."""
        ws = getattr(u, "_viscous_dc_workspace", None)
        homog = [c.bc.homogeneous() for c in u]
        if ws is None or ws[0].grid != u.grid or [c.bc for c in ws[0]] != homog:
            make = lambda name: VectorField([CellField(u.grid, bc=bc, name=f"{name}.{'xyz'[a]}")  # noqa: E731
                                             for a, bc in enumerate(homog)], name)
            ws = u._viscous_dc_workspace = (make("dc_work"), make("dc_z"))
        return ws

    def solve(self, u: VectorField, mu: FaceField | None, rho, dt: float, *, nrelax: int = 4,
              tolerance: float = TOLERANCE, history: list | None = None, extra=None) -> MGStats:
        """Advance ``u`` in place; ``u``'s interior on entry is the right-hand side ``u^*``.

        ``extra``: an implicit term ``E`` (e.g. :class:`~basilisk_cupy.terms.convection.uccd6.ImplicitHyperviscosity`)
        solved together, ``(I - dt/rho L + dt E) u = u^*``; its ``precondition(r, dt, bcs)``
        follows the viscous multigrid. ``mu = None``: ``(I + dt E) u = u^*`` alone.
        """
        grid = u.grid
        xp = grid.xp
        mu_cell = cell_viscosity(mu) if mu is not None else None
        b = [c.interior.copy() for c in u]
        work, z = self._workspace(u)
        homog = [c.bc for c in z]
        low = lows = None
        if mu is None:
            pass
        elif grid.is_uniform:
            low = ViscosityProblem.reuse(z, mu, rho, dt)
            lows = None
        else:
            from ..linear.poisson import HelmholtzProblem  # noqa: PLC0415

            lam = -rho / dt  # div(mu grad z) - rho/dt z = -rho/dt r
            lows = [HelmholtzProblem.reuse(c, mu, lam if hasattr(lam, "shape") else xp.full(grid.n, lam)) for c in z]
            low = lows[0]

        def true_residual():
            ah = self.high(u, mu_cell, rho, dt, mu, extra)
            r = [bi - ai for bi, ai in zip(b, ah)]
            return max(to_float(xp.max(xp.abs(ri))) for ri in r), r

        def apply_high0(v):
            for c, vi in zip(work, v):
                c.interior = vi
            return self.high(work, mu_cell, rho, dt, mu, extra)

        def precondition(r):
            if mu is None:
                out = [ri.copy() for ri in r]
            else:
                for c in z:
                    c.data[...] = 0.0
                if lows is None:
                    mg_solve(list(z), r, low, nrelax=nrelax, tolerance=0.0, nitermax=self.inner_cycles,
                             nitermin=self.inner_cycles, warn=False)
                else:
                    for c, ri, problem in zip(z, r, lows):
                        mg_solve(c, -rho / dt * ri, problem, nrelax=nrelax, tolerance=0.0,
                                 nitermax=self.inner_cycles, nitermin=self.inner_cycles, warn=False)
                out = [c.interior.copy() for c in z]
            return extra.precondition(out, dt, homog) if extra is not None else out

        stats = MGStats(nrelax=0, minlevel=low.levels[0].n[0] if low is not None else grid.n[0], tolerance=tolerance)
        stats.resb, r = true_residual()
        stats.resa = stats.resb
        if history is not None:
            history.append(stats.resa)
        dirs: list = []
        while stats.i < self.maxiter and stats.resa > tolerance:
            zk = precondition(r)
            stats.nrelax += self.inner_cycles
            wk = apply_high0(zk)
            for zj, wj, nj in dirs:
                beta = _dot(xp, wk, wj) / nj
                wk = [w - beta * x for w, x in zip(wk, wj)]
                zk = [v - beta * x for v, x in zip(zk, zj)]
            nk = _dot(xp, wk, wk)
            if nk == 0.0:
                break
            step = _dot(xp, r, wk) / nk
            for c, v in zip(u, zk):
                c.interior = c.interior + step * v
            r = [ri - step * w for ri, w in zip(r, wk)]
            dirs.append((zk, wk, nk))
            stats.i += 1
            stats.resa = max(to_float(xp.max(xp.abs(ri))) for ri in r)
            if len(dirs) >= self.memory or stats.resa <= tolerance:
                dirs.clear()
                stats.resa, r = true_residual()
            if history is not None:
                history.append(stats.resa)
        stats.resa, _ = true_residual()
        u.fill_ghosts()
        if stats.resa > tolerance:
            import warnings  # noqa: PLC0415

            warnings.warn(f"{self.name} viscosity: residual {stats.resa:g} > tolerance {tolerance:g} after "
                          f"{stats.i} corrections", RuntimeWarning, stacklevel=2)
        return stats


class CCDViscosity(DCViscosity):
    """High-order implicit viscosity: ``A_H`` = CCD full stress (``split`` or ``layered``)."""

    name = "ccd"

    def __init__(self, ccd: CCDCellDerivative | None = None, *, form: str = "split", **kw) -> None:
        super().__init__(CCDStress(ccd, form), **kw)


class FD2Viscosity(DCViscosity):
    """2nd-order implicit viscosity with Basilisk's stencil structure (:class:`FD2Stress`),
    for mapped grids (on a uniform grid it converges in one or two corrections, ``A_L``
    being Basilisk's own operator)."""

    name = "fd2"

    def __init__(self, **kw) -> None:
        super().__init__(FD2Stress(), **kw)

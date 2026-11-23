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


def _dot(xp, x: list, y: list) -> float:
    return to_float(sum(xp.sum(a * b) for a, b in zip(x, y)))


class CCDViscosity:
    """High-order implicit viscosity by defect correction (``A_H`` = CCD full stress, ``A_L`` = Basilisk)."""

    name = "ccd"

    def __init__(self, ccd: CCDCellDerivative | None = None, *, form: str = "split", memory: int = 10,
                 inner_cycles: int = 2, maxiter: int = NITERMAX) -> None:
        self.stress = CCDStress(ccd, form)
        self.memory = memory
        self.inner_cycles = inner_cycles
        self.maxiter = maxiter

    def high(self, u: VectorField, mu_cell, rho, dt: float) -> list:
        """``A_H u = u - dt/rho L_H u``."""
        u.fill_ghosts()
        lap = self.stress(u, mu_cell)
        return [c.interior - dt / rho * l for c, l in zip(u, lap)]

    def _workspace(self, u: VectorField):
        """Correction fields with homogeneous conditions, kept with ``u`` (reuse of problems/graphs)."""
        ws = getattr(u, "_viscous_dc_workspace", None)
        homog = [c.bc.homogeneous() for c in u]
        if ws is None or ws[0].grid != u.grid or [c.bc for c in ws[0]] != homog:
            make = lambda name: VectorField([CellField(u.grid, bc=bc, name=f"{name}.{'xyz'[a]}")  # noqa: E731
                                             for a, bc in enumerate(homog)], name)
            ws = u._viscous_dc_workspace = (make("dc_work"), make("dc_z"))
        return ws

    def solve(self, u: VectorField, mu: FaceField, rho, dt: float, *, nrelax: int = 4,
              tolerance: float = TOLERANCE, history: list | None = None) -> MGStats:
        """Advance ``u`` in place; ``u``'s interior on entry is the right-hand side ``u^*``."""
        grid = u.grid
        xp = grid.xp
        mu_cell = cell_viscosity(mu)
        b = [c.interior.copy() for c in u]
        work, z = self._workspace(u)
        low = ViscosityProblem.reuse(z, mu, rho, dt)

        def true_residual():
            ah = self.high(u, mu_cell, rho, dt)
            r = [bi - ai for bi, ai in zip(b, ah)]
            return max(to_float(xp.max(xp.abs(ri))) for ri in r), r

        def apply_high0(v):
            for c, vi in zip(work, v):
                c.interior = vi
            return self.high(work, mu_cell, rho, dt)

        def precondition(r):
            for c in z:
                c.data[...] = 0.0
            mg_solve(list(z), r, low, nrelax=nrelax, tolerance=0.0, nitermax=self.inner_cycles,
                     nitermin=self.inner_cycles, warn=False)
            return [c.interior.copy() for c in z]

        stats = MGStats(nrelax=0, minlevel=low.levels[0].n[0], tolerance=tolerance)
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

            warnings.warn(f"CCD viscosity: residual {stats.resa:g} > tolerance {tolerance:g} after {stats.i} "
                          f"corrections", RuntimeWarning, stacklevel=2)
        return stats

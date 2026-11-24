"""Conservative level set (CLS) interface, TwoPhaseFlow's method (paper ch. 3, 5; ``src/twophase/levelset``).

The interface indicator is ``psi = H_eps(-phi) = (1 - tanh(phi / 2 eps)) / 2`` (``psi = 1`` in
the liquid, ``phi`` the signed distance, negative in the liquid; here level sets are
positive inside, ``psi = H_eps(Phi)``), ``eps = C_eps Delta`` with ``C_eps = 1.5`` (TwoPhaseFlow's
``epsilon_factor``). Per physical step (paper section 5b, stages A, B):

* **A, transport**: ``d_t psi + div(psi u_f) = 0`` in conservative form, TVD-RK3 (Shu--Osher),
  face flux ``(P_f psi) u_f`` with FCCD's face value ``P_f`` (4th order, central; TwoPhaseFlow's
  ``fccd_flux`` mode), the plain face difference ``D_f`` and ``psi`` clipped to ``[0, 1]`` after
  every stage (TwoPhaseFlow's ``clip_bounds``). ``u_f`` must be divergence-free for ``D_f``: with
  FCCD operators pass ``ops.wall_free_flux(uf, periodic)``.
* **B, mass closure**: ``psi += (M - sum psi dV) / sum w dV * w``, ``w = 4 psi (1 - psi)``
  (interface-localised), then clipped (TwoPhaseFlow ``apply_mass_correction``).

Re-initialisation: the paper's standard path is the Ridge--Eikonal distance reconstruction
(stage D, a later step here); it calls the Olsson--Kreiss compression--diffusion equation
``d_tau psi + div(psi (1 - psi) n) = div(eps grad psi)`` (``n = grad psi / |grad psi|``) the
continuous equilibrium only (not adopted), while TwoPhaseFlow's code defaults to it
(``SplitReinitializer``: explicit compression with a dissipative CCD divergence, Crank--Nicolson
ADI diffusion, 4 pseudo-steps, nodal grid). :meth:`CLSInterface.reinitialize` solves the same
equation in conservative finite-volume form on the cell-centred grid, explicitly: face fluxes
``psi(1 - psi) n`` (cell values averaged to the faces, ``n`` from the CCD gradient with
TwoPhaseFlow's floor ``|grad psi| >= 1e-6``) and ``eps`` times FCCD's face gradient, pseudo-step
``dtau = min(Delta^2 / (4 d eps), Delta / 2)`` (TwoPhaseFlow ``compute_dtau``), clipping and the
stage-B closure at the end; its face difference keeps the mass exactly.
"""

from __future__ import annotations

from ..backend import array_module
from ..discretization.ccd import CCDCellDerivative, FCCDFaceComplex
from ..fields import CellField, FaceField

EPSILON_FACTOR = 1.5


def psi_from_phi(phi, eps: float):
    """``H_eps``: ``(1 + tanh(phi / 2 eps)) / 2`` for a level set ``phi`` positive in the liquid."""
    xp = array_module(phi)
    return 0.5 * (1.0 + xp.tanh(phi / (2.0 * eps)))


def phi_from_psi(psi, eps: float, delta: float = 1e-6):
    """Inverse map (positive in the liquid), ``psi`` limited to ``[delta, 1 - delta]`` first."""
    xp = array_module(psi)
    p = xp.minimum(xp.maximum(psi, delta), 1.0 - delta)
    return 2.0 * eps * xp.arctanh(2.0 * p - 1.0)


def mass_correction(psi, dv, target):
    """Stage B (TwoPhaseFlow ``apply_mass_correction``): interface-weighted, then clipped."""
    xp = array_module(psi)
    m = xp.sum(psi * dv)
    w = 4.0 * psi * (1.0 - psi)
    W = xp.sum(w * dv)
    Ws = xp.where(W > 1e-12, W, 1.0)
    ratio = (xp.asarray(target) - m) / Ws
    gate = xp.where(W > 1e-12, 1.0, 0.0)
    return xp.clip(psi + (gate * ratio) * w, 0.0, 1.0)


class CLSInterface:
    """``psi`` (a :class:`CellField`, boundary conditions of a volume fraction) and its width ``eps``."""

    def __init__(self, psi: CellField, eps: float | None = None, *, ops: FCCDFaceComplex | None = None,
                 mass_closure: bool = True) -> None:
        grid = psi.grid
        if grid.ndim != 2 or not grid.is_uniform:
            raise NotImplementedError("CLS is implemented on uniform 2-D grids")
        self.psi = psi
        self.eps = EPSILON_FACTOR * float(min(grid.h)) if eps is None else float(eps)
        self.ops = ops or FCCDFaceComplex()
        self.ccd = self.ops.ccd if hasattr(self.ops, "ccd") else CCDCellDerivative()
        self.mass_closure = mass_closure
        self.dv = float(grid.h[0] * grid.h[1])

    @classmethod
    def from_level_set(cls, grid, phi_cells, bc, eps: float | None = None, **kw) -> "CLSInterface":
        """``psi = H_eps(phi)`` from cell-centred values of a signed distance positive in the liquid."""
        e = EPSILON_FACTOR * float(min(grid.h)) if eps is None else float(eps)
        return cls(CellField.from_interior(grid, psi_from_phi(phi_cells, e), bc), e, **kw)

    def mass(self) -> float:
        return float(self.psi.interior.sum()) * self.dv

    # -- stage A ---------------------------------------------------------------------------
    def _rhs(self, q, uf: FaceField):
        f = CellField.from_interior(self.psi.grid, q, self.psi.bc)
        f.fill_ghosts()
        grid = self.psi.grid
        out = None
        for ax in range(2):
            flux = self.ops.interp_f(f, ax) * uf[ax]
            n = grid.n[ax]
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
            d = (flux[tuple(hi)] - flux[tuple(lo)]) / float(grid.h[ax])
            out = -d if out is None else out - d
        return out

    def advect(self, uf: FaceField, dt: float) -> None:
        """One TVD-RK3 step with the face velocity ``uf`` (frozen), clipping each stage, then stage B."""
        xp = self.psi.grid.xp
        q0 = self.psi.interior
        m0 = xp.sum(q0) * self.dv
        q1 = xp.clip(q0 + dt * self._rhs(q0, uf), 0.0, 1.0)
        q2 = xp.clip(0.75 * q0 + 0.25 * (q1 + dt * self._rhs(q1, uf)), 0.0, 1.0)
        q3 = xp.clip(q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * self._rhs(q2, uf)), 0.0, 1.0)
        if self.mass_closure:
            q3 = mass_correction(q3, self.dv, m0)
        self.psi.interior = q3
        self.psi.fill_ghosts()

    # -- re-initialisation (compression--diffusion, provisional) -----------------------------------
    def reinitialize(self, n_steps: int = 4) -> None:
        grid = self.psi.grid
        xp = grid.xp
        h = float(min(grid.h))
        dtau = min(0.5 * h * h / (2.0 * 2 * self.eps), 0.5 * h)
        q = self.psi.interior
        m0 = xp.sum(q) * self.dv
        for _ in range(n_steps):
            f = CellField.from_interior(grid, q, self.psi.bc)
            f.fill_ghosts()
            grads = [self.ccd.d1(f, ax) for ax in range(2)]
            norm = xp.maximum(xp.sqrt(xp.maximum(grads[0] ** 2 + grads[1] ** 2, 1e-12)), 1e-6)
            comp = CellField.from_interior(grid, q * (1.0 - q), self.psi.bc)
            comp.fill_ghosts()
            div = None
            for ax in range(2):
                nf = CellField.from_interior(grid, grads[ax] / norm, _odd_bc(self.psi.bc, ax))
                nf.fill_ghosts()
                flux = (self.ops.interp_f(comp, ax) * self.ops.interp_f(nf, ax)
                        - self.eps * self.ops.grad_f(f, ax))
                flux = _zero_walls(flux, ax, self.psi.bc)
                n = grid.n[ax]
                lo = [slice(None)] * 2
                hi = [slice(None)] * 2
                lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
                d = (flux[tuple(hi)] - flux[tuple(lo)]) / float(grid.h[ax])
                div = d if div is None else div + d
            q = xp.clip(q - dtau * div, 0.0, 1.0)
        if self.mass_closure:
            q = mass_correction(q, self.dv, m0)
        self.psi.interior = q
        self.psi.fill_ghosts()


def _odd_bc(bc, axis):
    """Boundary conditions of a normal component: periodic stays, walls normal to ``axis`` antisymmetric."""
    from ..boundary import BoundaryConditions, Dirichlet, Periodic  # noqa: PLC0415

    out = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            cond = bc.get(ax, side)
            out.set((ax, side), Periodic() if cond.periodic else (Dirichlet(0.0) if ax == axis else cond))
    return out


def _zero_walls(flux, axis, bc):
    """No flux through walls (the reinitialisation conserves psi exactly)."""
    if bc.get(axis, 0).periodic:
        return flux
    idx0 = [slice(None)] * 2
    idx1 = [slice(None)] * 2
    idx0[axis], idx1[axis] = 0, -1
    flux = flux.copy()
    flux[tuple(idx0)] = 0.0
    flux[tuple(idx1)] = 0.0
    return flux

"""UCCD6 momentum convection: skew-symmetric CCD with order-preserving hyperviscosity.

TwoPhaseFlow (paper §4.5 ``04d_uccd6.tex``, eq. ``uccd6_semidiscrete``; NS form in
``src/twophase/ns_terms/uccd6_convection.py``), for every component ``u_j``::

    -C_j = sum_k [ -1/2 ( u_k D1_k u_j + D1_k (u_k u_j) )
                   - sigma max|u_k| h_k^7 (D2_k)^4 u_j ]

``D1``, ``D2`` are the cell-centred CCD operators of :class:`CCDCellDerivative`
(6th order). ``D1 u_j`` and the first ``D2 u_j`` use the velocity's boundary
conditions; the product ``u_k u_j`` takes the product of the wall values where
both components are Dirichlet there (no-slip walls) and the extrapolating
closure elsewhere; the repeated second derivatives have no boundary data and use
the extrapolating closure. ``(-D2)^4 = D2^4``; the hyperviscosity acts as ``-sigma |u| h^7 d^8/dx^8``
on resolved modes and damps the Nyquist mode (``omega_2^8`` symbol), so the
interior stays 6th order.

On a periodic grid the skew part conserves the discrete kinetic energy exactly
(``D1`` is skew-symmetric) and the hyperviscosity only removes it.
``sigma = 1e-3`` is TwoPhaseFlow's default (``uccd6_sigma``).

Explicit time stepping: the hyperviscosity has the real eigenvalue
``-sigma |u| omega_2(pi)^8 / h`` at the Nyquist mode (``omega_2(pi)^2 = 96/10``,
``omega_2^8 = 8493.5``). The EXT2/BDF2 convection of :class:`IMEXBDF2NS` is
stable on the negative real axis for ``lambda dt >= -4/3``, so
``sigma |u| dt / h <= 4/3 / 8493.5``: CFL ``<= 0.157`` with ``sigma = 1e-3``
(paper remark ``rem:uccd6_explicit_cfl``). :meth:`max_stable_dt` returns that
bound and the stepper respects it.
"""

from __future__ import annotations

from ...discretization.ccd import CCDCellDerivative
from ...fields import FaceField, VectorField

OMEGA2_SQ_NYQUIST = 96.0 / 10.0  # CCD 2nd-derivative symbol at theta = pi (Chu & Fan closed form)
EXT2_REAL_LIMIT = 4.0 / 3.0      # EXT2/BDF2 stability on the negative real axis, constant step


class UCCD6Convection:
    name = "uccd6"

    def __init__(self, ccd: CCDCellDerivative | None = None, sigma: float = 1.0e-3) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.ccd = ccd or CCDCellDerivative()
        self.sigma = float(sigma)

    def max_stable_dt(self, u: VectorField, uf: FaceField | None = None) -> float:
        """Largest step for which the explicit (EXT2) hyperviscosity stays stable."""
        grid = u.grid
        xp = grid.xp
        rate = sum(self.sigma * float(xp.max(xp.abs(u[k].interior))) * OMEGA2_SQ_NYQUIST ** 4 / grid.h[k]
                   for k in range(grid.ndim))
        return EXT2_REAL_LIMIT / rate if rate > 0.0 else float("inf")

    def hyperviscosity(self, comp, axis: int):
        """``(D2)^4`` of one velocity component along ``axis`` (first ``D2`` with its BC)."""
        grid = comp.grid
        periodic = comp.bc.is_periodic(axis)
        d = self.ccd.d2(comp, axis)
        for _ in range(3):
            d = self.ccd.d2_array(d, grid, axis, periodic)
        return d

    def tendency(self, u: VectorField, uf: FaceField | None = None) -> list:
        """``-C(u)`` for every component (interior arrays); ``uf`` is not used."""
        grid = u.grid
        xp = grid.xp
        u.fill_ghosts()
        vel = [c.interior for c in u]
        out = []
        for comp in u:
            acc = None
            for k in range(grid.ndim):
                d1 = self.ccd.d1(comp, k)
                cons = self.ccd.d1_product(u[k], comp, k)
                term = -0.5 * (vel[k] * d1 + cons)
                coeff = self.sigma * grid.h[k] ** 7
                term = term - coeff * xp.max(xp.abs(vel[k])) * self.hyperviscosity(comp, k)
                acc = term if acc is None else acc + term
            out.append(acc)
        return out

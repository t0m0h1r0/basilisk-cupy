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

    def __init__(self, ccd: CCDCellDerivative | None = None, sigma: float = 1.0e-3, implicit: bool = False) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.ccd = ccd or CCDCellDerivative()
        self.sigma = float(sigma)
        #: move the hyperviscosity to the implicit part of the stepper (:class:`ImplicitHyperviscosity`)
        self.implicit = implicit

    def implicit_part(self, u: VectorField) -> "ImplicitHyperviscosity":
        """The hyperviscosity with its coefficients frozen at ``u`` (one step)."""
        grid = u.grid
        xp = grid.xp
        coeffs = [self.sigma * grid.h[k] ** 7 * float(xp.max(xp.abs(u[k].interior))) for k in range(grid.ndim)]
        return ImplicitHyperviscosity(self, coeffs, grid.h)

    def max_stable_dt(self, u: VectorField, uf: FaceField | None = None) -> float:
        """Largest step for which the explicit (EXT2) hyperviscosity stays stable."""
        if self.implicit:
            return float("inf")
        grid = u.grid
        xp = grid.xp
        rate = sum(self.sigma * float(xp.max(xp.abs(u[k].interior))) * OMEGA2_SQ_NYQUIST ** 4 / grid.h[k]
                   for k in range(grid.ndim))
        return EXT2_REAL_LIMIT / rate if rate > 0.0 else float("inf")

    def spectrum(self, u: VectorField, uf: FaceField | None = None) -> tuple[float, float]:
        """``(max |Im lambda|, max |Re lambda|)`` in 1/time: CCD ``D1`` and the hyperviscosity."""
        from .spectra import advective_rate  # noqa: PLC0415

        grid = u.grid
        xp = grid.xp
        real = 0.0 if self.implicit else sum(
            self.sigma * float(xp.max(xp.abs(u[k].interior))) * OMEGA2_SQ_NYQUIST ** 4 / grid.h[k]
            for k in range(grid.ndim))
        return advective_rate(u, None, "ccd"), real

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
        if not grid.is_uniform:
            raise NotImplementedError(f"{self.name} on a non-uniform grid is not implemented yet (step 4f-3)")
        u.fill_ghosts()
        vel = [c.interior for c in u]
        out = []
        for comp in u:
            acc = None
            for k in range(grid.ndim):
                d1 = self.ccd.d1(comp, k)
                cons = self.ccd.d1_product(u[k], comp, k)
                term = -0.5 * (vel[k] * d1 + cons)
                if not self.implicit:
                    coeff = self.sigma * grid.h[k] ** 7
                    term = term - coeff * xp.max(xp.abs(vel[k])) * self.hyperviscosity(comp, k)
                acc = term if acc is None else acc + term
            out.append(acc)
        return out


class ImplicitHyperviscosity:
    """``H u = sum_k c_k (D2_k)^4 u`` with frozen ``c_k = sigma max|u_k| h_k^7``, for an implicit stage
    ``(I - tau V + tau H) u = r``.

    Every 1-D factor ``(D2)^4`` (first ``D2`` with the component's closure, the others
    extrapolating) has a real spectrum, up to ``omega_2(pi)^8 = 8493.5`` over ``h^8``. With the
    compact wall closures one wall eigenvalue is slightly negative (measured ``-1.66 / h^8``
    Dirichlet, ``-0.020 / h^8`` Neumann, independent of n): a growth rate ``1.7 sigma |u| / h``,
    2e-4 of the Nyquist damping (the previous wide closures gave the advection ``-D1`` itself
    ``Re lambda = 0.096 |u| / h`` on Dirichlet walls; compact: 1e-15). Making the factor
    non-negative needs either the field's closure in all four factors (1st order at the wall,
    measured) or ``(E B)^T (E B)`` (3rd order); neither was adopted. The axes commute, and
    :meth:`precondition` solves ``(I + tau H) z = r`` by fast diagonalization (per-axis eigenvectors). With walls the
    eigenvector matrices are non-normal (condition ~5e5), so this is used as a preconditioner
    of the GCR solve in :class:`~basilisk_cupy.terms.viscous.DCViscosity`, whose residual applies
    ``H`` exactly.
    """

    def __init__(self, conv: UCCD6Convection, coeffs: list, h: tuple) -> None:
        self.conv = conv
        self.coeffs = coeffs
        self.h = h
        self._dev: dict = {}

    def apply(self, u: VectorField) -> list:
        u.fill_ghosts()
        xp = u.grid.xp
        out = []
        for comp in u:
            acc = xp.zeros_like(comp.interior)
            for k, c in enumerate(self.coeffs):
                if c:
                    acc = acc + c * self.conv.hyperviscosity(comp, k)
            out.append(acc)
        return out

    def _factors(self, xp, n: int, h: float, lo: str, hi: str):
        key = (xp.__name__, n, h, lo, hi)
        hit = self._dev.get(key)
        if hit is None:
            lam, V, Vi = _hyperviscosity_eigen(n, h, lo, hi)
            hit = self._dev[key] = (xp.asarray(lam), xp.asarray(V), xp.asarray(Vi))
        return hit

    def precondition(self, r: list, tau: float, bcs: list) -> list:
        """Approximate ``(I + tau H)^{-1} r`` for each component (homogeneous conditions ``bcs``)."""
        from ...backend import array_module  # noqa: PLC0415
        from ...boundary import LOWER, UPPER  # noqa: PLC0415
        from ...discretization.ccd import _kind  # noqa: PLC0415

        out = []
        for ri, bc in zip(r, bcs):
            xp = array_module(ri)
            z = ri.astype(xp.complex128)
            denom = 1.0
            factors = []
            for k, c in enumerate(self.coeffs):
                lam, V, Vi = self._factors(xp, ri.shape[k], float(self.h[k]), _kind(bc.get(k, LOWER)),
                                           _kind(bc.get(k, UPPER)))
                factors.append((V, Vi))
                shape = [1] * ri.ndim
                shape[k] = -1
                denom = denom + tau * c * lam.reshape(shape)
                z = xp.moveaxis(xp.moveaxis(z, k, -1) @ Vi.T, -1, k)
            z = z / denom
            for k, (V, _) in enumerate(factors):
                z = xp.moveaxis(xp.moveaxis(z, k, -1) @ V.T, -1, k)
            out.append(z.real.copy())
        return out


def _hyperviscosity_eigen(n: int, h: float, lo: str, hi: str):
    """Eigen-decomposition of the 1-D ``(D2)^4`` (``M = V diag(lam) V^-1``), host arrays."""
    import numpy as np  # noqa: PLC0415

    from ...discretization.ccd import EXTRAPOLATE, PERIODIC, ccd_matrices  # noqa: PLC0415

    d2_bc = ccd_matrices(n, h, lo, hi)[1]
    rep = PERIODIC if lo == PERIODIC else EXTRAPOLATE
    d2 = ccd_matrices(n, h, rep, rep)[1]
    lam, V = np.linalg.eig(d2 @ d2 @ d2 @ d2_bc)
    return lam, V, np.linalg.inv(V)

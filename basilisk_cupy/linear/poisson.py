"""Poisson--Helmholtz solver ``div(alpha grad a) + lambda a = b`` (Basilisk ``poisson()``).

``alpha`` is a face field (e.g. the specific volume 1/rho on faces), ``lambda``
a cell field or scalar. The residual is evaluated with the supplied
:class:`~basilisk_cupy.discretization.FaceComplex` (``L_high``); relaxation
always uses the compact 2nd-order operator (``L_low``), on GPUs with
red/black Gauss--Seidel as Basilisk does (``GAUSS_SEIDEL || _GPU``).
With ``ops = fd2`` this is exactly Basilisk's solver; with a higher-order
``ops`` it becomes a defect-correction iteration.
"""

from __future__ import annotations

from ..backend import to_float
from ..boundary import BoundaryConditions, Dirichlet
from ..discretization.base import FaceComplex
from ..discretization.fd2 import FD2FaceComplex
from ..fields import CellField, FaceField
from ..grid.hierarchy import build_levels
from ..kernels.relax_gpu import relax_redblack
from ..kernels.residual_gpu import helmholtz_residual
from .multigrid import NITERMAX, NITERMIN, TOLERANCE, MGStats, mg_solve, restrict_cell, restrict_face


class HelmholtzProblem:
    """Multigrid problem for ``div(alpha grad a) + lambda a = b``."""

    def __init__(self, a: CellField, alpha: FaceField | None, lam=None, *,
                 ops: FaceComplex | None = None, min_cells: int = 2) -> None:
        grid = a.grid
        xp = grid.xp
        self.ops = ops or FD2FaceComplex()
        self.min_cells = min_cells
        self.levels = build_levels(grid, min_cells=min_cells)
        self.correction_bcs: list[BoundaryConditions] = [a.bc.homogeneous()]
        self.alpha_fine, self.alpha, self.lam, self.has_lambda = self._coefficients(grid, alpha, lam)
        # GPU: one fused kernel per colour (kernels/relax_gpu.py); otherwise red/black masks.
        # Mapped (non-uniform) grids use the array path with finite-volume coefficients.
        self.fused = xp.__name__ == "cupy" and grid.dtype == "float64" and grid.is_uniform
        self._masks: list = []

    def _coefficients(self, grid, alpha, lam):
        """alpha and lambda on every level (Basilisk: ``restriction ({alpha,lambda})``)."""
        xp = grid.xp
        if alpha is None:
            alpha = FaceField.constant(grid, 1.0, name="unity")
        alphas = [list(alpha.components)]
        for _ in range(len(self.levels) - 1):
            alphas.append([restrict_face(c, ax) for ax, c in enumerate(alphas[-1])])
        # own copies (also of the finest level): reuse() overwrites them in place
        alphas = [[xp.array(c, dtype=grid.dtype, order="C", copy=True) for c in lev] for lev in alphas[::-1]]
        if lam is None or (not hasattr(lam, "shape") and float(lam) == 0.0):
            return alpha, alphas, [0.0] * len(self.levels), False
        if not hasattr(lam, "shape"):
            lam = xp.full(grid.n, float(lam), dtype=grid.dtype)
        lams = [lam]
        for _ in range(len(self.levels) - 1):
            lams.append(restrict_cell(lams[-1]))
        return alpha, alphas, [xp.array(v, dtype=grid.dtype, order="C", copy=True) for v in lams[::-1]], True

    @classmethod
    def reuse(cls, a: CellField, alpha, lam=None, *, ops=None, min_cells: int = 2) -> "HelmholtzProblem":
        """The problem built earlier for the unknown ``a`` with new coefficients, or a new one.

        On the GPU the coefficients are copied into the earlier problem's
        arrays, so the multigrid cycles it recorded as CUDA graphs
        (:mod:`.cycle_graph`) stay valid. Elsewhere a new problem is built.
        """
        old = getattr(a, "_helmholtz_problem", None)
        if (old is None or not old.fused or old.min_cells != min_cells or old.levels[-1] != a.grid
                or old.correction_bcs != [a.bc.homogeneous()]):
            problem = cls(a, alpha, lam, ops=ops, min_cells=min_cells)
            if problem.fused:
                a._helmholtz_problem = problem
            return problem
        alpha_fine, alphas, lams, has_lambda = old._coefficients(a.grid, alpha, lam)
        if has_lambda != old.has_lambda:
            problem = cls(a, alpha, lam, ops=ops, min_cells=min_cells)
            a._helmholtz_problem = problem
            return problem
        for dst_lev, src_lev in zip(old.alpha, alphas):
            for dst, src in zip(dst_lev, src_lev):
                dst[...] = src
        if has_lambda:
            for dst, src in zip(old.lam, lams):
                dst[...] = src
        old.alpha_fine = alpha_fine
        old.ops = ops or FD2FaceComplex()
        return old

    @property
    def masks(self) -> list:
        """Red/black masks per level (array path only; built on first use)."""
        if not self._masks:
            for g in self.levels:
                idx = g.xp.indices(g.n).sum(axis=0)
                self._masks.append(((idx % 2) == 0, (idx % 2) == 1))
        return self._masks

    # L_high residual on the finest level -------------------------------------------
    def device_residual_supported(self) -> bool:
        """Whether :meth:`device_residual` applies (GPU kernels and the fd2 residual operator)."""
        return self.fused and type(self.ops) is FD2FaceComplex

    def device_residual(self, al: list[CellField], bl: list, resl: list, norm) -> None:
        """Residual into ``resl`` and ``max |res|`` into the device scalar ``norm`` (no sync,
        no allocation: kernels/residual_gpu.py). ``bl`` must be contiguous float64."""
        a = al[0]
        grid = a.grid
        helmholtz_residual(a.data, bl[0], resl[0], self.alpha[-1], self.lam[-1] if self.has_lambda else None,
                           grid.n, grid.h, grid.ghosts, norm)

    def residual(self, al: list[CellField], bl: list, resl: list) -> float:
        a, b, res = al[0], bl[0], resl[0]
        xp = a.grid.xp
        if self.device_residual_supported() and res.flags.c_contiguous:
            norm = xp.zeros(1, dtype=xp.uint64)
            self.device_residual(al, [xp.ascontiguousarray(b, dtype=xp.float64)], resl, norm)
            return to_float(norm.view(xp.float64)[0])
        flux = FaceField(a.grid, [self.alpha_fine[ax] * self.ops.grad_f(a, ax) for ax in range(a.grid.ndim)])
        res[...] = b - self.ops.div_f(flux, periodic=tuple(a.bc.is_periodic(k) for k in range(a.grid.ndim)))
        if self.has_lambda:
            res -= self.lam[-1] * a.interior
        return to_float(xp.max(xp.abs(res)))

    # L_low relaxation (5/7-point stencil, red/black Gauss--Seidel) -------------------
    def relax(self, dal: list[CellField], resl: list, level: int) -> None:
        da, res = dal[0], resl[0]
        grid = self.levels[level]
        xp = grid.xp
        alpha = self.alpha[level]
        lam = self.lam[level]
        if self.fused:
            relax_redblack(da.data, res, alpha, lam if self.has_lambda else None, grid.n, grid.h, grid.ghosts)
            return
        if not grid.is_uniform:
            self._relax_lines(da, res, level)
            return
        for mask in self.masks[level]:
            num = -res
            den = -lam if self.has_lambda else 0.0
            for ax in range(grid.ndim):
                n = grid.n[ax]
                sl_lo = [slice(None)] * grid.ndim
                sl_hi = [slice(None)] * grid.ndim
                sl_lo[ax] = slice(0, n)
                sl_hi[ax] = slice(1, n + 1)
                a_lo = alpha[ax][tuple(sl_lo)]
                a_hi = alpha[ax][tuple(sl_hi)]
                h2 = grid.h[ax] ** 2
                num = num + (a_hi * da.data[grid.shifted(ax, 1)] + a_lo * da.data[grid.shifted(ax, -1)]) / h2
                den = den + (a_hi + a_lo) / h2
            interior = da.data[grid.interior()]
            new = xp.where(den != 0, num / xp.where(den != 0, den, 1.0), 0.0)
            da.data[grid.interior()] = xp.where(mask, new, interior)


    # mapped grids: finite-volume coefficients and zebra line relaxation ----------------------
    def _fv_coefficients(self, level: int):
        """``(c_lo, c_hi)`` per axis with ``L a = sum c_hi (a_+ - a) - c_lo (a - a_-)``:
        ``alpha_face / (centre distance * cell width)`` (``alpha / h^2`` on a uniform axis)."""
        grid = self.levels[level]
        alpha = self.alpha[level]
        out = []
        for ax in range(grid.ndim):
            n = grid.n[ax]
            lo = [slice(None)] * grid.ndim
            hi = [slice(None)] * grid.ndim
            lo[ax] = slice(0, n)
            hi[ax] = slice(1, n + 1)
            a_lo, a_hi = alpha[ax][tuple(lo)], alpha[ax][tuple(hi)]
            if grid.axis_uniform(ax):
                h2 = grid.h[ax] ** 2
                out.append((a_lo / h2, a_hi / h2))
            else:
                m = grid.metric(ax)
                width = m.along(m.width)
                out.append((a_lo / (m.along(m.spacing[:n]) * width), a_hi / (m.along(m.spacing[1:]) * width)))
        return out

    def _relax_lines(self, da: CellField, res, level: int) -> None:
        """Alternating zebra line Gauss--Seidel (lines along every wall axis, one tridiagonal solve per line).

        Stretched cells are anisotropic (aspect ratio > 3 near clustered walls), where the
        point red/black smoother of Basilisk stalls (measured: 11 cycles on a uniform 64^2
        grid, 52 with tanh(beta=2) on both axes); solving whole lines restores it. The
        correction's homogeneous conditions enter the end rows (ghost = -a for Dirichlet,
        +a for Neumann). Lines of one colour do not touch each other, so they are solved together.
        """
        grid = self.levels[level]
        xp = grid.xp
        ndim = grid.ndim
        coef = self._fv_coefficients(level)
        den = sum(c_lo + c_hi for c_lo, c_hi in coef)
        if self.has_lambda:
            den = den - self.lam[level]
        bc = self.correction_bcs[0]
        line_axes = [ax for ax in range(ndim) if not bc.is_periodic(ax)] or [None]
        interior = grid.interior()
        for ax in line_axes:
            others = [b for b in range(ndim) if b != ax]
            if others:
                idx = xp.indices(grid.n)
                parity = sum(idx[b] for b in others) % 2
            for colour in (0, 1):
                da.fill_ghosts()
                rhs = -res
                for b in (others if ax is not None else range(ndim)):
                    c_lo, c_hi = coef[b]
                    rhs = rhs + c_hi * da.data[grid.shifted(b, 1)] + c_lo * da.data[grid.shifted(b, -1)]
                if ax is None:  # fully periodic: point relaxation (Jacobi on one colour)
                    idx = xp.indices(grid.n)
                    mask = (sum(idx) % 2) == colour
                    new = xp.where(den != 0, rhs / xp.where(den != 0, den, 1.0), 0.0)
                    da.data[interior] = xp.where(mask, new, da.data[interior])
                    continue
                c_lo, c_hi = coef[ax]
                s_lo = -1.0 if isinstance(bc.get(ax, 0), Dirichlet) else 1.0
                s_hi = -1.0 if isinstance(bc.get(ax, 1), Dirichlet) else 1.0
                new = _solve_lines(xp, ax, -c_lo, den, -c_hi, rhs, s_lo, s_hi)
                if others:
                    da.data[interior] = xp.where(parity == colour, new, da.data[interior])
                else:
                    da.data[interior] = new
                    break


def _solve_lines(xp, axis: int, lower, diag, upper, rhs, s_lo: float, s_hi: float):
    """Thomas algorithm along ``axis`` for every line at once; the end rows fold the ghost
    reflection ``a_ghost = s a_first`` into the diagonal."""
    mv = lambda v: xp.moveaxis(xp.broadcast_to(v, rhs.shape), axis, 0)  # noqa: E731
    lo, d, up, r = mv(lower), mv(diag).copy(), mv(upper), mv(rhs)
    n = r.shape[0]
    d[0] = d[0] + lo[0] * s_lo
    d[n - 1] = d[n - 1] + up[n - 1] * s_hi
    cp = xp.empty_like(r)
    rp = xp.empty_like(r)
    cp[0] = up[0] / d[0]
    rp[0] = r[0] / d[0]
    for i in range(1, n):
        m = d[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / m
        rp[i] = (r[i] - lo[i] * rp[i - 1]) / m
    x = xp.empty_like(r)
    x[n - 1] = rp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = rp[i] - cp[i] * x[i + 1]
    return xp.moveaxis(x, 0, axis)


def poisson(a: CellField, b, alpha: FaceField | None = None, lam=None, *,
            ops: FaceComplex | None = None, tolerance: float = TOLERANCE, nrelax: int = 4,
            min_cells: int = 2, nitermax: int = NITERMAX, nitermin: int = NITERMIN,
            method: str = "auto", memory: int = 10) -> MGStats:
    """Solve ``div(alpha grad a) + lambda a = b`` in place (``a`` is the initial guess).

    ``method``: ``"mg"`` runs Basilisk's multigrid with the ``ops`` residual
    (plain defect correction when ``ops`` is not fd2); ``"dc"`` runs the
    GCR-accelerated defect correction of :mod:`.defect_correction` with
    ``memory`` directions (1 = TwoPhaseFlow's line-search correction);
    ``"auto"`` picks ``"mg"`` for fd2 and ``"dc"`` otherwise.
    """
    ops = ops or FD2FaceComplex()
    if method == "auto":
        method = "mg" if ops.name == "fd2" else "dc"
    if method == "dc":
        from .defect_correction import defect_correction  # noqa: PLC0415

        stats = defect_correction(a, b, alpha, lam, ops=ops, tolerance=tolerance, maxiter=nitermax,
                                  memory=memory, nrelax=nrelax, min_cells=min_cells)
        if stats.resa > tolerance:
            import warnings  # noqa: PLC0415

            warnings.warn(f"defect correction: res {stats.resa:g} > tolerance {tolerance:g} after "
                          f"{stats.i} corrections", RuntimeWarning, stacklevel=2)
        return stats
    if method != "mg":
        raise ValueError(f"unknown method {method!r}")
    problem = HelmholtzProblem.reuse(a, alpha, lam, ops=ops, min_cells=min_cells)
    return mg_solve(a, b, problem, nrelax=nrelax, tolerance=tolerance,
                    nitermax=nitermax, nitermin=nitermin)

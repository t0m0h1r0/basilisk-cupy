"""Implicit viscous diffusion (port of Basilisk ``viscosity.h``, Cartesian, non-AXI).

Solves, with backward Euler in time,

    u^{n+1} - dt/rho div(2 mu D(u^{n+1})) = u^n

for all velocity components at once (they are coupled through the
transpose part of the stress). ``mu`` is a face field, ``rho`` a cell
field. Residual and relaxation reproduce Basilisk's "naive" Cartesian
discretisation, including the frozen copy ``ua`` used for the corner
terms of the red/black relaxation. Square cells are assumed, as in
Basilisk (one ``Delta``).
"""

from __future__ import annotations

from ..backend import to_float
from ..fields import CellField, FaceField, VectorField
from ..grid.hierarchy import build_levels
from ..kernels.residual_gpu import viscosity_residual
from ..kernels.viscosity_gpu import relax_viscosity_redblack
from .multigrid import NITERMAX, NITERMIN, TOLERANCE, MGStats, mg_solve, restrict_cell, restrict_face
from .lexicographic import lexicographic
from .lexicographic import relax_viscosity as relax_viscosity_lex


def _face_lo_hi(face, axis: int, n: int):
    lo = [slice(None)] * face.ndim
    hi = [slice(None)] * face.ndim
    lo[axis] = slice(0, n)
    hi[axis] = slice(1, n + 1)
    return face[tuple(lo)], face[tuple(hi)]


class ViscosityProblem:
    def __init__(self, u: VectorField, mu: FaceField, rho, dt: float, min_cells: int = 2) -> None:
        grid = u.grid
        if not grid.is_uniform:
            raise NotImplementedError("Basilisk viscosity() needs a uniform grid (single Delta); on a mapped grid "
                                      "use CCDViscosity / FD2Viscosity (terms/viscous.py)")
        if len(set(round(h, 14) for h in grid.h)) != 1:
            raise ValueError("viscosity(): square cells are required (Basilisk uses a single Delta)")
        xp = grid.xp
        self.min_cells = min_cells
        self.levels = build_levels(grid, min_cells=min_cells)
        self.correction_bcs = [c.bc.homogeneous() for c in u]
        # GPU: one fused kernel per colour (kernels/viscosity_gpu.py); otherwise red/black masks.
        self.fused = xp.__name__ == "cupy" and grid.dtype == "float64"
        self.dt = dt
        self.mu, self.rho = self._coefficients(mu, rho)
        self._masks: list = []
        if self.fused:  # the frozen copies and [dt, rho] live on the device once (CUDA graphs)
            self.frozen = [[g.zeros_cell() for _ in range(grid.ndim)] for g in self.levels]
            self.params = xp.zeros(2, dtype=xp.float64)
            self._set_params()

    def _coefficients(self, mu: FaceField, rho):
        xp = self.levels[-1].xp
        mus = [list(mu.components)]
        for _ in range(len(self.levels) - 1):
            mus.append([restrict_face(c, ax) for ax, c in enumerate(mus[-1])])
        # own copies (also of the finest level): reuse() overwrites them in place
        mus = [[xp.array(c, dtype=xp.float64, order="C", copy=True) for c in lev] for lev in mus[::-1]]
        if hasattr(rho, "shape"):
            rhos = [rho]
            for _ in range(len(self.levels) - 1):
                rhos.append(restrict_cell(rhos[-1]))
            return mus, [xp.array(r, dtype=xp.float64, order="C", copy=True) for r in rhos[::-1]]
        return mus, [float(rho)] * len(self.levels)

    def _set_params(self) -> None:
        xp = self.levels[-1].xp
        rho = self.rho[-1] if not hasattr(self.rho[-1], "shape") else 0.0
        self.params[...] = xp.asarray([self.dt, rho], dtype=xp.float64)

    @classmethod
    def reuse(cls, u: VectorField, mu: FaceField, rho, dt: float, min_cells: int = 2) -> "ViscosityProblem":
        """The problem built earlier for ``u`` with new ``mu``, ``rho``, ``dt``, or a new one.

        On the GPU the coefficients are copied into the earlier problem's
        arrays (and dt, rho into its device parameters), so its recorded CUDA
        graphs stay valid. Elsewhere a new problem is built.
        """
        old = getattr(u, "_viscosity_problem", None)
        if (old is None or not old.fused or old.min_cells != min_cells or old.levels[-1] != u.grid
                or old.correction_bcs != [c.bc.homogeneous() for c in u]
                or hasattr(rho, "shape") != hasattr(old.rho[-1], "shape")):
            problem = cls(u, mu, rho, dt, min_cells=min_cells)
            if problem.fused:
                u._viscosity_problem = problem
            return problem
        mus, rhos = old._coefficients(mu, rho)
        for dst_lev, src_lev in zip(old.mu, mus):
            for dst, src in zip(dst_lev, src_lev):
                dst[...] = src
        if hasattr(rho, "shape"):
            for dst, src in zip(old.rho, rhos):
                dst[...] = src
        else:
            old.rho = rhos
        old.dt = dt
        old._set_params()
        return old

    @property
    def masks(self) -> list:
        """Red/black masks per level (array path only; built on first use)."""
        if not self._masks:
            for g in self.levels:
                idx = g.xp.indices(g.n).sum(axis=0)
                self._masks.append(((idx % 2) == 1, (idx % 2) == 0))  # Basilisk: odd i + j first
        return self._masks

    # Basilisk residual_viscosity ("naive" Cartesian version) ------------------------
    def device_residual_supported(self) -> bool:
        return self.fused

    def device_residual(self, a: list[CellField], b: list, res: list, norm) -> None:
        """Residual into ``res`` and ``max |res|`` into the device scalar ``norm`` (no sync,
        no allocation: kernels/residual_gpu.py). ``b`` must be contiguous float64."""
        grid = a[0].grid
        viscosity_residual([c.data for c in a], b, res, self.mu[-1], self.rho[-1], self.params, grid.n,
                           grid.h[0] ** 2, grid.ghosts, norm)

    def residual(self, a: list[CellField], b: list, res: list) -> float:
        grid = a[0].grid
        xp = grid.xp
        if self.device_residual_supported() and all(r.flags.c_contiguous for r in res):
            norm = xp.zeros(1, dtype=xp.uint64)
            self.device_residual(a, [xp.ascontiguousarray(v, dtype=xp.float64) for v in b], res, norm)
            return to_float(norm.view(xp.float64)[0])
        ndim = grid.ndim
        h2 = grid.h[0] ** 2
        mu = self.mu[-1]
        maxres = 0.0
        for i in range(ndim):
            ui = a[i].data
            u0 = ui[grid.at()]
            acc = 0.0
            for j in range(ndim):
                mlo, mhi = _face_lo_hi(mu[j], j, grid.n[j])
                if j == i:
                    acc = acc + 2.0 * mhi * (ui[grid.at({i: 1})] - u0) - 2.0 * mlo * (u0 - ui[grid.at({i: -1})])
                else:
                    uj = a[j].data
                    acc = acc + mhi * (ui[grid.at({j: 1})] - u0
                                       + (uj[grid.at({i: 1})] + uj[grid.at({i: 1, j: 1})]) / 4.0
                                       - (uj[grid.at({i: -1})] + uj[grid.at({i: -1, j: 1})]) / 4.0)
                    acc = acc - mlo * (u0 - ui[grid.at({j: -1})]
                                       + (uj[grid.at({i: 1, j: -1})] + uj[grid.at({i: 1})]) / 4.0
                                       - (uj[grid.at({i: -1, j: -1})] + uj[grid.at({i: -1})]) / 4.0)
            res[i][...] = b[i] - u0 + self.dt / self.rho[-1] * acc / h2
            maxres = max(maxres, to_float(xp.max(xp.abs(res[i]))))
        return maxres

    # Basilisk relax_viscosity, red/black variant (GAUSS_SEIDEL || _GPU) -------------------
    def relax(self, da: list[CellField], res: list, level: int) -> None:
        grid = self.levels[level]
        xp = grid.xp
        ndim = grid.ndim
        h2 = grid.h[0] ** 2
        mu = self.mu[level]
        rho = self.rho[level]
        dt = self.dt
        if lexicographic() and grid.is_uniform and ndim == 2 and grid.h[0] == grid.h[1]:
            relax_viscosity_lex(xp, [c.data for c in da], res, mu, rho, dt, grid.n, float(grid.h[0]), grid.ghosts)
            return
        if self.fused:
            relax_viscosity_redblack([c.data for c in da], res, mu, rho, grid.n, h2, self.params, grid.ghosts,
                                     frozen=self.frozen[level])
            return
        ua = [c.data.copy() for c in da]  # frozen copy with filled ghosts
        for mask in self.masks[level]:
            new = []
            for i in range(ndim):
                ui = da[i].data
                num = 0.0
                den = 0.0
                for j in range(ndim):
                    mlo, mhi = _face_lo_hi(mu[j], j, grid.n[j])
                    if j == i:
                        num = num + 2.0 * mhi * ui[grid.at({i: 1})] + 2.0 * mlo * ui[grid.at({i: -1})]
                        den = den + 2.0 * mhi + 2.0 * mlo
                    else:
                        uj, vj = da[j].data, ua[j]
                        num = num + mhi * (ui[grid.at({j: 1})]
                                           + (uj[grid.at({i: 1})] + vj[grid.at({i: 1, j: 1})]) / 4.0
                                           - (uj[grid.at({i: -1})] + vj[grid.at({i: -1, j: 1})]) / 4.0)
                        num = num - mlo * (-ui[grid.at({j: -1})]
                                                       + (vj[grid.at({i: 1, j: -1})] + uj[grid.at({i: 1})]) / 4.0
                                                       - (vj[grid.at({i: -1, j: -1})] + uj[grid.at({i: -1})]) / 4.0)
                        den = den + mhi + mlo
                new.append((res[i][...] * h2 + dt / rho * num) / (h2 + dt / rho * den))
            for i in range(ndim):
                cur = da[i].data[grid.at()]
                da[i].data[grid.at()] = xp.where(mask, new[i], cur)


def viscosity(u: VectorField, mu: FaceField, rho=1.0, dt: float = 1.0, *, nrelax: int = 4,
              tolerance: float = TOLERANCE, nitermax: int = NITERMAX, nitermin: int = NITERMIN) -> MGStats:
    """Advance ``u`` by one implicit viscous step in place (Basilisk ``viscosity()``)."""
    r = [c.interior.copy() for c in u]
    problem = ViscosityProblem.reuse(u, mu, rho, dt)
    return mg_solve(list(u), r, problem, nrelax=nrelax, tolerance=tolerance,
                    nitermax=nitermax, nitermin=nitermin)

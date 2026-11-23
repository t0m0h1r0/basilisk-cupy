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
from ..kernels.viscosity_gpu import relax_viscosity_redblack
from .multigrid import NITERMAX, NITERMIN, TOLERANCE, MGStats, mg_solve, restrict_cell, restrict_face


def _face_lo_hi(face, axis: int, n: int):
    lo = [slice(None)] * face.ndim
    hi = [slice(None)] * face.ndim
    lo[axis] = slice(0, n)
    hi[axis] = slice(1, n + 1)
    return face[tuple(lo)], face[tuple(hi)]


class ViscosityProblem:
    def __init__(self, u: VectorField, mu: FaceField, rho, dt: float, min_cells: int = 2) -> None:
        grid = u.grid
        if len(set(round(h, 14) for h in grid.h)) != 1:
            raise ValueError("viscosity(): square cells are required (Basilisk uses a single Delta)")
        xp = grid.xp
        self.dt = dt
        self.levels = build_levels(grid, min_cells=min_cells)
        self.correction_bcs = [c.bc.homogeneous() for c in u]
        mus = [list(mu.components)]
        for _ in range(len(self.levels) - 1):
            mus.append([restrict_face(c, ax) for ax, c in enumerate(mus[-1])])
        self.mu = mus[::-1]
        if hasattr(rho, "shape"):
            rhos = [rho]
            for _ in range(len(self.levels) - 1):
                rhos.append(restrict_cell(rhos[-1]))
            self.rho = rhos[::-1]
        else:
            self.rho = [float(rho)] * len(self.levels)
        # GPU: one fused kernel per colour (kernels/viscosity_gpu.py); otherwise red/black masks.
        self.fused = xp.__name__ == "cupy" and grid.dtype == "float64"
        self.masks = []
        for g in self.levels:
            idx = xp.indices(g.n).sum(axis=0)
            self.masks.append(((idx % 2) == 0, (idx % 2) == 1))

    # Basilisk residual_viscosity ("naive" Cartesian version) ------------------------
    def residual(self, a: list[CellField], b: list, res: list) -> float:
        grid = a[0].grid
        xp = grid.xp
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
        if self.fused:
            relax_viscosity_redblack([c.data for c in da], res, mu, rho, grid.n, h2, dt, grid.ghosts)
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
    problem = ViscosityProblem(u, mu, rho, dt)
    return mg_solve(list(u), r, problem, nrelax=nrelax, tolerance=tolerance,
                    nitermax=nitermax, nitermin=nitermin)

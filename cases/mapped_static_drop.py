"""Static drop on a grid refined in y (docs/notes/39): spurious currents, pressure jump, mass.

Drop ``R = 0.25`` at the centre of ``[-0.5, 0.5]^2`` (walls), sigma = 1, mu = 0.1, equal densities, CLS + sharp
GFM, Stokes (no advection: BCG does not take mapped grids), ``FD2Viscosity`` (Basilisk's ``viscosity()`` needs
square uniform cells). The y axis carries ``MonitorMapping(1 + (alpha - 1) exp(-((s - s0)/w)^2))`` (cells
``alpha`` times smaller about ``y = y0``); ``alpha = 1`` is the uniform grid through the same code.

    python -m cases.mapped_static_drop
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.grid.mapping import MonitorMapping
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.terms.viscous import FD2Viscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS


def band(alpha: float, centre: float = 0.5, width: float = 0.08) -> MonitorMapping | None:
    """Cells ``alpha`` times smaller about ``s = centre`` of the unit axis (None for alpha = 1)."""
    if alpha == 1:
        return None
    return MonitorMapping(lambda s, a=alpha, c=centre, w=width: 1.0 + (a - 1.0) * np.exp(-((s - c) / w) ** 2))


def run(n: int, alpha: float = 1.0, steps: int = 50, centre: float = 0.75, dt_factor: float | None = None,
        eps: str = "max"):
    """``eps``: CLS width 1.5 times the largest (``"max"``) or the smallest (``"min"``) cell width."""
    warnings.filterwarnings("ignore")
    m = band(alpha, centre)
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), mapping=None if m is None else (None, m))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10,
                    viscous=FD2Viscosity(), stokes=True)
    x, y = grid.meshgrid()
    from basilisk_cupy.interface.cls import cell_widths, min_width

    wmax = max(float(np.max(cell_widths(grid, ax))) for ax in range(2))
    e = 1.5 * (wmax if eps == "max" else min_width(grid))
    cls = CLSInterface.from_level_set(grid, 0.25 - np.sqrt(x * x + y * y), BoundaryConditions(), eps=e,
                                      ops=FCCDFaceComplex(order=4))
    kw = {} if dt_factor is None else {"capillary_dt_factor": dt_factor}
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1, pressure_jump=True, **kw)
    tp.add_surface_tension(1.0)
    ns.start()
    m0 = cls.mass()
    umax = []
    for _ in range(steps):
        ns.step()
        umax.append(max(float(np.max(np.abs(c.interior))) for c in ns.u))
    p = ns.p.interior
    r = np.sqrt(x * x + y * y) * np.ones(grid.n)
    inside, outside = r < 0.15, r > 0.35  # away from the interface band
    dv = grid.cell_volumes() * np.ones(grid.n)
    jump = float(np.sum((p * dv)[inside]) / np.sum(dv[inside]) - np.sum((p * dv)[outside]) / np.sum(dv[outside]))
    return {"umax": umax, "jump": jump, "mass": abs(cls.mass() - m0) / m0, "dt": float(ns.clock.dt), "t": ns.clock.t}


def main() -> None:
    for n, a, e in ((32, 1, "max"), (32, 4, "min"), (32, 4, "max"), (64, 1, "max")):
        r = run(n, a, eps=e, steps=200)
        u = r["umax"]
        print(f"N={n} alpha={a} eps 1.5 h_{e}: max|u| at 10/50/200 steps {u[9]:.2e} / {u[49]:.2e} / {u[-1]:.2e}  jump {r['jump']:.4f} "
              f"(exact 4)  mass change {r['mass']:.1e}  dt {r['dt']:.2e}", flush=True)


if __name__ == "__main__":
    main()

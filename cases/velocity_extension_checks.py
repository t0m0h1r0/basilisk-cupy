"""Side effects of the velocity extension (docs/notes/44): translation, reversed vortex, static drop.

    python -m cases.velocity_extension_checks
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases.velocity_extension import attach


def translation(n: int, kinematics: str, R: float = 0.15, u=(1.0, 1.0), cfl: float = 0.25):
    """A circle carried once around the periodic unit box by a uniform velocity: psi L1 error, mass change,
    curvature spread in the band."""
    bc = BoundaryConditions.periodic(2)
    grid = CartesianGrid((n, n), origin=(0.0, 0.0))
    x, y = grid.meshgrid()
    phi = R - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    cls = CLSInterface.from_level_set(grid, phi, bc, ops=FCCDFaceComplex(order=4))
    if kinematics == "extension":
        attach(cls)
    q0 = cls.psi.interior.copy()
    m0 = cls.mass()
    uf = FaceField(grid, [np.full(grid.face_shape(0), u[0]), np.full(grid.face_shape(1), u[1])], "uf")
    h = 1.0 / n
    steps = int(math.ceil(1.0 / (cfl * h)))
    dt = 1.0 / steps
    for _ in range(steps):
        cls.advect(uf, dt)
    k = cls.curvature()
    band = (cls.psi.interior > 0.1) & (cls.psi.interior < 0.9)
    return {"L1": float(np.abs(cls.psi.interior - q0).sum()) * h * h, "mass": cls.mass() / m0 - 1,
            "kappa_mean": float(k[band].mean()), "kappa_std": float(k[band].std()), "steps": steps}


def static_drop(n: int, kinematics: str, rho2: float = 1.0, steps: int = 200, factor: float = 1.0):
    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10)
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, 0.25 - np.sqrt(x * x + y * y), BoundaryConditions())
    if kinematics == "extension":
        attach(cls)
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, pressure_jump=True, capillary_dt_factor=factor)
    tp.add_surface_tension(1.0)
    ns.start()
    m0 = cls.mass()
    umax = []
    for _ in range(steps):
        ns.step()
        umax.append(max(float(np.max(np.abs(c.interior))) for c in ns.u))
        if not np.isfinite(umax[-1]) or umax[-1] > 10:
            break
    r = np.sqrt(x * x + y * y) * np.ones(grid.n)
    p = tp.full_pressure(ns)
    jump = float(p[r < 0.15].mean() - p[r > 0.35].mean())
    return {"umax": umax, "jump": jump, "mass": abs(cls.mass() - m0) / m0}


def main() -> None:
    from cases.reversed_cls import run as reversed_run

    for n in (32, 64):
        for k in ("transport", "extension"):
            r = translation(n, k)
            print(f"translation N={n} {k:9s}: psi L1 {r['L1']:.2e} mass {r['mass']:+.1e} kappa {r['kappa_mean']:.3f} "
                  f"+- {r['kappa_std']:.3f} (exact {1 / 0.15:.3f})", flush=True)
    for n in (64, 128):
        for k in ("transport", "extension"):
            r = reversed_run(n, kinematics=k)
            print(f"reversed N={n} {k:9s}: psi-f L1 {r['final'][0]:.3e} sharp L1 {r['sharp'][0]:.3e} mass {r['mass_change']:+.1e}",
                  flush=True)
    for n, rho2 in ((32, 1.0), (64, 1.0), (32, 0.01)):
        for k in ("transport", "extension"):
            r = static_drop(n, k, rho2)
            u = r["umax"]
            print(f"static drop N={n} ratio {1 / rho2:.0f} {k:9s}: max|u| 10/50/200 {u[9]:.2e} / {u[49]:.2e} / {u[-1]:.2e} "
                  f"({len(u)} steps) jump {r['jump']:.4f} mass {r['mass']:.1e}", flush=True)


if __name__ == "__main__":
    main()

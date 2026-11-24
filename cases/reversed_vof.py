"""Time-reversed VOF advection in a vortex (port of Basilisk ``src/test/reversed.c``, constant grid).

A circle of radius 0.2 at (-0.2, -0.236338) is stretched by the vortex
``psi = -1.5 sin(2 pi t / T) sin(pi (x + 1/2)) sin(pi (y + 1/2)) / pi`` (T = 15) and brought
back. The reference (``reversed.cref``, constant grid) gives the volume sums and the error
norms ``avg, rms, max`` of ``fraction(circle) - f`` at ``t = T``.

    python -m cases.reversed_vof [--nmax 7] [--backend numpy|cupy]
"""

from __future__ import annotations

import argparse
import math

from basilisk_cupy.constants import PI

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection
from basilisk_cupy.time.stepping import Clock, dtnext, timestep

T = 15.0
DT = 0.1
CFL = 0.5  # vof.h caps the CFL at 1/2
#: reversed.cref: N -> (f.sum, avg, rms, max) at t = T
REFERENCE = {32: (0.124638306824, 0.0605584, 0.21841, 1.0), 64: (0.125405298502, 0.00992636, 0.0733979, 1.0),
             128: (0.125600659213, 0.00143235, 0.017043, 0.474599)}


def circle(x, y):
    return 0.2 ** 2 - ((x + 0.2) ** 2 + (y + 0.236338) ** 2)


def velocity(grid, t):
    """Face velocity from the vertex stream function (exactly divergence-free for the face difference)."""
    xp = grid.xp
    xv, yv = xp.meshgrid(grid.faces(0), grid.faces(1), indexing="ij")
    psi = -1.5 * math.sin(2.0 * PI * t / T) * xp.sin(PI * (xv + 0.5)) * xp.sin(PI * (yv + 0.5)) / PI
    h = float(grid.h[0])
    return FaceField(grid, [-1.0 * (psi[:, 1:] - psi[:, :-1]) / h, 1.0 * (psi[1:, :] - psi[:-1, :]) / h])


def run(n: int, backend: str = "numpy"):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    bc = BoundaryConditions()  # Basilisk default for scalars: symmetric (Neumann 0)
    f = CellField.from_interior(grid, fraction(grid, circle), bc)
    cf = CellField.from_interior(grid, f.interior.copy(), bc)
    itf = VOFInterface(f, [VOFTracer(cf)])
    clock = Clock(events=[T / 4 * k for k in range(1, 5)])
    dv = float(grid.h[0] * grid.h[1])
    sum0 = float(f.interior.sum()) * dv
    cfl_max = 0.0
    while clock.t < T - 1e-9 * T:
        uf = velocity(grid, clock.t)
        cfl = 0.8 if clock.i == 0 else CFL  # vof.h's stability event runs after the first velocity event
        dt = dtnext(timestep(uf, DT, cfl, clock), clock)
        cfl_max = max(cfl_max, vof_advection([itf], uf, dt, clock.i))
        clock.t = clock.tnext
        clock.i += 1
    e = fraction(grid, circle) - f.interior
    a = xp.abs(e)
    c = xp.where(f.interior > 1e-6, cf.interior / xp.where(f.interior > 1e-6, f.interior, 1.0), 1.0)
    return {"sum0": sum0, "sum": float(f.interior.sum()) * dv, "min": float(f.interior.min()), "max": float(f.interior.max()),
            "avg": float(a.sum()) * dv, "rms": math.sqrt(float((a * a).sum()) * dv), "emax": float(a.max()),
            "cdev": float(xp.abs(c - 1.0).max()), "steps": clock.i, "cfl": cfl_max}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=7)
    ap.add_argument("--backend", default="numpy")
    args = ap.parse_args()
    for level in range(5, args.nmax + 1):
        n = 1 << level
        r = run(n, args.backend)
        ref = REFERENCE.get(n)
        print(f"{n} sum {r['sum0']:.12f} -> {r['sum']:.12f}  avg {r['avg']:.6g} rms {r['rms']:.6g} max {r['emax']:.6g}"
              f"  f in [{r['min']:.3g}, {r['max']:.3g}]  |cf/f - 1| {r['cdev']:.1e}  steps {r['steps']}"
              + (f"   ref {ref}" if ref else ""))


if __name__ == "__main__":
    main()

"""Consistency of the split operators of CenteredNS about a CLS drop (docs/notes/35, Phase 2).

``P``: the approximate projection as the cell velocity sees it (``uf = interp(u)``, project, ``u + dt
face_to_cell(-alpha grad p)``); ``B``: the implicit viscous solve ``(I - dt/rho div mu grad)^{-1}``.
For constant coefficients both are functions of the Laplacian and commute; a density or viscosity jump
breaks this. Reported, for a smooth random divergence-free-ish field ``u``:

* ``comm = |PBu - BPu| / |u|`` (the commutator),
* ``idem = |PPu - Pu| / |Pu|`` (P is not a projector on cells: interpolation/averaging inconsistency),
* ``divres = max |div interp(Pu)| dt`` (the cell velocity after the projection is not divergence-free).

    python -m cases.operator_commutator
"""

from __future__ import annotations

import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.constants import PI
from basilisk_cupy.fields import CellField, FaceField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.linear.viscosity import viscosity
from basilisk_cupy.projection.approximate import project
from basilisk_cupy.time.basilisk_centered import CenteredNS

#: (name, rho2, mu2) with rho1 = 1, mu1 = 0.1: density and viscosity ratios varied apart
CASES = [("A: rho 1, mu 1", 1.0, 0.1), ("B: rho 100, mu 1", 0.01, 0.1), ("C: rho 100, nu 1", 0.01, 0.001),
         ("D: rho 1, mu 100", 1.0, 0.001), ("E: rho 1000, mu 1", 0.001, 0.1)]


class Ops:
    def __init__(self, n=32, rho2=1.0, mu2=0.1, mu1=0.1, factor=1.0):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
        x, y = grid.meshgrid()
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-14)
        cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
        TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=mu1, mu2=mu2)
        ns.start()
        h = 1.0 / n
        self.dt = factor * math.sqrt((1.0 + rho2) / 2.0 * h ** 3 / PI)
        self.grid, self.ns = grid, ns
        self.p = CellField(grid, bc=ns.p.bc, name="ptmp")

    def _set(self, u):
        for c, v in zip(self.ns.u, u):
            c.interior = v.copy()
        self.ns.u.fill_ghosts()

    def _get(self):
        return [c.interior.copy() for c in self.ns.u]

    def P(self, u):
        ns, dt = self.ns, self.dt
        self._set(u)
        uf = ns.face_velocity()
        self.p.interior = 0.0 * self.p.interior
        project(uf, self.p, ns.alpha, dt, ops=ns.ops, tolerance=1e-14)
        self.p.fill_ghosts()
        out = []
        for ax in range(2):
            gf = -ns.alpha[ax] * ns.ops.grad_f(self.p, ax)
            out.append(u[ax] + dt * ns.ops.face_to_cell(gf, ax, periodic=False))
        return out

    def B(self, u):
        self._set(u)
        viscosity(self.ns.u, self.ns.mu, self.ns.rho, self.dt, tolerance=1e-14)
        return self._get()

    def div(self, u):
        self._set(u)
        uf = self.ns.face_velocity()
        return float(np.abs(self.ns.ops.div_f(uf, periodic=())).max()) * self.dt


def norm(u):
    return math.sqrt(sum(float((c * c).sum()) for c in u))


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def smooth_field(n, seed=0, kmax=4):
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = [np.zeros((n, n)), np.zeros((n, n))]
    for c in u:
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                c += rng.standard_normal() * np.sin(PI * kx * X) * np.sin(PI * ky * Y) / (kx * kx + ky * ky)
    return u


def lagged_gradient(o):
    """``G = dt face_to_cell(alpha grad q)`` for the pressure ``q = x^2 - y^2`` (the l = 2 mode's): what one
    outer iteration leaves when the pressure lags by ``q``."""
    ns, grid = o.ns, o.grid
    x, y = grid.meshgrid()
    q = CellField(grid, bc=ns.p.bc, name="q")
    q.interior = x * x - y * y
    q.fill_ghosts()
    G = [o.dt * ns.ops.face_to_cell(ns.alpha[ax] * ns.ops.grad_f(q, ax), ax, periodic=False) for ax in range(2)]
    G1 = [o.dt * ns.ops.face_to_cell(ns.ops.grad_f(q, ax), ax, periodic=False) for ax in range(2)]
    return G, G1


def measure(n=32, rho2=1.0, mu2=0.1, mu1=0.1, factor=1.0, seed=0):
    o = Ops(n, rho2, mu2, mu1, factor)
    u = smooth_field(n, seed)
    Pu, Bu = o.P(u), o.B(u)
    G, G1 = lagged_gradient(o)
    return {"comm": norm(sub(o.P(Bu), o.B(Pu))) / norm(u), "idem": norm(sub(o.P(Pu), Pu)) / norm(Pu),
            "divres": o.div(Pu), "dt": o.dt,
            # one outer iteration with a pressure lagging by q: correction(dt), viscosity, correction(-dt)
            # leave (B - I) G in u, which the projection does not remove (zero without viscosity)
            "lag": norm(o.P(sub(o.B(G), G))) / norm(G1)}


def main() -> None:
    for n in (32, 64):
        for name, rho2, mu2 in CASES:
            r = measure(n, rho2, mu2)
            print(f"N={n} {name:20s} comm {r['comm']:.3e}  idem {r['idem']:.3e}  div*dt {r['divres']:.2e}  "
                  f"lag {r['lag']:.3e}", flush=True)
        for rho2 in (1.0, 0.1, 0.01, 0.001):
            r = measure(n, rho2, 0.1)
            print(f"N={n} ratio {1 / rho2:6.0f}       comm {r['comm']:.3e}  idem {r['idem']:.3e}  div*dt {r['divres']:.2e}  "
                  f"lag {r['lag']:.3e}", flush=True)


if __name__ == "__main__":
    main()

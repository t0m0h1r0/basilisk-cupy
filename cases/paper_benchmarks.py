"""The four benchmarks of the TwoPhaseFlow paper (chapter 14) with the paper's settings (phase 3).

=========  ==================================================================================================
case       setting (paper)
=========  ==================================================================================================
capwave    [0, 0.02]^2 m, 32^2, x periodic / y walls, water--air, y0 = 0.01, A0 = 2e-4, lambda = 0.01, g = 0,
           T = 0.046742983863 s (14a)
drop       [0, 0.02]^2 m, 32^2, periodic, water drop: ellipse a = 5.5 mm, b = 4.5 mm at the centre (D0 = 0.1),
           T_RL = 0.105460663082 s (14b; the paper integrates a reduced linear chart, not Navier--Stokes)
bubble     [0, 0.01] x [0, 0.02] m, 32 x 64, x periodic / y walls, air bubble R = 2.5 mm at (5, 5) mm in water,
           rho 1000 / 1.2, mu 1e-3 / 1.8e-5, sigma 0.072, g = 9.81, t <= 0.03 s (14c; the paper reports a
           breakdown after t ~ 0.04 s)
rt         [0, 0.5] x [0, 2], 128 x 512, walls, water (1000, 1e-3) above silicone oil (950, 5e-3),
           sigma 0.04, g = 9.81, y0 = 1, A0 = 0.01, m = 1, T = 7 (14d)
=========  ==================================================================================================

Phase 1 (psi = 1) is the liquid (capwave, drop, bubble) or the lower oil (rt). Differences from the paper:
uniform grids (the paper rebuilds an interface-fitted alpha = 2 grid every step), the CLS transport instead of
the swept-volume cell fraction (capwave), and the time step from tension.h's bound times sqrt(5/12).

    python -m cases.paper_benchmarks --case capwave [--backend cupy] [stepper options as key=value]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from basilisk_cupy import CartesianGrid
from basilisk_cupy.backend import to_numpy

WATER_AIR = dict(rho1=998.2, rho2=1.204, mu1=1.002e-3, mu2=1.825e-5, sigma=0.0728)


class VOFRun:
    """Basilisk's VOF two-phase flow on ``CenteredNS`` with the diagnostics interface of the CLS steppers
    (``psi`` is the volume fraction)."""

    def __init__(self, grid, level, kw) -> None:
        from basilisk_cupy import BoundaryConditions, CellField
        from basilisk_cupy.boundary import Neumann, Periodic
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS

        periodic = tuple(kw.get("periodic", ()))
        walls = {w: 0.0 for w, ax in (("left", 0), ("right", 0), ("bottom", 1), ("top", 1)) if ax not in periodic}
        self.ns = CenteredNS(grid, periodic=periodic, uf_normal=walls, tolerance=kw.get("vof_tolerance", 1e-8))
        bc = BoundaryConditions()
        for ax in range(2):
            for side in (0, 1):
                bc.set((ax, side), Periodic() if ax in periodic else Neumann(0.0))
        self.f = CellField.from_interior(grid, fraction(grid, level), bc)
        self.tp = TwoPhase(self.ns, self.f, rho1=kw["rho1"], rho2=kw["rho2"], mu1=kw["mu1"], mu2=kw["mu2"])
        self.tp.add_surface_tension(kw["sigma"])
        if kw.get("gravity") is not None:
            G, Z = kw["gravity"]
            self.tp.add_reduced_gravity(G, Z)
        self.grid, self.cls = grid, self
        self.dv = float(grid.h[0] * grid.h[1])

    def __getattr__(self, key):
        return getattr(self.ns, key)

    @property
    def psi(self):
        return self.f.interior

    def mass(self) -> float:
        return float(self.grid.xp.sum(self.f.interior)) * self.dv

    def sharp_volume(self) -> float:
        return self.mass()

    def kinetic_energy(self) -> float:
        xp = self.grid.xp
        rho = self.ns.rho if not isinstance(self.ns.rho, float) else self.ns.rho
        return 0.5 * float(sum(xp.sum(rho * c.interior ** 2) for c in self.ns.u)) * self.dv

    def max_velocity(self) -> float:
        xp = self.grid.xp
        return max(float(xp.max(xp.abs(c.interior))) for c in self.ns.u)

    def start(self, events=None):
        self.ns.start(events=events)

    def step(self):
        self.ns.step()


def _stepper(grid, phi, kw, level=None):
    kw = dict(kw)
    if kw.pop("method", None) == "vof":
        return VOFRun(grid, level, kw)
    if str(kw.get("interface", "cls")).startswith("vof"):
        phi = level  # the VOF fraction is built from the level function on the vertices
    # dimensional problems: bound the relative volume change per projection step (Basilisk's project())
    kw.setdefault("volume_tolerance", True)
    kw.setdefault("tolerance", 1e-8)  # converged to 1e-5 in the capillary wave's amplitude ratio
    tableau = kw.pop("tableau", None)
    if tableau:
        from basilisk_cupy.time.two_phase_imex_rk import TwoPhaseIMEXRK

        return TwoPhaseIMEXRK(grid, phi, tableau=tableau, **kw)
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    return TwoPhaseIMEX(grid, phi, **kw)


def ellipse_distance(x, y, a, b, iters: int = 30):
    """Signed distance to the ellipse ``(x/a)^2 + (y/b)^2 = 1`` (positive inside), by Newton on the angle of
    the closest point (``a >= b``)."""
    X, Y = np.abs(to_numpy(x)), np.abs(to_numpy(y))
    t = np.arctan2(a * Y, b * X)
    for _ in range(iters):
        c, s = np.cos(t), np.sin(t)
        f = (a * a - b * b) * s * c - X * a * s + Y * b * c
        df = (a * a - b * b) * (c * c - s * s) - X * a * c - Y * b * s
        t = np.clip(t - f / np.where(np.abs(df) > 1e-30, df, 1e-30), 0.0, math.pi / 2)
    d = np.hypot(X - a * np.cos(t), Y - b * np.sin(t))
    inside = (X / a) ** 2 + (Y / b) ** 2 < 1.0
    return np.where(inside, d, -d)


def setup(case: str, n: int | None = None, backend: str = "numpy", **kw):
    """``(ns, info)``; ``n`` scales the grid (cells along x)."""
    if case == "capwave":
        n = n or 32
        grid = CartesianGrid((n, n), length=(0.02, 0.02), backend=backend)
        x, y = grid.meshgrid()
        xp = grid.xp
        k = 2 * math.pi / 0.01
        eta = 0.01 + 2e-4 * xp.cos(k * x)
        phi = (eta - y) / xp.sqrt(1.0 + (2e-4 * k * xp.sin(k * x)) ** 2)
        ns = _stepper(grid, phi, dict(WATER_AIR, periodic=(0,), **kw),
                      level=lambda x, y: 0.01 + 2e-4 * np.cos(k * x) - y)
        return ns, dict(k=k, y0=0.01, A0=2e-4, T=0.046742983863)
    if case == "drop":
        n = n or 32
        grid = CartesianGrid((n, n), length=(0.02, 0.02), backend=backend)
        x, y = grid.meshgrid()
        phi = grid.xp.asarray(ellipse_distance(x - 0.01, y - 0.01, 5.5e-3, 4.5e-3))
        ns = _stepper(grid, phi, dict(WATER_AIR, periodic=(0, 1), **kw),
                      level=lambda x, y: 1.0 - ((x - 0.01) / 5.5e-3) ** 2 - ((y - 0.01) / 4.5e-3) ** 2)
        R = math.sqrt(5.5e-3 * 4.5e-3)
        w0 = math.sqrt(6 * 0.0728 / ((998.2 + 1.204) * R ** 3))
        return ns, dict(R=R, D0=0.1, omega0=w0, T=2 * math.pi / w0, centre=(0.01, 0.01))
    if case == "bubble":
        n = n or 32
        grid = CartesianGrid((n, 2 * n), length=(0.01, 0.02), backend=backend)
        x, y = grid.meshgrid()
        phi = grid.xp.sqrt((x - 0.005) ** 2 + (y - 0.005) ** 2) - 0.0025
        ns = _stepper(grid, phi, dict(rho1=1000.0, rho2=1.2, mu1=1e-3, mu2=1.8e-5, sigma=0.072, periodic=(0,),
                                      gravity=((0.0, -9.81), (0.0, 0.0)), **kw),
                      level=lambda x, y: (x - 0.005) ** 2 + (y - 0.005) ** 2 - 0.0025 ** 2)
        return ns, dict(T=0.03)
    if case == "rt":
        n = n or 128
        grid = CartesianGrid((n, 4 * n), length=(0.5, 2.0), backend=backend)
        x, y = grid.meshgrid()
        xp = grid.xp
        k = 2 * math.pi / 0.5
        eta = 1.0 + 0.01 * xp.cos(k * x)
        phi = (eta - y) / xp.sqrt(1.0 + (0.01 * k * xp.sin(k * x)) ** 2)
        ns = _stepper(grid, phi, dict(rho1=950.0, rho2=1000.0, mu1=5e-3, mu2=1e-3, sigma=0.040,
                                      gravity=((0.0, -9.81), (0.0, 0.0)), **kw),
                      level=lambda x, y: 1.0 + 0.01 * np.cos(k * x) - y)
        return ns, dict(k=k, y0=1.0, A0=0.01, T=7.0)
    raise ValueError(f"unknown case {case!r}")


# -- diagnostics -----------------------------------------------------------------------------------------

def crossing_heights(psi, grid):
    """Height of the first ``psi = 1/2`` crossing (from below) in every column."""
    q = to_numpy(psi)
    _, y = grid.meshgrid()
    y = to_numpy(y)
    a, b = q[:, :-1] - 0.5, q[:, 1:] - 0.5
    cross = (a * b <= 0) & (a != b)
    t = np.where(cross, a / np.where(cross, a - b, 1.0), 0.0)
    yc = y[:, :-1] + t * float(grid.h[1])
    first = np.argmax(cross, axis=1)
    return yc[np.arange(q.shape[0]), first]


def column_heights(psi, grid):
    """Interface height of every column from the column integral of phase 1 (below the interface):
    ``y_bottom + sum_j psi h``. Exact for a VOF fraction and for a symmetric CLS profile; unlike the psi = 1/2
    crossing it does not depend on how sharp the profile is (a VOF fraction jumps within one cell)."""
    q = to_numpy(psi)
    return float(grid.origin[1]) + q.sum(axis=1) * float(grid.h[1])


def mode_amplitude(ns, info) -> float:
    """First Fourier coefficient of the interface height, ``(2/N) sum (eta - y0) cos(k x)``, with ``eta`` from the
    column integral (:func:`column_heights`)."""
    grid = ns.grid
    x, _ = grid.meshgrid()
    xs = to_numpy(x)[:, 0]
    eta = column_heights(ns.psi, grid)
    return float(2.0 / len(xs) * np.sum((eta - info["y0"]) * np.cos(info["k"] * xs)))


def moments(ns, weight):
    """Centroid, velocity and deformation ``(sqrt Ixx - sqrt Iyy)/(sqrt Ixx + sqrt Iyy)`` of a phase weight."""
    grid = ns.grid
    x, y = (to_numpy(c) for c in grid.meshgrid())
    w = to_numpy(weight)
    W = w.sum()
    xc, yc = (w * x).sum() / W, (w * y).sum() / W
    u, v = (to_numpy(c.interior) for c in ns.u)
    ixx, iyy = (w * (x - xc) ** 2).sum() / W, (w * (y - yc) ** 2).sum() / W
    D = (math.sqrt(ixx) - math.sqrt(iyy)) / (math.sqrt(ixx) + math.sqrt(iyy))
    return dict(xc=float(xc), yc=float(yc), uc=float((w * u).sum() / W), vc=float((w * v).sum() / W), D=float(D))


def axis_radii(ns, centre):
    """Half-axes along x and y through ``centre``: half the integral of phase 1 on the lines ``y = yc`` and
    ``x = xc`` (``psi`` linearly interpolated between the cell rows / columns). Exact for a VOF fraction and for a
    symmetric CLS profile, unlike the psi = 1/2 crossing (a VOF fraction jumps within one cell)."""
    grid = ns.grid
    q = to_numpy(ns.psi)
    h = [float(v) for v in grid.h]
    o = [float(v) for v in grid.origin]
    radii = []
    for ax in (0, 1):
        t = 1 - ax
        f = (centre[t] - o[t]) / h[t] - 0.5  # fractional cell index of the line across the other axis
        j = int(math.floor(f))
        w = f - j
        line = (1 - w) * np.take(q, j, axis=t) + w * np.take(q, j + 1, axis=t)
        radii.append(0.5 * float(line.sum()) * h[ax])
    return radii


def diagnostics(case, ns, info) -> dict:
    out = dict(t=float(ns.clock.t), mass=ns.mass(), volume=ns.cls.sharp_volume(), ke=ns.kinetic_energy(),
               umax=ns.max_velocity())
    if case in ("capwave", "rt"):
        out["amplitude"] = mode_amplitude(ns, info)
    elif case == "drop":
        out.update(moments(ns, ns.psi))
        a, b = axis_radii(ns, info["centre"])
        out.update(a=a, b=b, D=(a - b) / (a + b))
    elif case == "bubble":
        out.update(moments(ns, 1.0 - to_numpy(ns.psi)))
    return out


def run(case: str, n: int | None = None, t_end: float | None = None, samples: int = 200, backend: str = "numpy",
        umax_limit: float = 5.0, snapshots: tuple = (), after_properties=None, **kw) -> dict:
    """Time series of the diagnostics at ``samples`` equally spaced output times (hit exactly)."""
    ns, info = setup(case, n, backend, **kw)
    if after_properties is not None:  # a diagnostic hook run after the two-phase properties of every step
        ns.on("properties", after_properties, after=True)
    t_end = t_end or info["T"]
    times = [t_end * (i + 1) / samples for i in range(samples)]
    ns.start(events=times)
    series = [diagnostics(case, ns, info)]
    snaps = {}
    t0 = time.time()
    k = 0
    while k < len(times):
        ns.step()
        u = ns.max_velocity()
        if not math.isfinite(u) or u > umax_limit:  # blow-up: stop before the step collapses
            series.append(dict(t=float(ns.clock.t), failed=True, umax=u))
            break
        if abs(ns.clock.t - times[k]) <= 1e-9 * max(1.0, t_end):
            series.append(diagnostics(case, ns, info))
            for ts in snapshots:  # psi at the output times closest to the requested ones
                if abs(times[k] - ts) <= 0.5 * t_end / samples:
                    snaps[float(ts)] = to_numpy(ns.psi).tolist()
            k += 1
    if ns.grid.xp.__name__ == "cupy":
        ns.grid.xp.cuda.Stream.null.synchronize()
    return dict(case=case, info=info, series=series, steps=ns.clock.i, wall=time.time() - t0, snapshots=snaps,
                psi=to_numpy(ns.psi).tolist() if samples and ns.grid.n[0] <= 64 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("capwave", "drop", "bubble", "rt"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--out", default=None)
    ap.add_argument("options", nargs="*", help="stepper keywords, e.g. scheme=cnab2 transport=ab2 tableau=ars222")
    args = ap.parse_args()
    kw = {}
    for o in args.options:
        key, val = o.split("=", 1)
        try:
            kw[key] = json.loads(val)
        except json.JSONDecodeError:
            kw[key] = val
    r = run(args.case, args.n, args.t_end, args.samples, args.backend, **kw)
    last = r["series"][-1]
    print(json.dumps({"steps": r["steps"], "wall": round(r["wall"], 1), "last": last, "info": r["info"]}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    main()

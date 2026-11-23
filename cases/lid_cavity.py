"""Lid-driven cavity at Re = 1000 (port of Basilisk ``src/test/lid.c``).

Unit box centred on the origin, mu = 1e-3, DT = 0.1, CFL = 0.8, no-slip
walls and a lid moving at u = 1, ``uf.n = 0`` on every wall. Every 0.1 time
units ``t energy du`` is printed (Basilisk's log, compare with
``lid.ref`` / ``lid.gpu.ref``); the run stops when du < 1e-5. The final
centreline profiles are compared with Ghia et al. (1982).

    python -m cases.lid_cavity [--n 64] [--t-end 50] [--backend numpy|cupy]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from basilisk_cupy import CartesianGrid, Dirichlet
from basilisk_cupy.time.basilisk_centered import CenteredNS

#: First rows of Basilisk's lid.gpu.ref (t, energy, du); lid.ref (CPU) agrees to 4 digits.
REFERENCE = [
    (0.1, 0.001572, 0.448), (0.2, 0.003192, 0.182), (0.3, 0.004351, 0.121), (0.4, 0.005229, 0.091),
    (0.5, 0.005956, 0.0767), (0.6, 0.006578, 0.0652), (0.7, 0.007129, 0.0596), (0.8, 0.007628, 0.0556),
    (0.9, 0.008084, 0.0544), (1.0, 0.008508, 0.0536), (1.1, 0.008904, 0.0523),
]
#: Steady state of lid.ref: t = 44.0, energy = 0.04279.
REFERENCE_STEADY = (44.0, 0.04279)

#: Ghia, Ghia & Shin (1982), Re = 1000, as shipped with Basilisk (src/test/xprof.ghia,
#: yprof.ghia), coordinates centred on the origin.
GHIA_XPROF = [
    (-0.327052, -0.383699),
    (-0.397406, -0.297251),
    (-0.217948, -0.27788),
    (-0.428914, -0.222276),
    (-0.43629, -0.201989),
    (-0.444335, -0.181701),
    (-0.046595, -0.106804),
    (0.001598, -0.060949),
    (-0.49933, -0.000882),
    (0.118733, 0.057217),
    (0.235193, 0.186849),
    (0.352315, 0.333239),
    (0.45404, 0.466401),
    (0.461386, 0.511382),
    (0.469392, 0.574884),
    (0.476719, 0.659554),
    (0.5, 0.999118),
]  # (y, u.x(0, y))
GHIA_YPROF = [
    (-0.500577, 0.00069404),
    (-0.43768, 0.275621),
    (-0.429602, 0.290847),
    (-0.421523, 0.303994),
    (-0.406521, 0.326826),
    (-0.343624, 0.371038),
    (-0.273803, 0.330015),
    (-0.265724, 0.32307),
    (-0.000289, 0.0252893),
    (0.304962, -0.318994),
    (0.359781, -0.427191),
    (0.40652, -0.515279),
    (0.445182, -0.392034),
    (0.45326, -0.336623),
    (0.461339, -0.277749),
    (0.46884, -0.214023),
    (0.5, -6.20706e-17),
]  # (x, u.y(x, 0))


def setup(n: int = 64, backend: str = "numpy") -> CenteredNS:
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tangential = {w: Dirichlet(0.0) for w in walls}
    tangential["top"] = Dirichlet(1.0)
    return CenteredNS(grid, tangential=tangential, uf_normal={w: 0.0 for w in walls},
                      mu=1e-3, DT=0.1, cfl=0.8)


def energy(ns: CenteredNS) -> float:
    return ns.kinetic_energy()


def interpolate(f, point) -> float:
    """Bilinear interpolation of a cell field with filled ghosts (Basilisk ``interpolate``)."""
    grid = f.grid
    f.fill_ghosts()
    idx, w = [], []
    for ax, x in enumerate(point):
        s = (x - grid.origin[ax]) / grid.h[ax] - 0.5 + grid.ghosts  # padded index coordinate
        i = int(s // 1)
        i = min(max(i, grid.ghosts - 1), grid.ghosts + grid.n[ax] - 1)
        idx.append(i)
        w.append(s - i)
    d = f.data
    (i, j), (a, b) = idx, w
    val = ((1 - a) * (1 - b) * d[i, j] + a * (1 - b) * d[i + 1, j]
           + (1 - a) * b * d[i, j + 1] + a * b * d[i + 1, j + 1])
    return float(val)


def run(n: int = 64, t_end: float = 50.0, backend: str = "numpy", log=print, tol_du: float = 1e-5):
    """Run and return the list of (t, energy, du) rows."""
    ns = setup(n, backend)
    steps = int(round(t_end / 0.1))
    times = [round(0.1 * k, 10) for k in range(1, steps + 1)]
    ns.start(events=times)
    xp = ns.grid.xp
    un = ns.u[0].interior.copy()
    rows = []
    for t in times:
        ns.run(t)
        du = float(xp.max(xp.abs(ns.u[0].interior - un)))
        un = ns.u[0].interior.copy()
        if du < tol_du:
            break
        rows.append((ns.clock.t, energy(ns), du))
        if log:
            log(f"{ns.clock.t:f} {rows[-1][1]:.4g} {du:.3g}")
    return ns, rows


def profiles(ns: CenteredNS):
    xprof = [(y, interpolate(ns.u[0], (0.0, y)), ref) for y, ref in GHIA_XPROF]
    yprof = [(x, interpolate(ns.u[1], (x, 0.0)), ref) for x, ref in GHIA_YPROF]
    return xprof, yprof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--t-end", type=float, default=50.0)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--out", type=Path, default=None, help="directory for xprof/yprof files")
    args = ap.parse_args()
    ns, rows = run(args.n, args.t_end, args.backend)
    xprof, yprof = profiles(ns)
    print("# u.x(0, y) vs Ghia:  y  ours  ghia")
    for r in xprof:
        print(f"{r[0]:g} {r[1]:.4f} {r[2]:.4f}")
    print("# u.y(x, 0) vs Ghia:  x  ours  ghia")
    for r in yprof:
        print(f"{r[0]:g} {r[1]:.4f} {r[2]:.4f}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "xprof").write_text("".join(f"{y:g} {interpolate(ns.u[0], (0.0, y)):g}\n"
                                                for y in [-0.5 + 0.01 * k for k in range(101)]))
        (args.out / "yprof").write_text("".join(f"{x:g} {interpolate(ns.u[1], (x, 0.0)):g}\n"
                                                for x in [-0.5 + 0.01 * k for k in range(101)]))


if __name__ == "__main__":
    main()

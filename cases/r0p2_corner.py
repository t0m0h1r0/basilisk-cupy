"""docs/notes/120 (A-2 of #145, R0-P2): why two layers at rest with an oblique plane interface and reduced.h's gravity
are not balanced in 9 of docs/notes/115's 72 cases (three geometries times the three face densities, coarse grids,
the interface reaching a wall within a cell of a corner).

reduced.h's force is the jump of phi = (rho2 - rho1) G . (x_I - Z) across the interface, x_I the interface's
position from the height functions. For a plane through Z normal to G, phi = 0 exactly: any force at rest comes from
heights that are not on the plane. Prediction (fixed before the runs): such cells exist in exactly the failing
geometries, next to the corner; the heights there come from ghost cells that the slip walls' symmetry condition
mirrors (an implicit 90 degree contact angle), not from the plane, which meets the wall at 90 - theta.

  census()       for docs/notes/115's 24 geometries: max |phi| at t = 0 with the library's ghosts, against the
                 balance measured in docs/figures/r0p2_pressure_momentum.json (failing <=> phi != 0?).
  ghosts()       the three failing geometries: phi with the plane itself extended into the ghost cells ("exact"),
                 and with only the corner blocks exact; and which height column (x or y) the bad cells use.
  balance()      the full solver at rest (smooth, density ratio 1000) with the library's ghosts and with the plane's
                 ghosts in the height functions (hook), to t = 1, and one geometry to t = 20 (the long-time growth of
                 docs/notes/115, section 5); the cells' and the faces' velocities.

    python -m cases.r0p2_corner          # docs/figures/r0p2_corner.json (about 5 min, 8 processes)
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import sys

import numpy as np

GEOMETRIES = ([(th, 0.0137, n) for th in (0.0, 10.0) for n in (32, 64, 128)]
              + [(th, off, n) for th in (26.565, 45.0) for off in (0.0137, 0.1171, 0.2313) for n in (32, 64, 128)])
FAILING = [(45.0, 0.0137, 32), (45.0, 0.0137, 64), (26.565, 0.2313, 32)]


def _H():
    importlib.import_module("basilisk_cupy.interface.heights")
    return sys.modules["basilisk_cupy.interface.heights"]           # (the package exports a function of that name)


def _plane(th_deg, off):
    t = math.radians(th_deg)
    nx, ny = math.sin(t), math.cos(t)
    return (lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off))), (nx, ny)


def _exact_pad(phi, n, width):
    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction

    h = 1.0 / n
    g = CartesianGrid((n + 2 * width,) * 2, length=((n + 2 * width) * h,) * 2, origin=(-width * h, -width * h))
    return to_numpy(fraction(g, phi))


def _heights_with(c, bc, ext, corners_only=False):
    """heights() with the field outside the box taken from ``ext`` (the plane, width 4 around the box) instead of the
    walls' mirror; the interior is ``c`` (the current volume fraction). corners_only: the mirror everywhere except
    the four corner blocks."""
    H = _H()
    G, n = 2, c.shape
    F = ext.copy()
    F[4:-4, 4:-4] = c
    cpad = F[2:-2, 2:-2]
    if corners_only:
        mirror = H.pad_bc(c, bc, G)
        keep = np.zeros_like(mirror, bool)
        for a in (slice(0, G), slice(-G, None)):
            for b in (slice(0, G), slice(-G, None)):
                keep[a, b] = True
        cpad = np.where(keep, cpad, mirror)
    hs = []
    for axis in range(2):
        h = None
        for j in (-1, 1):
            idx = [slice(2, -2), slice(2, -2)]
            idx[axis] = slice(2 + 2 * j, 2 + 2 * j + n[axis] + 4)
            spad = F[tuple(idx)]
            if corners_only:
                spad = np.where(keep, spad, H.pad_bc(H._sl(cpad, axis, 2 * j, G, n), bc, G))
            h = H._half_column(np, cpad, spad, h, axis, j, n)
        hs.append(h)
    return H._column_propagation(np, hs)


def potential(th, off, n, mode="mirror"):
    """max |phi| / (rho2 - rho1) over the interfacial cells at t = 0, the bad cells (|phi| > 1e-12) and the column each
    uses ("x": the x-heights, i.e. the dominant normal along x)."""
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction

    from cases.r0p2_pressure_momentum import _bc

    H = _H()
    phi, (nx, ny) = _plane(th, off)
    g = CartesianGrid((n, n))
    f0 = to_numpy(fraction(g, phi))
    f = CellField.from_interior(g, f0, _bc(()))
    hs = None if mode == "mirror" else _heights_with(f0, f.bc, _exact_pad(phi, n, 4), mode == "exact corners only")
    p = to_numpy(H.position(f0, f.bc, g, (-nx, -ny), (0.5, 0.5 + off), hs=hs))
    ok = p < H.NODATA / 2
    bad = np.argwhere(ok & (np.abs(p) > 1e-12))
    cp1 = H.pad_bc(f0, f.bc, 1)
    gx = np.abs(cp1[2:, 1:-1] - cp1[:-2, 1:-1])
    gy = np.abs(cp1[1:-1, 2:] - cp1[1:-1, :-2])
    cells = [{"cell": [int(i), int(j)], "f": float(f0[i, j]), "column": "x" if not gx[i, j] < gy[i, j] else "y",
              "phi": float(p[i, j])} for i, j in bad]
    return {"theta": th, "offset": off, "n": n, "mode": mode, "max_phi": float(np.abs(np.where(ok, p, 0)).max()),
            "bad": cells}


def census():
    path = os.path.join("docs", "figures", "r0p2_pressure_momentum.json")
    res = json.load(open(path))
    fails = {}
    for o in res["hydrostatic"]:
        if o["gravity"] != "reduced" or o["ratio"] != 1000.0 or not o["conserving"]:
            continue
        key = (o["theta"], o.get("offset", 0.0137), o["n"])
        fails.setdefault(key, []).append(max(h["cells"] for h in o["history"]) > 1e-8)
    rows = []
    for th, off, n in GEOMETRIES:
        p = potential(th, off, n)
        rows.append({"theta": th, "offset": off, "n": n, "max_phi": p["max_phi"], "bad_cells": len(p["bad"]),
                     "failing_kinds": sum(fails.get((th, off, n), [])), "kinds": len(fails.get((th, off, n), []))})
    return rows


@contextlib.contextmanager
def plane_ghosts(th, off, n):
    """Hook: the height functions take the plane (instead of the walls' mirror) outside the box."""
    H = _H()
    ext = _exact_pad(_plane(th, off)[0], n, 4)
    saved = H.heights
    H.heights = lambda c, bc: _heights_with(np.asarray(c), bc, ext)
    try:
        yield
    finally:
        H.heights = saved


def balance(th, off, n, ghosts="mirror", t_end=1.0, samples=(0.01, 0.1, 0.25, 0.5, 1.0)):
    from cases.r0p2_pressure_momentum import hydrostatic

    with plane_ghosts(th, off, n) if ghosts == "plane" else contextlib.nullcontext():
        o = hydrostatic("smooth", th, "reduced", n, t_end=t_end, offset=off, samples=samples)
    o["ghosts"] = ghosts
    return o


def sliver_periodic(nv=(1, 2), off=0.0137, n=32, steps=40, noise=1e-16, dt_max="cfl"):
    """Audit on PR #165 (its point 3): away from walls. A periodic oblique band (two plane interfaces), no force,
    a round-off velocity: do round-off f-slivers (0 < f < 1e-12 or 1 - 1e-12 < f < 1) appear, and does the height
    functions' position of the REAL interfacial cells (1e-6 < f < 1 - 1e-6) leave the exact lines? Returned: the
    largest distance (in cells) of a real interfacial cell's position from the nearest exact interface, at t = 0 and
    over the run, and the largest number of sliver cells. Correction (#193, audit on #178): the first version had no
    step bound (``dt_max=None``): with a round-off velocity the CFL step grew to 3e13 and the noise's CFL number
    passed 1/2, which made the slivers and the jumps; with ``DT = 0.5 / n`` (the default now) neither appears."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    H = _H()
    a, b = nv
    r = math.hypot(a, b)
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    if dt_max == "cfl":  # #193: a step bound (0.5 h, as the other runs); None reproduces the first version
        ns.DT = 0.5 / n

    def phi(x, y):
        z = (a * x + b * y - off) % 1.0
        return np.where(z < 0.5, np.minimum(z, 0.5 - z), -np.minimum(z - 0.5, 1 - z)) / r

    f0 = to_numpy(fraction(grid, phi, refine=8))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3,
                  conserving=True)
    rng = np.random.default_rng(0)
    for c in ns.u.components:
        c.interior = noise * rng.standard_normal((n, n))

    def dev():
        f = to_numpy(tp.f.interior)
        p = to_numpy(H.position(tp.f.interior, tp.f.bc, grid, (a, b), (0.0, 0.0)))     # a x_I + b y_I
        ok = (p < H.NODATA / 2) & (f > 1e-6) & (f < 1 - 1e-6)
        d = ((p - off + 0.25) % 0.5) - 0.25
        return float(np.abs(np.where(ok, d, 0.0)).max()) / r * n

    d0, dm, sl, first = dev(), 0.0, 0, None
    ns.start()
    for k in range(steps):
        ns.step()
        f = to_numpy(tp.f.interior)
        sl = max(sl, int(((f > 0) & (f < 1e-12)).sum() + ((f < 1) & (f > 1 - 1e-12)).sum()))
        d = dev()
        if first is None and d > 0.1:
            first = k + 1
        dm = max(dm, d)
    return {"n": list(nv), "offset": off, "N": n, "dt_max": dt_max, "dt_last": float(ns.clock.dt), "dev0_cells": d0, "devmax_cells": dm, "first_step_over_0.1": first,
            "slivers_max": sl, "half_period_cells": 0.25 / r * n}


def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return balance(*a[0], **a[1])


def main() -> None:
    import multiprocessing as mp

    J = [((th, off, n), {"ghosts": gh}) for th, off, n in FAILING for gh in ("mirror", "plane")]
    J += [((45.0, 0.0137, 32), {"ghosts": gh, "t_end": 20.0, "samples": (1.0, 5.0, 10.0, 20.0)})
          for gh in ("mirror", "plane")]
    with mp.get_context("spawn").Pool(len(J)) as pool:
        bal = pool.map(_job, J)
    res = {"sliver_periodic": [sliver_periodic(nv, off, n) for nv in ((1, 1), (1, 2), (2, 1))
                               for off in (0.0137, 0.1171) for n in (32, 64)],
           "census": census(),
           "ghosts": [potential(th, off, n, m) for th, off, n in FAILING
                      for m in ("mirror", "exact", "exact corners only")],
           "balance": bal}
    with open(os.path.join("docs", "figures", "r0p2_corner.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote docs/figures/r0p2_corner.json")


if __name__ == "__main__":
    main()


def plot() -> None:
    """docs/figures/r0p2_corner.png: (a) the cells' max velocity at rest, mirrored against the plane's ghosts;
    (b) max |phi| at t = 0 for docs/notes/115's 24 geometries, the failing ones marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(os.path.join("docs", "figures", "r0p2_corner.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b in R["balance"]:
        t = [h["t"] for h in b["history"]]
        v = [max(h["cells"], 1e-17) for h in b["history"]]
        lab = "%g deg, d=%g, N=%d, %s" % (b["theta"], b["offset"], b["n"], b["ghosts"])
        ax[0].loglog(t, v, "-o" if b["ghosts"] == "mirror" else "--x", ms=4, label=lab)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("max |u| (cells)")
    ax[0].set_title("(a) two layers at rest, reduced gravity")
    ax[0].legend(fontsize=6)
    c = R["census"]
    ax[1].semilogy(range(len(c)), [max(r["max_phi"], 1e-17) for r in c], "o", color="0.4")
    bad = [k for k, r in enumerate(c) if r["failing_kinds"]]
    ax[1].semilogy(bad, [c[k]["max_phi"] for k in bad], "s", mfc="none", ms=10, color="tab:red",
                   label="unbalanced in note 115")
    ax[1].set_xticks(range(len(c)))
    ax[1].set_xticklabels(["%g/%g/%d" % (r["theta"], r["offset"], r["n"]) for r in c], rotation=90, fontsize=5)
    ax[1].set_ylabel("max |phi| / ((rho2 - rho1) G) at t = 0")
    ax[1].set_title("(b) the potential of the plane, 24 geometries")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0p2_corner.png"), dpi=130)

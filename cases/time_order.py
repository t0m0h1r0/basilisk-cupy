"""Temporal order of the multistep IMEX schemes (:class:`IMEXMultistepNS`) and IMEX-RK tableaux (:class:`IMEXRKNS`) by self-convergence (docs/notes/51).

Periodic unit box, N = 16 (fd2 operators), mu = 0.05, two-mode initial velocity from the stream function
``sin(2 pi x) sin(4 pi y) + 0.5 cos(4 pi x) sin(2 pi y)`` (kx != ky; not an exact solution, so the error is
measured against the same scheme with ``kref`` steps). Runs go to separate processes (``cases.parallel``).

    python -m cases.time_order [--jobs N]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.imex_multistep import IMEXMultistepNS

SCHEMES = ["sbdf1", "sbdf2", "sbdf3", "sbdf4", "cnab2", "mcnab2", "cnlf", ("am3", "ab3"), ("theta0.6", "ab2"),
           ("bdf2", "ab2"), "ars111", "ars121", "ars222", "ars233", "ars343", "ars443"]


def run(scheme, n: int = 16, steps: int = 20, t_end: float = 0.2, mu: float = 0.05, stokes: bool = False):
    grid = CartesianGrid((n, n))
    x, y = grid.meshgrid()
    tp = 2 * math.pi
    # u = d psi/dy, v = -d psi/dx
    u = tp * 2 * np.sin(tp * x) * np.cos(2 * tp * y) + 0.5 * tp * np.cos(2 * tp * x) * np.cos(tp * y)
    v = -tp * np.cos(tp * x) * np.sin(2 * tp * y) + 0.5 * 2 * tp * np.sin(2 * tp * x) * np.sin(tp * y)
    kw = dict(periodic=(0, 1), mu=mu, stokes=stokes, tolerance=1e-12, fixed_dt=t_end / steps)
    if isinstance(scheme, str) and scheme.startswith("ars"):
        from basilisk_cupy.time.imex_rk import IMEXRKNS

        ns = IMEXRKNS(grid, tableau=scheme, **kw)
    else:
        ns = IMEXMultistepNS(grid, scheme=scheme, **kw)
    ns.u[0].interior = 0.05 * u
    ns.u[1].interior = 0.05 * v
    ns.start(events=[t_end])
    ns.run(t_end)
    return np.stack([ns.u[0].interior, ns.u[1].interior])


def orders(scheme, ks=(10, 20, 40, 80), kref=1280, jobs=None, **kw):
    from cases.parallel import pmap

    res = pmap(run, [dict(scheme=scheme, steps=k, **kw) for k in (*ks, kref)], jobs)
    ref = res[-1]
    errs = [float(np.abs(r - ref).max()) for r in res[:-1]]
    return errs, [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--stokes", action="store_true", help="drop the convection term")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--schemes", nargs="+", default=None, help="subset of SCHEMES (names; pairs as impl+expl)")
    args = ap.parse_args()
    from cases.parallel import pmap

    ks, kref = (10, 20, 40, 80), 1280
    schemes = SCHEMES if not args.schemes else [tuple(n.split("+")) if "+" in n else n for n in args.schemes]
    items = [dict(scheme=s, steps=k, stokes=args.stokes, n=args.n) for s in schemes for k in (*ks, kref)]
    res = pmap(run, items, args.jobs)
    m = len(ks) + 1
    for i, s in enumerate(schemes):
        block = res[i * m:(i + 1) * m]
        errs = [float(np.abs(r - block[-1]).max()) for r in block[:-1]]
        rates = [math.log2(errs[j] / errs[j + 1]) for j in range(len(errs) - 1)]
        name = s if isinstance(s, str) else "+".join(s)
        print(f"{name:12s} " + " ".join(f"{e:.2e}" for e in errs) + "   rates " + " ".join(f"{r:.2f}" for r in rates))


if __name__ == "__main__":
    main()

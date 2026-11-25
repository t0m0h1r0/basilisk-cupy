"""Stability limit of the static drop against the grid and the density ratio, with Basilisk's splitting
(K = 0), one coupled iteration (K = 1) and converged iterations (docs/notes/36).

Static drop R = 0.25, sigma = 1, mu = 0.1 (both phases), GFM, solvers to 1e-14, 800 steps; the limit (a
multiple of tension.h's step) by bisection. Converged: ``coupled_iterations=30``, ``coupled_tolerance=1e-10``
on ``dt |g_{k+1} - g_k|``.

    python -m cases.coupled_limits --n 32 --rho2 0.01 --scheme converged
"""

from __future__ import annotations

import argparse
import json
import warnings

from cases.capillary_dt_scan import growth

SCHEMES = {"K0": {}, "K1": {"coupled_iterations": 1},
           "converged": {"coupled_iterations": 30, "coupled_tolerance": 1e-10, "coupled_on_failure": "ignore"}}


def limit(n, rho2, scheme, lo=0.5, hi=3.0, bisections=6, steps=800, backend="numpy"):
    """(stable, unstable) factors bracketing the limit."""
    warnings.filterwarnings("ignore")

    def stable(f):
        return growth(n, rho2, f, steps=steps, mu=0.1, ns_kw=dict(SCHEMES[scheme]), solver_tolerance=1e-14,
                      backend=backend)["stable"]

    if not stable(lo):
        return (0.0, lo)
    if stable(hi):
        return (hi, float("inf"))
    for _ in range(bisections):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--rho2", type=float, default=0.01)
    ap.add_argument("--scheme", default="converged", choices=list(SCHEMES))
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--bisections", type=int, default=6)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=3.0)
    args = ap.parse_args()
    lo, hi = limit(args.n, args.rho2, args.scheme, lo=args.lo, hi=args.hi, bisections=args.bisections,
                   backend=args.backend)
    print(json.dumps({"n": args.n, "rho2": args.rho2, "scheme": args.scheme, "limit": [lo, hi]}), flush=True)


if __name__ == "__main__":
    main()

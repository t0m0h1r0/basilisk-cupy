"""Grid convergence of the oscillating drop (14b) against the 2-D viscous mode (beta = -0.179, omega = 59.555),
docs/notes/63: TwoPhaseIMEX (ab2, xi-SDF every 10, height-function curvature) and VOF, N = 32 / 64 / 128.

    python -m cases.phase3_drop_convergence [--jobs N]
"""

from __future__ import annotations

import argparse


def run(variant: str, n: int):
    from cases.phase3_drop_bubble import run as rd, summarise_drop

    r = rd("drop", variant, n)
    beta, om, res, vol, umax = summarise_drop(r)
    return {"beta": beta, "omega": om, "fit_res": res, "umax": umax, "steps": r["steps"], "wall": r["wall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--reinit-time", action="store_true", help="re-initialise at the same physical interval")
    args = ap.parse_args()
    from cases.parallel import pmap

    def show(k, it, r):
        print(f"{it['variant']:22s} N {it['n']:3d}: beta {r['beta']:+.3f} ({r['beta'] / -0.179428:.2f} x) omega "
              f"{r['omega']:.3f} ({100 * (r['omega'] / 59.554813 - 1):+.2f}%) fit res {r['fit_res']:.3f} max|u| "
              f"{r['umax']:.3f} ({r['steps']} steps, {r['wall']:.0f}s)", flush=True)

    items = [dict(variant=v, n=n) for v in ("imex-ab2-xi10-heights", "vof") for n in (32, 64, 128)]
    if args.reinit_time:
        items = [dict(variant="imex-ab2-xi20-heights", n=64), dict(variant="imex-ab2-xi60-heights", n=128),
                 dict(variant="imex-ab2-xi60-heights", n=64)]
    pmap(run, items, args.jobs, on_result=show)


if __name__ == "__main__":
    main()

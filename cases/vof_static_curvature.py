"""Static drop (R = 0.25, sigma = 1, mu = 0.1, density ratio 1000, t = 0.5) with exact initial fractions: tension.h
against the CCD column curvature (docs/notes/66), CSF and ghost-fluid jump, one run after another on the GPU.

    python -m cases.vof_static_curvature [--backend cupy] [--n 64 128]
"""

from __future__ import annotations

import argparse


def run(n: int, interface: str, curvature: str, backend: str = "numpy", t_end: float = 0.5, DT: float | None = None,
        scheme: str = "sbdf2"):
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX
    from cases.vof_curvature_ccd import circle_fractions

    g = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    x, y = (to_numpy(v) for v in g.meshgrid())
    ns = TwoPhaseIMEX(g, lambda x, y: 0.0625 - x * x - y * y, rho1=1.0, rho2=0.001, mu1=0.1, mu2=0.1, sigma=1.0,
                      tolerance=1e-8, interface=interface, vof_curvature=curvature, scheme=scheme,
                      **({"DT": DT} if DT else {}))
    ns.vof.interior = g.xp.asarray(circle_fractions(n, 0.0, 0.0, 0.25))
    ns.start()
    um = 0.0
    hist = []
    while ns.clock.t < t_end - 1e-12:
        ns.DT = min(t_end - ns.clock.t, DT or 1e30)
        ns.step()
        u = ns.max_velocity()
        um = max(um, u)
        if ns.clock.i % 100 == 0:
            hist.append((ns.clock.i, round(float(ns.clock.t), 4), float(u)))
        if not u < 50:
            break
    p = to_numpy(ns.pressure())
    r = np.hypot(x, y)
    return {"jump": float(p[r < 0.1].mean() - p[r > 0.4].mean()), "umax": um, "uend": ns.max_velocity(),
            "steps": ns.clock.i, "hist": hist}


def stability(n: int = 128):
    """The N = 128 blow-up of the IMEX VOF (tension.h's curvature): the step halved and SBDF1 (CPU, in parallel)."""
    from cases.parallel import pmap

    items = [dict(n=n, interface="vof", curvature="basilisk"),
             dict(n=n, interface="vof", curvature="basilisk", DT=1.3e-4),
             dict(n=n, interface="vof", curvature="basilisk", scheme="sbdf1")]
    pmap(run, items, 3, on_result=lambda k, it, r: print(it, f"jump err {r['jump'] - 4:+.2e} umax {r['umax']:.2e}",
                                                          r["hist"][:3], "...", r["hist"][-3:], flush=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--n", type=int, nargs="*", default=[64, 128])
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--interfaces", nargs="*", default=["vof", "vof-gfm"])
    args = ap.parse_args()
    if args.stability:
        stability()
        return
    for n in args.n:
        for itf in args.interfaces:
            for curv in ("basilisk", "ccd"):
                r = run(n, itf, curv, args.backend)
                print(f"N {n:3d} {itf:7s} {curv:8s}: jump err {r['jump'] - 4:+.2e} umax {r['umax']:.2e} "
                      f"uend {r['uend']:.2e} ({r['steps']} steps)", flush=True)


if __name__ == "__main__":
    main()

"""Phase 3, benchmark 1: the paper's capillary wave (14a; water--air, lambda = 10 mm, x periodic / y walls) against
the two-fluid normal mode (``cases.two_fluid_modes``), docs/notes/58.

The interface amplitude (first Fourier coefficient of the crossing heights) is sampled 100 times per period over
5 periods; ``a(t) = A e^{beta t} cos(omega t + phi)`` is fitted by least squares (beta and omega by a
Gauss--Newton refinement of the peak estimate). Also reported: the paper's measure, the amplitude ratio at
t = T, and the volume drift.

    python -m cases.phase3_capwave [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

T = 0.046742983863


def fit(t, a):
    """Least-squares ``A e^{beta t} cos(omega t + phi)`` (returns beta, omega, rms residual / amplitude)."""
    t, a = np.asarray(t), np.asarray(a)
    w0 = 2 * math.pi / T
    best = None
    for beta in np.linspace(-20, 5, 101):
        for om in np.linspace(0.8 * w0, 1.2 * w0, 81):
            e = np.exp(beta * t)
            M = np.stack([e * np.cos(om * t), e * np.sin(om * t)], axis=1)
            c, *_ = np.linalg.lstsq(M, a, rcond=None)
            r = float(np.sum((M @ c - a) ** 2))
            if best is None or r < best[0]:
                best = (r, beta, om)
    _, beta, om = best
    for _ in range(30):  # local refinement
        improved = False
        for db, do in ((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)):
            b2, o2 = beta + db, om + do
            e = np.exp(b2 * t)
            M = np.stack([e * np.cos(o2 * t), e * np.sin(o2 * t)], axis=1)
            c, *_ = np.linalg.lstsq(M, a, rcond=None)
            r = float(np.sum((M @ c - a) ** 2))
            if r < best[0]:
                best, beta, om, improved = (r, b2, o2), b2, o2, True
        if not improved:
            break
    return beta, om, math.sqrt(best[0] / len(a)) / float(np.max(np.abs(a)))


def run(variant: str, n: int = 32, periods: int = 5):
    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run as bench

    kw = {"xi10": dict(reinit_every=10, reinit_method="xi_sdf")}
    opts = {
        "imex": dict(**kw["xi10"]),
        "imex-ab2": dict(transport="ab2", **kw["xi10"]),
        "rk-ars222": dict(tableau="ars222", **kw["xi10"]),
        "imex-noreinit": {},
        "vof": dict(method="vof"),
    }[variant]
    t0 = time.time()
    r = bench("capwave", n=n, t_end=periods * T, samples=100 * periods, **opts)
    s = [d for d in r["series"] if "amplitude" in d]
    t = np.array([d["t"] for d in s])
    a = np.array([d["amplitude"] for d in s])
    beta, om, res = fit(t, a)
    k = int(round(100))
    return {"beta": beta, "omega": om, "fit_res": res, "ratio_T": float(a[k] / a[0]) if len(a) > k else float("nan"),
            "volume": s[-1]["mass"] / s[0]["mass"] - 1.0, "failed": bool(r["series"][-1].get("failed")),
            "t_end": float(r["series"][-1]["t"]), "steps": r["steps"], "wall": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap
    from cases.two_fluid_modes import mode

    s = mode(2 * math.pi / 0.01, 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    print(f"normal mode: beta {s.real:.4f} 1/s, omega {s.imag:.4f} 1/s; ratio at T {math.exp(s.real * T):.5f} "
          f"(paper 0.793553)")
    items = [dict(variant=v, n=n) for n in (32, 64) for v in ("imex", "imex-ab2", "rk-ars222", "imex-noreinit", "vof")]
    for it, r in zip(items, pmap(run, items, args.jobs)):
        print(f"N {it['n']:3d} {it['variant']:14s}: beta {r['beta']:+8.3f} ({r['beta'] / s.real:5.2f} x) omega "
              f"{r['omega']:8.3f} ({100 * (r['omega'] / s.imag - 1):+.2f}%) ratio(T) {r['ratio_T']:.4f} fit res "
              f"{r['fit_res']:.3f} volume {r['volume']:+.1e} {'FAILED at t=%.3f' % r['t_end'] if r['failed'] else ''}"
              f" ({r['steps']} steps, {r['wall']:.0f}s)")


if __name__ == "__main__":
    main()

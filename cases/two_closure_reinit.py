"""The paper's two scalar closures of the re-initialisation (lambda_Gamma, eps_eff) in a flow (docs/notes/54).

``CLSInterface.reinitialize(method="xi_sdf_vw")`` against the default ``"xi_sdf"`` (one uniform shift closing
the diffuse mass) and no re-initialisation, every ``every`` steps of :class:`TwoPhaseIMEX`:

* capillary wave (capwave.c, 16 points per wavelength): RMS error against Prosperetti, damping, period;
* static drop (R = 0.25, sigma = 1, mu = 0.1, N = 32, t = 0.5): jump, parasitic velocity.

The width ``eps_eff / eps`` of every re-initialisation is recorded.

    python -m cases.two_closure_reinit [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX


def _record(ns, widths):
    cls = ns.cls
    orig = cls.reinitialize

    def wrapped(*a, **k):
        orig(*a, **k)
        info = getattr(cls, "reinit_info", None)
        if info is not None:
            widths.append(info["eps_eff"])
            cls.reinit_info = None

    cls.reinitialize = wrapped


def capwave(method: str | None, every: int = 10, n: int = 32):
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay
    from cases.capwave_modes import period_decay

    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    x, y = grid.meshgrid()
    k = 2.0 * math.pi
    dist = (y - 0.01 * np.cos(k * x)) / np.sqrt(1.0 + (0.01 * k * np.sin(k * x)) ** 2)
    ns = TwoPhaseIMEX(grid, dist, mu1=cw.MU, mu2=cw.MU, sigma=1.0, tolerance=1e-6, fixed_dt=cw.STEP if n <= 32 else cw.STEP / 2,
                      reinit_every=every if method else 0, reinit_method=method or "xi_sdf")
    widths: list = []
    _record(ns, widths)
    times = cw.event_times()
    amps = [cw._cls_amplitude(np, ns.psi, grid)]
    ns.start(events=times[1:])
    while len(amps) < len(times):
        ns.step()
        if abs(ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, ns.clock.t):
            amps.append(cw._cls_amplitude(np, ns.psi, grid))
    P = np.array(cw.PROSPERETTI)
    a = np.array(amps)
    m = min(len(a), len(P))
    err = math.sqrt(float(np.mean((a[:m] - P[:m, 1]) ** 2))) / 0.01
    pr, dd = decay(a, P[:len(a), 0])
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, cw.MU)
    return {"error": err, "damping": math.log(dd) / math.log(dt_) - 1, "period": float(pr / pt - 1),
            "widths": widths, "amplitudes": amps}


def drop(method: str | None, every: int = 10, n: int = 32, rho2: float = 1.0, t_end: float = 0.5):
    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = TwoPhaseIMEX(grid, 0.25 - np.hypot(x, y), rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, sigma=1.0,
                      tolerance=1e-8, reinit_every=every if method else 0, reinit_method=method or "xi_sdf")
    widths: list = []
    _record(ns, widths)
    ns.start()
    m0 = ns.mass()
    umax = 0.0
    while ns.clock.t < t_end - 1e-12:
        ns.DT = t_end - ns.clock.t
        ns.step()
        umax = max(umax, ns.max_velocity())
    p = ns.pressure()
    r = np.hypot(x, y)
    return {"jump": float(p[r < 0.1].mean() - p[r > 0.4].mean()), "u_end": ns.max_velocity(), "u_max": umax,
            "mass": ns.mass() / m0 - 1.0, "widths": widths}


def _dispatch(kind, kw):
    return capwave(**kw) if kind == "cw" else drop(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    methods = (None, "xi_sdf", "xi_sdf_vw")
    items = [dict(kind="cw", kw=dict(method=m, n=n)) for n in (32, 64) for m in methods]
    items += [dict(kind="drop", kw=dict(method=m, rho2=r)) for r in (1.0, 0.001) for m in methods]
    res = pmap(_dispatch, items, args.jobs)
    for it, r in zip(items, res):
        w = r["widths"]
        ws = f"eps_eff/eps: n={len(w)} min {min(w):.3f} median {np.median(w):.3f} max {max(w):.3f}" if w else ""
        if it["kind"] == "cw":
            print(f"capwave N={it['kw']['n']} {it['kw']['method']}: RMS {r['error']:.4f} damping "
                  f"{100 * r['damping']:+.1f}% period {100 * r['period']:+.2f}%  {ws}")
        else:
            print(f"drop rho2={it['kw']['rho2']:g} {it['kw']['method']}: jump {r['jump']:.5f} u_end {r['u_end']:.2e} "
                  f"u_max {r['u_max']:.2e} mass {r['mass']:.1e}  {ws}")


if __name__ == "__main__":
    main()

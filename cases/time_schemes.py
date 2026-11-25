"""Multistep schemes and interface-transport velocities of :class:`TwoPhaseIMEX` compared (docs/notes/51).

* capillary wave (capwave.c, equal densities, 16 points per wavelength): damping and period errors against
  the normal mode for every (scheme, transport) at fixed steps ``STEP / d``;
* static drop (R = 0.25, sigma = 1, mu = 0.1, walls, N = 32): stability limit of the fixed step, a multiple
  of tension.h's bound ``sqrt(rho_m h^3 / (pi sigma))``, by bisection over 600 steps (solvers to 1e-11).

Every run is a separate process (``cases.parallel``).

    python -m cases.time_schemes [--jobs N] [--out docs/figures] [--skip-limits]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings

import numpy as np

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

SCHEMES = ("sbdf1", "sbdf2", "sbdf3", "cnab2", "mcnab2", "cnlf")
TRANSPORTS = ("frozen", "ab2", "ab3", "pc")


def _stepper(grid, phi, scheme: str, transport: str, psi_integrator: str = "ssprk3", **kw):
    """``TwoPhaseIMEX`` for a multistep scheme, ``TwoPhaseIMEXRK`` for an ARS tableau (``transport`` unused)."""
    if scheme.startswith("ars"):
        from basilisk_cupy.time.two_phase_imex_rk import TwoPhaseIMEXRK

        return TwoPhaseIMEXRK(grid, phi, tableau=scheme, **kw)
    return TwoPhaseIMEX(grid, phi, scheme=scheme, transport=transport, psi_integrator=psi_integrator, **kw)


def capwave(n: int = 32, scheme: str = "sbdf2", transport: str = "frozen", d: int = 1,
            psi_integrator: str = "ssprk3"):
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay
    from cases.capwave_modes import period_decay

    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    x, y = grid.meshgrid()
    k = 2.0 * math.pi
    dist = (y - 0.01 * np.cos(k * x)) / np.sqrt(1.0 + (0.01 * k * np.sin(k * x)) ** 2)
    ns = _stepper(grid, dist, scheme, transport, psi_integrator, mu1=cw.MU, mu2=cw.MU, sigma=1.0, tolerance=1e-6,
                  fixed_dt=cw.STEP / d)
    times = cw.event_times()
    amps = [cw._cls_amplitude(np, ns.psi, grid)]
    ns.start(events=times[1:])
    t0 = time.time()
    try:
        while len(amps) < len(times):
            ns.step()
            if abs(ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, ns.clock.t):
                amps.append(cw._cls_amplitude(np, ns.psi, grid))
    except (ValueError, FloatingPointError) as e:
        return {"error": float("nan"), "damping": float("nan"), "period": float("nan"), "fail": str(e)}
    P = np.array(cw.PROSPERETTI)
    a = np.array(amps)
    m = min(len(a), len(P))
    err = math.sqrt(float(np.mean((a[:m] - P[:m, 1]) ** 2))) / 0.01
    pr, dd = decay(a, P[:len(a), 0])
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, cw.MU)
    return {"error": err, "damping": math.log(dd) / math.log(dt_) - 1, "period": float(pr / pt - 1),
            "steps": ns.clock.i, "wall": time.time() - t0}


def tension_bound(n: int, rho2: float, sigma: float = 1.0) -> float:
    h = 1.0 / n
    return math.sqrt(0.5 * (1.0 + rho2) * h ** 3 / (math.pi * sigma))


def drop_stable(n: int, rho2: float, factor: float, scheme: str, transport: str, steps: int = 600) -> bool:
    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = _stepper(grid, 0.25 - np.hypot(x, y), scheme, transport, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1,
                  sigma=1.0, tolerance=1e-11, fixed_dt=factor * tension_bound(n, rho2), capillary_dt_factor=1e6)
    ns.start()
    us = []
    for _ in range(steps):
        ns.step()
        u = ns.max_velocity()
        us.append(u)
        if not math.isfinite(u) or u > 1.0:
            return False
    tail = np.log(np.maximum(np.array(us[len(us) // 2:]), 1e-300))
    return float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) < 1e-3


def limit(n: int = 32, rho2: float = 1.0, scheme: str = "sbdf2", transport: str = "frozen",
          lo: float = 0.25, hi: float = 4.0, bisections: int = 5):
    """(stable, unstable) multiples of tension.h's step bracketing the limit."""
    if not drop_stable(n, rho2, lo, scheme, transport):
        return (0.0, lo)
    if drop_stable(n, rho2, hi, scheme, transport):
        return (hi, float("inf"))
    for _ in range(bisections):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if drop_stable(n, rho2, mid, scheme, transport) else (lo, mid)
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--skip-limits", action="store_true")
    ap.add_argument("--schemes", nargs="+", default=None,
                    help="schemes to run (default: the multistep set; ars* names use TwoPhaseIMEXRK)")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    schemes = tuple(args.schemes) if args.schemes else SCHEMES
    from cases.parallel import pmap

    divs = (1, 2, 4, 8)
    def transports(s):
        return ("frozen",) if s.startswith("ars") else TRANSPORTS

    cw_items = [dict(scheme=s, transport=t, d=d) for s in schemes for t in transports(s) for d in divs]
    if not args.schemes:
        cw_items += [dict(scheme="sbdf2", transport="pc", d=1, psi_integrator=r) for r in ("euler", "ssprk2", "rk4")]
    lim_items = [] if args.skip_limits else [dict(rho2=r, scheme=s, transport=t) for r in (1.0, 0.001)
                                             for s in schemes for t in transports(s)[:1] + (
                                                 () if s.startswith("ars") else ("ab2", "pc"))]
    t0 = time.time()
    res = pmap(_dispatch, [dict(kind="cw", kw=i) for i in cw_items] + [dict(kind="lim", kw=i) for i in lim_items],
               args.jobs)
    print(f"wall {time.time() - t0:.0f}s")
    cw_res, lim_res = res[:len(cw_items)], res[len(cw_items):]
    print("capillary wave, 16 points: scheme transport d psi_integrator error damping period steps wall")
    for c, r in zip(cw_items, cw_res):
        print(f"{c['scheme']:7s} {c['transport']:6s} {c['d']} {c.get('psi_integrator', 'ssprk3'):7s} "
              f"{r['error']:.4f} {100 * r['damping']:+.2f}% {100 * r['period']:+.2f}% {r.get('steps', '-')} "
              f"{r.get('wall', 0):.0f}s {r.get('fail', '')}")
    print("static drop N = 32, stability limit (multiple of tension.h's step): rho2 scheme transport lo hi")
    for c, r in zip(lim_items, lim_res):
        print(f"{c['rho2']:g} {c['scheme']:7s} {c['transport']:6s} {r[0]:.3f} {r[1]:.3f}")
    with open(f"{args.out}/time_schemes{args.tag}.json", "w") as f:
        json.dump({"cw": [[c, r] for c, r in zip(cw_items, cw_res)],
                   "lim": [[c, list(r)] for c, r in zip(lim_items, lim_res)]}, f, default=float)
    if not args.schemes:
        figure(cw_items, cw_res, lim_items, lim_res, args.out)


def _dispatch(kind: str, kw: dict):
    return capwave(**kw) if kind == "cw" else limit(**kw)


def figure(cw_items, cw_res, lim_items, lim_res, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cases.capillary_wave import STEP

    ncol = 3 if lim_items else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.0))
    colors = dict(zip(SCHEMES, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, transport in zip(axes[:2], ("frozen", "pc")):
        for s in SCHEMES:
            pts = sorted((STEP / c["d"], 100 * r["damping"]) for c, r in zip(cw_items, cw_res)
                         if c["scheme"] == s and c["transport"] == transport and "psi_integrator" not in c
                         and math.isfinite(r["damping"]))
            if pts:
                ax.semilogx(*zip(*pts), "o-", color=colors[s], label=s)
        ax.set_xlabel("Δt")
        ax.set_ylabel("damping error per period [%]")
        ax.set_title(f"capillary wave, 16 points; ψ by {'u_f^n (paper)' if transport == 'frozen' else 'predictor-corrector'}")
        ax.legend(fontsize=8)
    lo = min(min(ax.get_ylim()[0] for ax in axes[:2]), -20)
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    if lim_items:
        ax = axes[2]
        labels = [f"{s}" for s in SCHEMES]
        width = 0.25
        for k, (rho2, tr) in enumerate([(1.0, "frozen"), (0.001, "frozen"), (0.001, "pc")]):
            vals = [next(r[0] for c, r in zip(lim_items, lim_res)
                         if c["scheme"] == s and c["rho2"] == rho2 and c["transport"] == tr) for s in SCHEMES]
            ax.bar(np.arange(len(SCHEMES)) + (k - 1) * width, vals, width,
                   label=f"density ratio {1 / rho2:g}, ψ by {tr}")
        ax.set_xticks(np.arange(len(SCHEMES)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("stable step / tension.h bound")
        ax.set_title("static drop N = 32: stability limit")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/time_schemes.png", dpi=150)
    print("wrote", f"{out}/time_schemes.png")


if __name__ == "__main__":
    main()

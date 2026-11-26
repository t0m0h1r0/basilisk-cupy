"""The paper's standard two-phase step (:class:`TwoPhaseIMEX`) against ``CenteredNS`` + CLS/GFM (docs/notes/50).

Both use the same CLS (FCCD face values of order 6, corrected divergence), the sharp GFM pressure jump
with fd2 faces and the CCD viscous solver; only the time integration differs (Basilisk's centred
splitting against IMEX-BDF2 with incremental pressure correction).

* static drop: R = 0.25 in the unit box with walls, sigma = 1, mu = 0.1, rho1 = 1, rho2 = 1 ... 0.001,
  up to t = 0.5;
* capillary wave (Basilisk ``capwave.c``, equal densities): RMS error against Prosperetti, damping and
  period of the first five peaks against the normal mode.

    python -m cases.two_phase_imex_check [--backend cupy] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.basilisk_centered import CenteredNS
from basilisk_cupy.time.two_phase_imex import WALLS, TwoPhaseIMEX


def _sync(grid):
    if grid.xp.__name__ == "cupy":
        grid.xp.cuda.Stream.null.synchronize()


def drop(n: int, rho2: float, integrator: str, t_end: float = 0.5, backend: str = "numpy", mu: float = 0.1,
         transport: str = "projected", interface: str = "cls", **kw):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    phi = 0.25 - xp.sqrt(x * x + y * y)
    if integrator == "imex":
        ns = TwoPhaseIMEX(grid, phi, rho1=1.0, rho2=rho2, mu1=mu, mu2=mu, sigma=1.0, tolerance=1e-8,
                          transport=transport, interface=interface, **kw)
        tp = ns
    else:
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in WALLS}, tolerance=1e-8, viscous=CCDViscosity())
        cls = CLSInterface.from_level_set(grid, phi, BoundaryConditions(), divergence="fccd",
                                          ops=FCCDFaceComplex(order=6))
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=mu, mu2=mu, pressure_jump=True)
        tp.add_surface_tension(1.0)
        tp.pressure = lambda: tp.full_pressure(ns)
    ns.start()
    m0 = float(tp.cls.mass())
    umax_hist, t0 = 0.0, time.time()
    while ns.clock.t < t_end - 1e-12:
        ns.DT = t_end - ns.clock.t
        ns.step()
        u = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
        if not math.isfinite(u) or u > 1e3:
            break
        umax_hist = max(umax_hist, u)
    _sync(grid)
    wall = time.time() - t0
    p = to_numpy(tp.pressure())
    r = np.hypot(to_numpy(x), to_numpy(y))
    return {"jump": float(p[r < 0.1].mean() - p[r > 0.4].mean()), "u_end": u, "u_max": umax_hist,
            "mass": float(tp.cls.mass()) / m0 - 1.0, "steps": ns.clock.i, "t": ns.clock.t, "wall": wall}


def capwave(n: int, integrator: str, convection: str = "central", backend: str = "numpy", DT: float | None = None,
            transport: str = "projected"):
    """``DT``: largest step (the output interval caps it anyway)."""
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay
    from cases.capwave_modes import period_decay

    if integrator == "centered":
        t0 = time.time()
        r = cw.run(n, backend, interface="cls", pressure_jump=True, face_order=6, divergence="fccd",
                   viscous=CCDViscosity(), DT=DT)
        wall = time.time() - t0
        amps, steps = r["amplitudes"], r["steps"]
    else:
        grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        k = 2.0 * math.pi
        dist = (y - 0.01 * xp.cos(k * x)) / xp.sqrt(1.0 + (0.01 * k * xp.sin(k * x)) ** 2)
        ns = TwoPhaseIMEX(grid, dist, mu1=cw.MU, mu2=cw.MU, sigma=1.0, convection=convection, tolerance=1e-6,
                          transport=transport,
                          DT=DT if DT is not None else 1e30)
        times = cw.event_times()
        amps = [cw._cls_amplitude(xp, ns.psi, grid)]
        ns.start(events=times[1:])
        t0 = time.time()
        while len(amps) < len(times):
            ns.step()
            if abs(ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, ns.clock.t):
                amps.append(cw._cls_amplitude(xp, ns.psi, grid))
        _sync(grid)
        wall, steps = time.time() - t0, ns.clock.i
    P = np.array(cw.PROSPERETTI)
    a = np.array(amps)
    m = min(len(a), len(P))
    err = math.sqrt(float(np.mean((a[:m] - P[:m, 1]) ** 2))) / 0.01
    pr, d = decay(a, P[:len(a), 0])
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, cw.MU)
    return {"error": err, "damping": math.log(d) / math.log(dt_) - 1, "period": pr / pt - 1,
            "amplitudes": amps, "steps": steps, "wall": wall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--jobs", type=int, default=None, help="parallel processes (default: cores - 2)")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.capillary_wave import STEP
    from cases.parallel import pmap

    variants = (("centered", "projected"), ("imex", "projected"), ("imex", "ab2"))
    cases = [dict(n=n, rho2=rho2, integrator=integ, transport=tr, backend=args.backend)
             for n in (32, 64) for rho2 in (1.0, 0.1, 0.01, 0.001) for integ, tr in variants]
    print("static drop (t = 0.5): n rho2 integrator transport jump u_end u_max mass steps wall")
    for c, r in zip(cases, pmap(drop, cases, args.jobs)):
        print(f"{c['n']} {c['rho2']:g} {c['integrator']:8s} {c['transport']:9s} {r['jump']:.5f} {r['u_end']:.2e} {r['u_max']:.2e} "
              f"{r['mass']:.1e} {r['steps']} {r['wall']:.1f}s")

    cases = [dict(n=n, integrator=integ, convection=conv, transport=tr, backend=args.backend) for n in (16, 32, 64)
             for integ, conv, tr in (("centered", "central", "projected"), ("imex", "central", "projected"),
                                     ("imex", "fuccd", "projected"), ("imex", "uccd6", "projected"),
                                     ("imex", "central", "ab2"))]
    divs = (1, 2, 4, 8, 16, 32)
    dvar = (("centered", "projected"), ("imex", "projected"), ("imex", "ab2"))
    dcases = [dict(n=32, integrator=integ, transport=tr, backend=args.backend,
                   DT=STEP / d * (1 + 1e-7) if d > 1 else None) for d in divs for integ, tr in dvar]
    res = pmap(capwave, cases + dcases, args.jobs)
    print("capillary wave: n integrator convection transport error damping period steps wall")
    runs = {}
    for c, r in zip(cases, res[:len(cases)]):
        conv = c["convection"] if c["integrator"] == "imex" else "-"
        runs[(c["n"], c["integrator"], conv, c["transport"])] = r
        print(f"{c['n']} {c['integrator']:8s} {conv:7s} {c['transport']:9s} {r['error']:.4f} {100 * r['damping']:+.1f}% "
              f"{100 * r['period']:+.2f}% {r['steps']} {r['wall']:.1f}s")
    print("capillary wave, N = 32, step / d: d integrator transport error damping period steps wall")
    dres = {}
    for i, (c, r) in enumerate(zip(dcases, res[len(cases):])):
        d = divs[i // len(dvar)]
        dres[(d, c["integrator"], c["transport"])] = r
        print(f"{d} {c['integrator']:8s} {c['transport']:9s} {r['error']:.4f} {100 * r['damping']:+.2f}% {100 * r['period']:+.2f}% "
              f"{r['steps']} {r['wall']:.1f}s")

    from cases import capillary_wave as cw

    P = np.array(cw.PROSPERETTI)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, n in zip(axes, (16, 32)):
        ax.plot(P[:, 0], P[:, 1], "k-", lw=1.0, label="Prosperetti")
        for (nn, integ, conv, tr), r in runs.items():
            if nn != n or conv in ("fuccd", "uccd6"):
                continue
            a = np.array(r["amplitudes"])
            lab = ("CenteredNS + CLS/GFM" if integ == "centered" else
                   "TwoPhaseIMEX, ψ by u_f^n (paper)" if tr == "projected" else "TwoPhaseIMEX, ψ by AB2 u_f")
            ax.plot(P[:len(a), 0], a, "--" if integ == "centered" else "-" if tr == "projected" else ":", lw=1.0,
                    label=f"{lab}: RMS {r['error']:.3f}")
        ax.set_title(f"capillary wave, {n // 2} points per wavelength")
        ax.set_xlabel("t ω₀")
        ax.set_ylabel("amplitude")
        ax.legend(fontsize=8)
    ax = axes[2]
    for integ, tr, lab, m in (("centered", "projected", "CenteredNS + CLS/GFM", "s--"),
                              ("imex", "projected", "TwoPhaseIMEX, ψ by u_f^n (paper)", "o-"),
                              ("imex", "ab2", "TwoPhaseIMEX, ψ by AB2 u_f", "^:")):
        ax.semilogx([STEP / d for d in divs], [100 * dres[(d, integ, tr)]["damping"] for d in divs], m, label=lab)
    ax.set_xlabel("Δt")
    ax.set_ylabel("damping error per period [%]")
    ax.set_title("16 points per wavelength: time-step dependence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/two_phase_imex.png", dpi=150)
    print("wrote", f"{args.out}/two_phase_imex.png")


if __name__ == "__main__":
    main()

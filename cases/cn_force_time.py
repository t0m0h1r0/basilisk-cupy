"""Time level of the surface-tension jump in Crank--Nicolson-type schemes (docs/notes/51, section 3).

``IMEXMultistepNS`` puts the face acceleration (the GFM jump, with the pressure) at the implicit level t^{n+1}
for every scheme. That is consistent for BDF (gamma = (1)), but for the Adams--Moulton / theta family the
forcing belongs to the same weights as the viscous term: ``sum_j gamma_j a^{n+1-j}`` (CN: ``(a^{n+1} + a^n)/2``).
This prototype blends ``a`` with the scheme's gamma weights after the "acceleration" hooks, and runs the capillary
wave (16 points) at steps ``STEP / d``.

    python -m cases.cn_force_time [--jobs N]
"""

from __future__ import annotations

import argparse


def attach_weighted_force(ns) -> None:
    """After TwoPhaseCLS has written ``a^{n+1}``: ``a <- sum_j gamma_j a^{n+1-j}`` (raw values remembered)."""
    nd = ns.grid.ndim
    ns._a_raw = []

    def blend(_ns):
        raw = [ns.a[ax].copy() for ax in range(nd)]
        g = ns.last_coefficients.gamma
        if len(g) > 1 and len(ns._a_raw) >= len(g) - 1:
            for ax in range(nd):
                ns.a[ax] = g[0] * raw[ax] + sum(g[j] * ns._a_raw[j - 1][ax] for j in range(1, len(g)))
        ns._a_raw = ([raw] + ns._a_raw)[:3]

    ns.on("acceleration", blend, after=True)


def capwave(scheme: str, transport: str, d: int, weighted: bool):
    import math
    import warnings

    import numpy as np

    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX
    from cases import capillary_wave as cw
    from cases.capwave_damping import decay
    from cases.capwave_modes import period_decay

    warnings.filterwarnings("ignore")
    grid = CartesianGrid((32, 32), length=(2.0, 2.0), origin=(0.0, -1.0))
    x, y = grid.meshgrid()
    k = 2.0 * math.pi
    dist = (y - 0.01 * np.cos(k * x)) / np.sqrt(1.0 + (0.01 * k * np.sin(k * x)) ** 2)
    ns = TwoPhaseIMEX(grid, dist, mu1=cw.MU, mu2=cw.MU, sigma=1.0, tolerance=1e-6, scheme=scheme,
                      transport=transport, fixed_dt=cw.STEP / d)
    if weighted:
        attach_weighted_force(ns)
    times = cw.event_times()
    amps = [cw._cls_amplitude(np, ns.psi, grid)]
    ns.start(events=times[1:])
    while len(amps) < len(times):
        ns.step()
        if abs(ns.clock.t - times[len(amps)]) <= 1e-9 * max(1.0, ns.clock.t):
            amps.append(cw._cls_amplitude(np, ns.psi, grid))
    P = np.array(cw.PROSPERETTI)
    pr, dd = decay(np.array(amps), P[:len(amps), 0])
    pt, dt_ = period_decay(2 * math.pi, 1.0, 1.0, cw.MU)
    return {"damping": math.log(dd) / math.log(dt_) - 1, "period": float(pr / pt - 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(scheme=s, transport=t, d=d, weighted=w) for s in ("cnab2", "mcnab2", "sbdf2")
             for t in ("frozen", "ab2", "pc") for w in (False, True) for d in (1, 2, 4, 8)
             if not (s == "sbdf2" and w)]
    res = pmap(capwave, items, args.jobs)
    print("scheme transport force: damping at d = 1 / 2 / 4 / 8")
    for i in range(0, len(items), 4):
        it = items[i]
        print(f"{it['scheme']:7s} {it['transport']:6s} {'weighted' if it['weighted'] else 'n+1     '}: "
              + " / ".join(f"{100 * r['damping']:+.2f}%" for r in res[i:i + 4]))


if __name__ == "__main__":
    main()

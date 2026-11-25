"""Fewer outer iterations of ``CenteredNS(coupled_iterations=K)`` (docs/notes/36).

From the oscillating drop of :mod:`cases.coupled_residuals` (after ``warm`` converged steps), ``steps`` more
steps are taken with each variant, stopping the iterations when ``dt |g_{k+1} - g_k|`` falls below ``rtol``
times its first value (``coupled_rtol``) or 1e-13:

* plain fixed-point iteration (the previous step's pressure to start);
* ``coupled_predictor="linear"``: ``2 p^n - p^{n-1}`` to start;
* ``coupled_anderson=m``: Anderson acceleration with ``m`` previous residuals;
* both; and the same with ``TwoPhaseCLS(jump_mode="smooth")`` (the predictor then extrapolates the pressure
  without its jump).

Reported: mean iterations per step, wall time per step, and ``max |u - u_ref|`` after ``steps`` steps
against the plain iteration run to 1e-13.

    python -m cases.coupled_acceleration
"""

from __future__ import annotations

import copy
import time
import warnings

import numpy as np

from cases.coupled_residuals import drop

VARIANTS = {"plain": {}, "predictor": {"coupled_predictor": "linear"}, "anderson1": {"coupled_anderson": 1},
            "anderson2": {"coupled_anderson": 2}, "anderson3": {"coupled_anderson": 3},
            "predictor+anderson2": {"coupled_predictor": "linear", "coupled_anderson": 2}}


def run(ns, steps, rtol, **attrs):
    m = copy.deepcopy(ns, {id(ns.grid): ns.grid})
    m.coupled_iterations = 40
    m.coupled_tolerance = 1e-13
    m.coupled_rtol = rtol
    m.coupled_on_failure = "ignore"
    m.coupled_failures = 0
    for k, v in attrs.items():
        setattr(m, k, v)
    its, t = [], time.perf_counter()
    for _ in range(steps):
        m.step()
        its.append(m.coupled_info["iterations"])
    return m, float(np.mean(its)), (time.perf_counter() - t) / steps, m.coupled_failures


def compare(n=32, rho2=0.01, factor=1.6, steps=20, rtols=(1e-2, 1e-3, 1e-4), jump_mode="sharp", warm=40):
    ns = drop(n, rho2, factor=factor, warm=warm, jump_mode=jump_mode)
    ref, *_ = run(ns, steps, None)
    uref = [c.interior.copy() for c in ref.u]
    umax = max(float(np.abs(c).max()) for c in uref)
    rows = []
    for rtol in rtols:
        for name, attrs in VARIANTS.items():
            m, it, sec, fails = run(ns, steps, rtol, **attrs)
            err = max(float(np.abs(a.interior - b).max()) for a, b in zip(m.u, uref)) / umax
            rows.append({"rtol": rtol, "variant": name, "iterations": it, "seconds": sec, "error": err,
                         "failures": fails})
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    for n in (32, 64):
        for jump_mode in ("sharp", "smooth"):
            for r in compare(n, jump_mode=jump_mode):
                print(f"N={n} {jump_mode} rtol {r['rtol']:.0e} {r['variant']:20s} it/step {r['iterations']:5.2f} "
                      f"s/step {r['seconds']:.3f} rel.err {r['error']:.2e} failures {r['failures']}", flush=True)


if __name__ == "__main__":
    main()

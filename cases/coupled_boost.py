"""docs/notes/102 (V6-4): advection and viscosity together on one Gram matrix, the Galilean-boosted two-layer Stokes
flow (the user's test). u = (w(y - U t) e^{-lambda t}, U) with rho, mu advected at U is an exact incompressible
two-phase Navier-Stokes solution (periodic, no surface tension). It does not depend on x, so on the tensor-product
discretisation it reduces exactly to the 1-D problem in y, solved here:

    half viscous step (interface frozen, G_A^n q' = -K_nu^n q, integrated exactly),
    PCCD-Galerkin advection (G_A^n -> G_A^{n+1}, the common partition),
    half viscous step (G_A^{n+1}, K_nu^{n+1}),

with the partial-cell phase-split PCCD (mode "blend_partial") for G_A and the minimum-viscous-energy stress for K_nu.
At CFL p/q the q step maps repeat with a shift of p cells (cases.pccd_floquet.run_periodic).

    python -m cases.coupled_boost
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd_floquet import step
from cases.pccd_jump import A_, B_
from cases.pccd_twophase import averages
from cases.viscous_minenergy import MinEnergy, _gen_eig, eigenfunction
from cases.viscous_stress import Setup, mus, stokes_rate


def _visc(G, K, dt):
    ev, V = _gen_eig(G, K)
    return V @ np.diag(np.exp(-ev * dt)) @ V.T @ G


def run(N, ratio, kind, scale=0.01, U=1.0, p=1, q=4, t_end=0.5, samples=(0.25,), advect=True):
    mu_H, mu_L = mus(ratio, kind)
    mu_H, mu_L = mu_H * scale, mu_L * scale
    lam = stokes_rate(mu_H, mu_L, ratio, A_, B_, which=1)[0]
    w = eigenfunction(mu_H, mu_L, ratio, A_, B_, lam)
    h = 1.0 / N
    D = p * h / q if advect else 0.0
    dt = (p * h / q) / U
    steps = int(round(t_end / dt))
    maps, eps_G, phi = [], 0.0, None
    G_prev = None
    for k in range(q):
        a0, b0 = (A_ + k * D) % 1.0, (B_ + k * D) % 1.0
        a1, b1 = (A_ + (k + 1) * D) % 1.0, (B_ + (k + 1) * D) % 1.0
        T, phi, G0, G1, _ = step(N, ratio, a0, b0, D if advect else 1e-14, "blend_partial", "galerkin", phi)
        if not advect:
            T = np.eye(N)
        K0 = MinEnergy(Setup(N, ratio, a0, b0, mu_H, mu_L)).K
        K1 = MinEnergy(Setup(N, ratio, a1, b1, mu_H, mu_L)).K
        if G_prev is not None:
            eps_G = max(eps_G, float(np.abs(G0 - G_prev).max() / np.abs(G0).max()))
        G_prev = G1
        maps.append((_visc(G0, K0, dt / 2), T, _visc(G1, K1, dt / 2), G0, G1))
    q0 = averages(N, A_, B_, ratio, lambda x: w(x % 1.0))
    u = q0.copy()
    errs, E, mom = {}, [], []
    at = {int(round(t / dt)): t for t in samples}
    pp = p if advect else 0
    for n in range(steps):
        j, k = divmod(n, q)
        sh = j * pp
        Vpre, T, Vpost, G0, G1 = maps[k]
        x = np.roll(u, -sh)
        E.append(float(x @ G0 @ x))
        mom.append(float(np.ones(N) @ G0 @ x))
        if n == 0:
            mscale = float((G0 @ np.ones(N)) @ np.abs(x))      # sum m |q0| (the mode's total momentum is ~ 0)
        x = Vpost @ (T @ (Vpre @ x))
        u = np.roll(x, sh)
        if n + 1 in at:
            t = (n + 1) * dt
            ex = averages(N, (A_ + U * t * advect) % 1.0, (B_ + U * t * advect) % 1.0, ratio,
                          lambda y: w((y - U * t * advect) % 1.0) * math.exp(-lam * t))
            errs[at[n + 1]] = float(np.abs(u - ex).max() / np.abs(q0).max())
    t = steps * dt
    ex = averages(N, (A_ + U * t * advect) % 1.0, (B_ + U * t * advect) % 1.0, ratio,
                  lambda y: w((y - U * t * advect) % 1.0) * math.exp(-lam * t))
    errs[t_end] = float(np.abs(u - ex).max() / np.abs(q0).max())
    dE = np.diff(E)
    return {"lambda": lam, "errors": errs, "eps_G": eps_G, "energy_max_increase": float(max(dE.max(), 0.0) / E[0]),
            "momentum_drift": float(abs(mom[-1] - mom[0]) / mscale), "steps": steps}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for kind in ("mu", "water-air", "nu"):
        for N in (32, 64, 128):
            jobs.append(("space", dict(N=N, ratio=1e3, kind=kind)))
            jobs.append(("U=0", dict(N=N, ratio=1e3, kind=kind, advect=False)))
    for kind in ("mu", "water-air"):
        for pq in ((1, 2), (1, 4), (1, 8), (1, 16)):
            jobs.append(("time", dict(N=64, ratio=1e3, kind=kind, p=pq[0], q=pq[1])))
    res = []
    with ProcessPoolExecutor(26) as ex:
        futs = [(n, kw, ex.submit(run, **kw)) for n, kw in jobs]
        for n, kw, f in futs:
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            res.append({"test": n, **kw, **r})
            print(n, kw, r, flush=True)
    json.dump(res, open("docs/figures/coupled_boost.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

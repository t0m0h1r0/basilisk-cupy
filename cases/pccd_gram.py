"""docs/notes/95 (P0): why the Gram matrix of the two-phase PCCD reconstructions is ill-conditioned.

For a reconstruction u_h = sum_j phi_j ubar_j (linear in the mass-weighted averages), G_ij = int rho phi_i phi_j and
M = diag(m). If a light pure cell j has a basis function reaching into the heavy phase,

    G_jj / m_j >= (rho_H / (rho_L h)) int_{Omega_H} |phi_j|^2      (the user's lower bound)

so lambda_max(M^{-1/2} G M^{-1/2}) grows like r unless the leak (1/h) int_H |phi_j|^2 is O(1/r).

Reconstructions (``mode``): "none" (note 92: PCCD with the mixed-cell recovery, no jump separation), "onesided",
"joint", "joint_light" (note 94's estimates; the cut-off jump term on its default side or on the light side), and the
docs/notes/95 constructions "local" (P1-a) and "split" (P1-b) from cases.pccd_local.

    python -m cases.pccd_gram
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG
from cases.pccd_density import pieces
from cases.pccd_jump import A_, B_, reconstruct
from cases.pccd_twophase import evaluate


def basis(N, ratio, a, b, mode):
    """A function phi(j, x) -> values of the j-th basis function at x, for the given mode."""
    if mode in ("local", "split", "blend"):
        from cases.pccd_local import basis_functions
        return basis_functions(N, ratio, a, b, mode)
    out = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        if mode == "none":
            r = reconstruct(N, ratio, False, a=a, b=b, ubar_in=e)
            out.append((r["coef"], []))
        else:
            r = reconstruct(N, ratio, True, a=a, b=b, estimate=True, ubar_in=e,
                            method="onesided" if mode == "onesided" else "joint", light=(mode == "joint_light"))
            out.append((r["coef"], r["terms"]))
    return lambda j, x: evaluate(N, out[j][0], out[j][1], x)


def quadrature(N, ratio, a, b, sub=4):
    """Points, rho-weights and the heavy flag on every cell's phase pieces (each piece cut into ``sub`` parts)."""
    h = 1.0 / N
    X, W, H, cell = [], [], [], []
    for i, ps in enumerate(pieces(N, a, b)):
        for s0, s1, hv in ps:
            for k in range(sub):
                p0, p1 = s0 + (s1 - s0) * k / sub, s0 + (s1 - s0) * (k + 1) / sub
                X.append((i + p0 + (p1 - p0) * XG) * h)
                W.append((p1 - p0) * h * WG * (1.0 if hv else 1.0 / ratio))
                H.append(np.full(XG.size, hv))
                cell.append(np.full(XG.size, i))
    return np.concatenate(X), np.concatenate(W), np.concatenate(H), np.concatenate(cell)


def stats(N, ratio, mode, a=A_, b=B_):
    h = 1.0 / N
    phi = basis(N, ratio, a, b, mode)
    X, W, Hv, cell = quadrature(N, ratio, a, b)
    B = np.stack([phi(j, X) for j in range(N)], axis=1)
    G = B.T @ (W[:, None] * B)
    m = np.bincount(cell, weights=W, minlength=N)
    s = 1.0 / np.sqrt(m)
    ev, V = np.linalg.eigh(G * s[:, None] * s[None, :])
    # the leak of the light pure cells' basis into the heavy phase: (1/h) int_H |phi_j|^2 (unweighted)
    ps = pieces(N, a, b)
    light = [j for j in range(N) if len(ps[j]) == 1 and not ps[j][0][2]]
    wH = np.where(Hv, W, 0.0)                       # rho_H = 1
    leak = np.array([(wH @ B[:, j] ** 2) / h for j in light])
    bound = (ratio / h) * leak.max() * h if light else 0.0   # r (1/h) int_H |phi|^2 ... in units of m_j = rho_L h
    top = np.abs(V[:, -1])
    return {"lmin": float(ev[0]), "lmax": float(ev[-1]), "leak_max": float(leak.max()),
            "leak_cell": int(light[int(np.argmax(leak))]), "bound_r_leak": float(ratio * leak.max()),
            "top_mode_cell": int(np.argmax(top)), "top_mode_light_share": float(np.sum(top[light] ** 2))}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    modes = ("none", "onesided", "joint", "joint_light", "local", "split")
    jobs = [(N, r, m) for N in (32, 64) for r in (1.0, 1e3, 1e6) for m in modes]
    res = []
    with ProcessPoolExecutor(18) as ex:
        futs = [(j, ex.submit(stats, *j)) for j in jobs]
        for (N, r, m), f in futs:
            try:
                out = f.result()
            except Exception as e:  # noqa: BLE001
                out = {"error": repr(e)}
            res.append({"N": N, "ratio": r, "mode": m, **out})
            print(N, r, m, out, flush=True)
    json.dump(res, open("docs/figures/pccd_gram.json", "w"), indent=1)


if __name__ == "__main__":
    main()


def stiffness(N, ratio, c, mode="blend_partial", eps=1e-7):
    """lambda_max(M^{-1/2} K M^{-1/2}) h^2 with K_ij = int rho phi_i' phi_j' (constant kinematic viscosity; the
    derivatives by central differences inside the pieces), the interface a in cell N/4 with heavy fraction c
    (docs/notes/97: the user's impossibility argument predicts growth ~ r at c ~ r^{-1/2})."""
    from cases.pccd_floquet import _basis

    h = 1.0 / N
    a = (N // 4 + 1 - c) * h
    phi = _basis(N, ratio, a, B_, mode)
    X, W, _, cell = quadrature(N, ratio, a, B_)
    dB = np.stack([(phi(j, X + eps * h) - phi(j, X - eps * h)) / (2 * eps * h) for j in range(N)], axis=1)
    K = dB.T @ (W[:, None] * dB)
    m = np.bincount(cell, weights=W, minlength=N)
    s = 1.0 / np.sqrt(m)
    return float(np.linalg.eigvalsh(K * s[:, None] * s[None, :])[-1] * h * h)


def _objects(N, ratio, a, mode):
    """Reconstruction objects with an analytic derivative() for the unit averages."""
    import json as _json

    from cases.pccd_closure import closure
    from cases.pccd_local import Split, SplitBlend

    out = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        if mode == "blend_partial":
            out.append(SplitBlend(N, ratio, a, B_, e, mixed_mode="partial"))
        elif mode == "split_partial":
            out.append(Split(N, ratio, a, B_, e, mixed_mode="partial"))
        elif mode == "blend_opt":
            cl = closure(np.array(_json.load(open("docs/figures/pccd_closure.json"))["best"]))
            out.append(SplitBlend(N, ratio, a, B_, e, closure=cl))
        else:
            raise ValueError(mode)
    return out


def stiffness_analytic(N, ratio, c, mode="blend_partial"):
    """As stiffness(), with the reconstructions' analytic derivatives (docs/notes/97 correction: the central
    differences of stiffness() can straddle a piece boundary of a tiny cut cell). Also returns the user's lower bound
    for the partial-cell form with the mixed-cell datum alone, (24/5)(r/c + 1/(1-c))/(rc + 1 - c)."""
    h = 1.0 / N
    a = (N // 4 + 1 - c) * h
    objs = _objects(N, ratio, a, mode)
    X, W, _, cell = quadrature(N, ratio, a, B_)
    dB = np.stack([o.derivative(X) for o in objs], axis=1)
    K = dB.T @ (W[:, None] * dB)
    m = np.bincount(cell, weights=W, minlength=N)
    s = 1.0 / np.sqrt(m)
    lam = float(np.linalg.eigvalsh(K * s[:, None] * s[None, :])[-1] * h * h)
    r = ratio
    bound = 24 / 5 * (r / c + 1 / (1 - c)) / (r * c + 1 - c)
    return {"lambda": lam, "bound_mixed_datum": float(bound)}

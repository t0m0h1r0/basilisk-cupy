"""docs/notes/95 (P1-c): compact implicit end closures for the phase-split PCCD, optimised under quartic exactness.

At the end face 0 of a phase run (cells 0..5 of that phase, face 1 the next face):

    v0 + p1 v1 + p2 g1 = sum_{k=0}^{5} c_k q_k,      g0 + p3 v1 + p4 g1 = sum_{k=0}^{5} d_k q_k

Exactness for u = s^p, p = 0..4 (cell averages ((k+1)^{p+1} - k^{p+1})/(p+1), v_f = f^p, g_f = p f^{p-1}) fixes c given
(p1, p2) up to one null direction t1 of the 5 x 6 moment matrix; likewise d given (p3, p4) up to t2: six parameters.
The objective is the user's: the reconstruction's amplification Lambda (max over the cells of sum_j |phi_j|) and
lambda_max(M^{-1/2} G M^{-1/2}), over interface positions with phase-mass ratios from 1e-4 to 1e4 (density ratio 1e6),
with a penalty when a run's CCD matrix is ill-conditioned. Nelder-Mead (numpy only).

    python -m cases.pccd_closure
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd_gram import quadrature
from cases.pccd_jump import B_
from cases.pccd_local import Split, explicit_closure

K = 6
Q = np.array([[((k + 1) ** (p + 1) - k ** (p + 1)) / (p + 1) for k in range(K)] for p in range(5)])
_, _, _Vt = np.linalg.svd(Q)
NULL = _Vt[-1]
QP = np.linalg.pinv(Q)


def closure(theta):
    p1, p2, t1, p3, p4, t2 = theta
    pw = np.arange(5)
    v1, g1 = np.ones(5), pw.astype(float)                  # at face 1: f^p and p f^(p-1)
    v0, g0 = (pw == 0).astype(float), (pw == 1).astype(float)
    c = QP @ (v0 + p1 * v1 + p2 * g1) + t1 * NULL
    d = QP @ (g0 + p3 * v1 + p4 * g1) + t2 * NULL
    return (np.array([1.0, 0.0, p1, p2]), c, np.array([0.0, 1.0, p3, p4]), d)


def evaluate(cl, N=32, ratio=1e6, zetas=(1e-4, 1e-2, 1.0, 1e2, 1e4), pts=21):
    """max Lambda and max lambda_max over the interface positions (the cut cell N/4 with phase-mass ratio zeta)."""
    h = 1.0 / N
    lam_max = amp = 0.0
    s1 = np.linspace(0.0005, 0.9995, pts)
    for z in zetas:
        c = z / (ratio + z)
        a = (N // 4 + 1 - c) * h
        fs = []
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.0
            fs.append(Split(N, ratio, a, B_, e, closure=cl))
        X = ((np.arange(N)[:, None] + s1[None, :]) * h).ravel()
        V = np.stack([f(X) for f in fs], axis=1)
        amp = max(amp, float(np.abs(V).sum(1).max()))
        Xq, W, _, cell = quadrature(N, ratio, a, B_)
        B = np.stack([f(Xq) for f in fs], axis=1)
        G = B.T @ (W[:, None] * B)
        m = np.bincount(cell, weights=W, minlength=N)
        sc = 1.0 / np.sqrt(m)
        lam_max = max(lam_max, float(np.linalg.eigvalsh(G * sc[:, None] * sc[None, :])[-1]))
    return amp, lam_max


def run_cond(cl, n=12):
    """Condition number of one run's CCD matrix with the closure at both ends."""
    from cases.pccd_local import hb  # noqa: F401  (import check)
    rv, cv, rg, cg = cl
    nf = n + 1
    A = np.zeros((2 * nf, 2 * nf))
    A[0, 0:4], A[1, 0:4] = rv, rg
    mir = np.array([1.0, -1.0, 1.0, -1.0])
    idx = [2 * n, 2 * n + 1, 2 * (n - 1), 2 * (n - 1) + 1]
    A[2 * n, idx], A[2 * n + 1, idx] = rv * mir, -rg * mir
    for f in range(1, n):
        r = 2 * f
        A[r, 2 * f], A[r, 2 * (f - 1)], A[r, 2 * (f + 1)] = 1.0, 7 / 16, 7 / 16
        A[r, 2 * (f + 1) + 1], A[r, 2 * (f - 1) + 1] = -1 / 16, 1 / 16
        A[r + 1, 2 * f + 1], A[r + 1, 2 * (f - 1) + 1], A[r + 1, 2 * (f + 1) + 1] = 1.0, -1 / 8, -1 / 8
        A[r + 1, 2 * (f + 1)], A[r + 1, 2 * (f - 1)] = 9 / 8, -9 / 8
    return float(np.linalg.cond(A))


def objective(theta, beta=1.0):
    cl = closure(theta)
    cond = run_cond(cl)
    if not np.isfinite(cond) or cond > 1e4:
        return 1e6
    try:
        amp, lam = evaluate(cl, zetas=(1e-2, 1e2), pts=11)
    except np.linalg.LinAlgError:
        return 1e6
    return lam + beta * amp + (0.0 if cond < 200 else (cond - 200) / 10)


def nelder_mead(f, x0, step=0.3, iters=300, tol=1e-6):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def explicit_theta():
    """theta of the explicit 5-cell closure (p's zero, c on the null direction such that c_5 = 0)."""
    t1 = -(QP @ np.eye(5)[0])[5] / NULL[5]
    t2 = -(QP @ np.eye(5)[1])[5] / NULL[5]
    return np.array([0.0, 0.0, t1, 0.0, 0.0, t2])


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    th0 = explicit_theta()
    starts = [th0, th0 + [0.3, 0.0, 0, 0.0, 0.3, 0], th0 + [0.5, -0.1, 0, 0.5, 0.0, 0],
              th0 + [0.2, 0.05, 0.1, -0.5, 0.2, 0.1], th0 + [0.45, -0.06, 0, -1.0, 0.1, 0],
              th0 + [0.1, 0.0, -0.2, 0.0, -0.1, 0.2]]
    with ProcessPoolExecutor(len(starts)) as ex:
        res = list(ex.map(nelder_mead, [objective] * len(starts), starts))
    best = min(res, key=lambda r: r[1])
    out = {"explicit": {"theta": th0.tolist(), "objective": objective(th0)},
           "runs": [{"theta": r[0].tolist(), "objective": r[1]} for r in res], "best": best[0].tolist()}
    cl = closure(best[0])
    out["best_closure"] = [x.tolist() for x in cl]
    amp, lam = evaluate(cl)
    amp0, lam0 = evaluate(explicit_closure())
    out["best_full"] = {"Lambda": amp, "lambda_max": lam, "cond_run": run_cond(cl)}
    out["explicit_full"] = {"Lambda": amp0, "lambda_max": lam0, "cond_run": run_cond(explicit_closure())}
    print(json.dumps(out, indent=1))
    json.dump(out, open("docs/figures/pccd_closure.json", "w"), indent=1)


if __name__ == "__main__":
    main()


def evaluate_blend(cl, N=32, ratio=1e6, zetas=(1e-4, 1e-2, 1.0, 1e2, 1e4), pts=21, mu0=0.1):
    """evaluate() for the continuous phase split (cases.pccd_local.SplitBlend)."""
    from cases.pccd_local import SplitBlend

    h = 1.0 / N
    out = []
    s1 = np.linspace(0.0005, 0.9995, pts)
    for z in zetas:
        c = z / (ratio + z)
        a = (N // 4 + 1 - c) * h
        fs = []
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.0
            fs.append(SplitBlend(N, ratio, a, B_, e, closure=cl, mu0=mu0))
        X = ((np.arange(N)[:, None] + s1[None, :]) * h).ravel()
        V = np.stack([f(X) for f in fs], axis=1)
        Xq, W, _, cell = quadrature(N, ratio, a, B_)
        B = np.stack([f(Xq) for f in fs], axis=1)
        G = B.T @ (W[:, None] * B)
        m = np.bincount(cell, weights=W, minlength=N)
        sc = 1.0 / np.sqrt(m)
        out.append((z, float(np.abs(V).sum(1).max()), float(np.linalg.eigvalsh(G * sc[:, None] * sc[None, :])[-1])))
    return out

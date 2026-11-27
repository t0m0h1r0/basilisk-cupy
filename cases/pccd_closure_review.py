"""docs/notes/97 (P0): the six optimised end closures of docs/notes/95 re-evaluated on every condition.

For each candidate theta (cases.pccd_closure, docs/figures/pccd_closure.json "runs") and the explicit 5-cell closure:
* the run matrix A (n = 12 cells, closure at both ends): kappa(A), kappa after row normalisation, and the response
  ||A^{-1} B||_inf of the face values and gradients to the cell averages;
* the continuous phase split (SplitBlend) at N = 32 over zeta = 1e-6 .. 1e6 and density ratios 1, 1e3, 1e6:
  max lambda_max(M^{-1/2} G M^{-1/2}) and max Lambda;
* the accuracy (note 93's profile, density ratio 1e6, N = 32 .. 256).

    python -m cases.pccd_closure_review
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd_closure import closure, evaluate_blend, explicit_theta
from cases.pccd_jump import A_, B_, exact
from cases.pccd_local import SplitBlend
from cases.pccd_twophase import averages

ZETAS = tuple(10.0 ** k for k in range(-6, 7))


def run_matrices(cl, n=12):
    rv, cv, rg, cg = cl
    k = cv.size
    nf = n + 1
    A = np.zeros((2 * nf, 2 * nf))
    B = np.zeros((2 * nf, n))
    A[0, 0:4], B[0, :k] = rv, cv
    A[1, 0:4], B[1, :k] = rg, cg
    mir = np.array([1.0, -1.0, 1.0, -1.0])
    idx = [2 * n, 2 * n + 1, 2 * (n - 1), 2 * (n - 1) + 1]
    A[2 * n, idx], B[2 * n, n - k:] = rv * mir, cv[::-1]
    A[2 * n + 1, idx], B[2 * n + 1, n - k:] = -rg * mir, -cg[::-1]
    for f in range(1, n):
        r = 2 * f
        A[r, 2 * f], A[r, 2 * (f - 1)], A[r, 2 * (f + 1)] = 1.0, 7 / 16, 7 / 16
        A[r, 2 * (f + 1) + 1], A[r, 2 * (f - 1) + 1] = -1 / 16, 1 / 16
        B[r, f - 1], B[r, f] = 15 / 16, 15 / 16
        A[r + 1, 2 * f + 1], A[r + 1, 2 * (f - 1) + 1], A[r + 1, 2 * (f + 1) + 1] = 1.0, -1 / 8, -1 / 8
        A[r + 1, 2 * (f + 1)], A[r + 1, 2 * (f - 1)] = 9 / 8, -9 / 8
        B[r + 1, f], B[r + 1, f - 1] = 3.0, -3.0
    return A, B


def matrix_stats(cl):
    A, B = run_matrices(cl)
    D = 1.0 / np.abs(A).max(axis=1)
    return {"cond": float(np.linalg.cond(A)), "cond_rownorm": float(np.linalg.cond(D[:, None] * A)),
            "response_inf": float(np.abs(np.linalg.solve(A, B)).sum(1).max())}


def accuracy(cl, ratio=1e6):
    out = []
    for N in (32, 64, 128, 256):
        ub = averages(N, A_, B_, ratio, exact)
        R = SplitBlend(N, ratio, A_, B_, ub, closure=cl)
        x = (np.arange(N * 41) + 0.5) / (N * 41)
        out.append(float(np.abs(R(x) - exact(x)).max()))
    return out


def review(name, theta):
    cl = closure(np.asarray(theta))
    res = {"name": name, "theta": list(map(float, theta)), **matrix_stats(cl)}
    for r in (1.0, 1e3, 1e6):
        rows = evaluate_blend(cl, ratio=r, zetas=ZETAS)
        res[f"Lambda_{r:g}"] = max(x[1] for x in rows)
        res[f"lmax_{r:g}"] = max(x[2] for x in rows)
        res[f"by_zeta_{r:g}"] = rows
    res["accuracy_1e6"] = accuracy(cl)
    return res


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    d = json.load(open("docs/figures/pccd_closure.json"))
    cands = [("explicit", explicit_theta())] + [(f"run{k} (obj {r['objective']:.3f})", r["theta"])
                                                for k, r in enumerate(d["runs"])]
    with ProcessPoolExecutor(len(cands)) as ex:
        res = list(ex.map(review, *zip(*cands)))
    for r in res:
        print(f"{r['name']:22s} cond {r['cond']:7.1f} rownorm {r['cond_rownorm']:6.1f} |A^-1 B| {r['response_inf']:6.1f}"
              f"  lmax {r['lmax_1']:.1f}/{r['lmax_1000']:.1f}/{r['lmax_1e+06']:.1f}"
              f"  Lambda {r['Lambda_1']:.1f}/{r['Lambda_1000']:.1f}/{r['Lambda_1e+06']:.1f}"
              f"  err256 {r['accuracy_1e6'][-1]:.2e} order {math.log2(r['accuracy_1e6'][1] / r['accuracy_1e6'][3]) / 2:.2f}",
              flush=True)
    json.dump(res, open("docs/figures/pccd_closure_review.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

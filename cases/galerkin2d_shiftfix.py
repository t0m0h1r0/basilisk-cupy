"""docs/notes/90: the correction of the velocity-difference shift (the user's review of PR #102).

* ``equivalence``: the corrected split fit against the plain shift (unconstrained weighted lstsq with the constant
  column, then the constant restores the cell mean, note 85), and the form of notes 88-89 against the least squares
  constrained to the cell mean.
* ``amplification``: Lambda_i = max_{x in K_i} sum_j |phi_ij(x)| in the cells next to an isolated interface, with the
  heavy phase a thin slice of the cut cell whose phase-mass ratio zeta = r c / (1 - c) is prescribed (c the heavy
  volume fraction): an edge-parallel slice (0 deg, c = delta / h) or a corner triangle (45 deg, c = (delta/h)^2 /
  (2 |nx ny|)).

    python -m cases.galerkin2d_shiftfix            # docs/figures/galerkin2d_shiftfix.json
    python -m cases.galerkin2d_shiftfix --rerun    # note 88 sections 3-5 again: galerkin2d_rank_shift.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from cases.galerkin2d import Geometry, Reconstruction2D

RK = {"window": "interface", "switch": "angle", "diff": True, "R": 5.5}


def equivalence(N=24, a=0, b=1, d=0.0123, ratio=1e6):
    """Largest coefficient difference (relative to the map's size) of the split fit against the plain shift and
    against the constrained least squares, over all piecewise fits of a grid."""
    out = {"shift_vs_plain": 0.0, "constrained_vs_lsq": 0.0, "fits": 0}
    for lsq in ("shift", "constrained"):
        rec = None
        orig = Reconstruction2D._fit_split
        log = []

        def spy(self, A, sw, k, cells, frame, pw, s):
            r = orig(self, A, sw, k, cells, frame, pw, s)
            log.append((A, sw, k, r[1], pw))
            return r
        Reconstruction2D._fit_split = spy
        try:
            rec = Reconstruction2D(N, Geometry(a, b, ratio=ratio, d=d), lsq=lsq, **RK)
        finally:
            Reconstruction2D._fit_split = orig
        worst = 0.0
        for A, sw, k, Wc, pw in log:
            if not pw:
                continue
            n = len(sw)
            E = np.eye(n)
            Bw = sw[:, None] * A
            if lsq == "shift":
                Ast = np.linalg.lstsq(Bw, np.diag(sw), rcond=None)[0]
                ref = Ast + np.outer(np.eye(A.shape[1])[0], E[k] - A[k] @ Ast)
            else:
                # min ||W^{1/2}(A a - u)|| subject to A_k a = u_k: the KKT system
                K = np.block([[Bw.T @ Bw, A[k][:, None]], [A[k][None, :], np.zeros((1, 1))]])
                rhs = np.vstack([Bw.T @ np.diag(sw), E[k][None, :]])
                ref = np.linalg.solve(K, rhs)[:-1]
            worst = max(worst, float(np.abs(Wc - ref).max() / np.abs(ref).max()))
        out["shift_vs_plain" if lsq == "shift" else "constrained_vs_lsq"] = worst
        out["fits"] = sum(1 for e in log if e[4])
    return out


def _slice_geometry(N, angle, c, ratio):
    """A band whose lower interface cuts cell (8, 8) leaving the heavy volume fraction c in it."""
    h = 1.0 / N
    if angle == 0:
        zA = (9 - c) * h                     # heavy above y = zA: a slice of height c h at the top of row 8
        return Geometry(0, 1, zA, zA + 0.5, ratio), 8 * N + 8
    t = np.sqrt(2 * c)                       # leg of the corner triangle (in h): area t^2 / 2 = c
    zA = (18 - t) * h                        # heavy above x + y = zA: the triangle at the top-right corner of (8, 8)
    return Geometry(1, 1, zA, zA + 0.5, ratio), 8 * N + 8


def amplification(N, angle, zeta, ratio, lsq, workers=1):
    c = zeta / (ratio + zeta)
    geom, i0 = _slice_geometry(N, angle, c, ratio)
    rec = Reconstruction2D(N, geom, lsq=lsq, workers=workers, **RK)
    h = 1.0 / N
    s1 = np.linspace(0.0, 1.0, 11)
    worst, where = 0.0, None
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ix, iy = 8 + dx, 8 + dy
            i = iy * N + ix
            P = np.stack(np.meshgrid((ix + s1) * h, (iy + s1) * h), -1).reshape(-1, 2)
            P = np.vstack([P, rec.qx[i]])
            _, V = rec.basis(i, P)
            lam = float(np.abs(V).sum(1).max())
            if lam > worst:
                worst, where = lam, (dx, dy)
    mass = rec.mass[i0] / (h * h)
    return {"Lambda": worst, "at": where, "c": c, "zeta_check": ratio * c / (1 - c), "cell_mass": mass}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    zetas = [0.01, 0.1, 1, 3.16, 10, 31.6, 100, 316, 1000, 1e4]
    jobs = [("equivalence", equivalence, {})]
    for ratio in (1e6, 1e3):
        for angle in (0, 45):
            for z in zetas:
                for lsq in ("shift", "constrained"):
                    jobs.append(("amp", amplification, {"N": 24, "angle": angle, "zeta": z, "ratio": ratio,
                                                        "lsq": lsq}))
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(30) as ex:
        futs = [(n, kw, ex.submit(fn, **kw)) for n, fn, kw in jobs]
        for n, kw, f in futs:
            r = f.result()
            res.append({"test": n, "args": kw, "result": r})
            print(f"{time.time() - t0:6.0f}s", n, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin2d_shiftfix.json", "w"), indent=1, default=str)




def rerun() -> None:
    """Notes 88's sections 3-5 again with the corrected shift (the default lsq="shift")."""
    from concurrent.futures import ProcessPoolExecutor

    from cases.galerkin2d import free_stream, one_step, recon_jump, run

    angles = {"0deg": (0, 1), "26.6deg": (1, 2), "45deg": (1, 1)}
    jobs = []
    for tr in ("geo", "galerkin", "corr"):
        for an in ("0deg", "45deg"):
            a, b = angles[an]
            jobs.append(("run", run, {"N": 48, "transport": tr, "a": a, "b": b, "samples": (0.05,), "rk": RK}))
    for an, (a, b) in angles.items():
        for N in (64, 48, 32, 24, 16):
            jobs.append(("one_step", one_step, {"N": N, "a": a, "b": b, "rk": RK}))
        jobs.append(("free_stream", free_stream, {"N": 24, "a": a, "b": b, "rk": RK}))
    for (a, b, N) in ((0, 1, 24), (1, 1, 24), (1, 1, 32)):
        for ratio in (1.0, 1e3):
            for sp in (4e-3, 2e-3):
                jobs.append(("recon_jump", recon_jump, {"N": N, "a": a, "b": b, "ratio": ratio, "spacing": sp,
                                                        "rk": RK}))
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(20) as ex:
        futs = [(n, kw, ex.submit(fn, **kw)) for n, fn, kw in jobs]
        for n, kw, f in futs:
            r = f.result()
            kw = {k: v for k, v in kw.items() if k != "rk"}
            res.append({"test": n, "args": kw, "result": r})
            print(f"{time.time() - t0:6.0f}s", n, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin2d_rank_shift.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys

    rerun() if "--rerun" in sys.argv else main()

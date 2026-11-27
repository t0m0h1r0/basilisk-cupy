"""監修（C）：ノート114 の「安定な 2 つの形」（加速度の形・引き戻さない形）を壊しにいく。
A の介入（cases.r0p1_source_feedback.solver）をそのまま使い、A が試していない条件：長い時間（2000 ステップ）、
気泡（周りが重い相）、薄い膜、斜めの帯、小さい液滴、2 つの液滴、有限の振幅の擾乱。対照として源の項なし（none）。"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

def geometry(name):
    def f0(grid):
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.fractions import fraction
        d = lambda r, cx, cy: (lambda x, y: r * r - (x - cx) ** 2 - (y - cy) ** 2)
        fr = lambda phi: to_numpy(fraction(grid, phi, refine=8))
        if name == "drop":   return fr(d(0.15, 0.5, 0.5))
        if name == "bubble": return 1.0 - fr(d(0.15, 0.5, 0.5))
        if name == "small":  return fr(d(0.05, 0.47, 0.52))
        if name == "film":   return fr(lambda x, y: 0.02 - np.abs(x - 0.513))
        if name == "slant":  return fr(lambda x, y: 0.13 - np.abs(((x + y) % 1.0) - 0.5))
        if name == "two":    return np.clip(fr(d(0.12, 0.3, 0.4)) + fr(d(0.1, 0.72, 0.65)), 0, 1)
        raise ValueError(name)
    return f0

def run(a):
    geo, mode, ratio, n, steps, pert = a
    from basilisk_cupy.backend import to_numpy
    from cases.r0p1_source_feedback import solver, U0
    grid, ns, tp = solver(n, ratio, mode, f0=geometry(geo))
    rng = np.random.default_rng(3)
    for ax in range(2):
        ns.u[ax].interior = U0[ax] + pert * rng.standard_normal((n, n)) if pert else U0[ax] + 0.0 * ns.u[ax].interior
    ns.start()
    es, t0 = [], time.time()
    for k in range(steps):
        try:
            ns.step()
        except Exception as exc:
            es.append(float("inf")); break
        e = max(float(np.abs(to_numpy(ns.u[ax].interior) - U0[ax]).max()) for ax in range(2))
        es.append(e)
        if not math.isfinite(e) or e > 1e2:
            break
    over = next((i + 1 for i, e in enumerate(es) if not (e <= max(1e-3, 30 * pert))), None)
    pick = {str(k): es[k - 1] for k in (50, 200, 500, 1000, 2000) if k <= len(es)}
    return {"geo": geo, "mode": mode, "ratio": ratio, "n": n, "pert": pert, "steps": len(es),
            "max": max(es), "over_at": over, "at": pick, "sec": time.time() - t0}

if __name__ == "__main__":
    J = []
    for geo in ("drop", "bubble", "small", "film", "slant", "two"):
        for mode in ("none", "face", "accel", "trapezoid"):
            for ratio in (1000.0, 1e6):
                J.append((geo, mode, ratio, 32, 2000, 0.0))
    for geo in ("drop", "bubble", "slant"):
        for mode in ("none", "accel", "trapezoid"):
            J.append((geo, mode, 1000.0, 32, 1000, 1e-3))
            J.append((geo, mode, 1000.0, 64, 600, 0.0))
    with mp.get_context("spawn").Pool(26) as pool, open("audit/pr155/break_stability.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J):
            fh.write(json.dumps(r) + "\n"); fh.flush()

"""監修（C）C-1：限定の適用の閾値（利得 > θ で切る。A は θ = 1 の 1 通り）への依存。A のフックの比較の 2 行だけを差し替える。"""
import os, math, warnings, json, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    theta, name, U, ratio, kw = a
    import cases.r0_aligned_free_stream as m
    new = []
    for old, rep in m._SWEEP_EDITS:
        rep = rep.replace("cut_ = (gain_ > rem_)", "cut_ = (gain_ > %r * rem_)" % theta).replace("gi = xp.where(gain_ > rem_, 0.0, gi)", "gi = xp.where(gain_ > %r * rem_, 0.0, gi)" % theta)
        new.append((old, rep))
    assert any(("%r * rem_" % theta) in r for _, r in new)
    m._SWEEP_EDITS[:] = new
    r = m.free_stream(name, "d1+d2lim", ratio, U=U, **kw)
    return theta, name, U, ratio, kw, r["max_err"]

if __name__ == "__main__":
    J = []
    for th in (0.25, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0, 1000.0):
        for ratio in (1e3, 1e6):
            J += [(th, "column", (1.0, 0.0), ratio, {}), (th, "column", (1.0, 0.37), ratio, {}), (th, "square", (1.0, 0.0), ratio, {}),
                  (th, "column", (1.0, 0.0), ratio, {"dt": 0.4999 / 32}), (th, "square", (1.0, 0.0), ratio, {"dt": 0.4999 / 32})]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool, open("audit/pr164/theta.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
    R = [json.loads(l) for l in open("audit/pr164/theta.jsonl")]
    for ratio in (1e3, 1e6):
        for th in (0.25, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0, 1000.0):
            rs = [r for r in R if r[0] == th and r[3] == ratio]
            print("ratio %-6g theta %-6g: worst max|u-U| over %d aligned cases %.2e" % (ratio, th, len(rs), max(r[5] for r in rs)), flush=True)

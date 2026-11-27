"""監修（C）：ノート113 の表の一部を A のコードで再現し、A が試していない条件（boost の向き、粘性 = λ）で壊しにいく。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp

def job(a):
    tag, name, n, flow, nu, U = a
    import cases.r0p1_unsteady_viscous as m
    m.BOOST = U
    o = m.run(n, 0.5, flow=flow, nu=nu, boost=True, **m.CONFIGS[name])
    e = o["errors"][-1] if o["errors"] else {}
    return (tag, name, n, flow, nu, U, e.get("max"), e.get("rms"), e.get("c1"), e.get("c2"), e.get("sum_dt2"), o.get("failed"))

if __name__ == "__main__":
    J = []
    for name in ("cons", "cons/ab", "cons/ab+visc", "cons/ab+cn", "cons/ab+visc+cn", "cons/ab+lag+cn", "plain", "plain+visc+cn"):
        for n in (64, 128):
            J.append(("repro", name, n, "tg", 0.01, (1.0, 0.37)))
            J.append(("repro", name, n, "asym", 0.004, (1.0, 0.37)))
    for name in ("cons", "cons/ab", "cons/ab+visc+cn", "cons/ab+lag+cn", "plain+visc+cn"):
        for n in (48, 64, 96, 128, 192):
            J.append(("U2", name, n, "asym", 0.004, (-0.6, 1.1)))       # 別の向き・大きさの boost
            J.append(("nu3", name, n, "tg", 0.03, (1.0, 0.37)))         # λ = 2.37（3 倍の粘性）
            J.append(("nu.1", name, n, "asym", 0.0004, (0.3, -0.8)))    # λ = 0.079（弱い粘性）、別の boost
    J.sort(key=lambda a: -a[2])
    with mp.get_context("spawn").Pool(24) as pool, open("audit/note113/rerun_and_break.jsonl", "w") as fh:
        for r in pool.imap_unordered(job, J):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r, flush=True)

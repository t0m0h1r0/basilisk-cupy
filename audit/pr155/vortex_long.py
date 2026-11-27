"""監修（C）：渦の中の液滴（厳密解なし）を、格子 4 点と長い時間で。加速度の形の運動量の非保存が細分化で減るか、
安定な 2 つの形が t = 0.25 より先でも発散しないか。あわせて単相の 2Δt の振動（引き戻さない形）の再現。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp

def job(a):
    from cases.r0p1_source_feedback import vortex, single_phase
    if a[0] == "single":
        return a, single_phase(a[2], a[1])
    mode, n, t = a[1:]
    return a, vortex(mode, n=n, t_end=t)

if __name__ == "__main__":
    J = [("vortex", m, n, 0.25) for m in ("none", "accel", "trapezoid") for n in (32, 48, 64, 96)]
    J += [("vortex", m, n, 1.0) for m in ("none", "accel", "trapezoid") for n in (32, 64)]
    J += [("single", m, n) for m in ("none", "face", "trapezoid") for n in (32, 64)]
    with mp.get_context("spawn").Pool(12) as pool:
        for a, r in pool.imap_unordered(job, J):
            if a[0] == "single":
                print("single", a[1:], {k: v for k, v in r.items() if k not in ("mode", "n")}, flush=True)
            else:
                print("vortex %-9s N=%-3d t=%.2f  failed=%s  umax %.3g  momentum change %.2e  energy %.5f -> %.5f" % (a[1], a[2], a[3], r["failed"], r["umax"], r["momentum_change"], r["energy0"], r["energy1"]), flush=True)

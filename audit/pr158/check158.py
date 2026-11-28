import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
def job(a):
    import cases.r0p1_source_feedback as m
    if a[0] == "g":
        _, mode, r, shape = a
        o = m.growth(r, mode, 32, 0.5, 300, shape=shape)
        return "growth %-12s %-6s ratio=%-6g rate %.3g unstable %s max %.2e" % (mode, shape, r, o["rate"], o["unstable"], o["max"])
    _, mode, n = a
    o = m.single_phase(n, mode)
    return "single %-12s N=%-3d max %.3e  g step-to-step %.1f%%" % (mode, n, o["max"], 100 * o["g_step_to_step"])
if __name__ == "__main__":
    J = [("g", m, r, s) for m in ("trapezoid", "trapezoid-p") for r in (1e3, 1e6) for s in ("drop", "bubble", "band")]
    J += [("s", m, n) for m in ("none", "trapezoid", "trapezoid-p") for n in (32, 64, 128, 256)]
    with mp.get_context("spawn").Pool(10) as pool:
        for line in pool.imap(job, J, chunksize=1):
            print(line, flush=True)

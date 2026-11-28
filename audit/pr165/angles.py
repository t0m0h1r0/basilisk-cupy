import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
def run(a):
    from cases.r0p2_corner import potential, balance
    th, off, n = a
    m = potential(th, off, n, "mirror")
    r = {}
    for g in ("mirror", "plane"):
        o = balance(th, off, n, ghosts=g, t_end=0.5, samples=(0.5,))
        h = o.get("history", [])
        r[g] = h[-1]["cells"] if h else float("nan")
    return th, off, n, m["max_phi"], r["mirror"], r["plane"]
if __name__ == "__main__":
    J = [(th, off, 64) for th in (5, 15, 20, 30, 35, 37.5, 40, 50, 55, 70) for off in (0.0, 0.0137, 0.11)]
    with mp.get_context("spawn").Pool(10, maxtasksperchild=1) as pool:
        R = list(pool.imap(run, J, chunksize=1))
    print("theta   off     max|phi|(t=0)  cells(t=0.5) mirror   plane-ghosts")
    for r in R:
        print("%-6g %+.4f  %.1e      %.1e    %.1e %s" % (r[0], r[1], r[3], r[4], r[5], "  <-- fails with phi=0 at t=0" if r[3] < 1e-12 and r[4] > 1e-8 else ("  <-- plane ghosts do not cure" if r[5] > 1e-8 else "")))
    print("fail (mirror):", sum(r[4] > 1e-8 for r in R), "of", len(R), "| with phi(t=0)=0:", sum(r[3] < 1e-12 and r[4] > 1e-8 for r in R), "| not cured by plane ghosts:", sum(r[5] > 1e-8 for r in R))

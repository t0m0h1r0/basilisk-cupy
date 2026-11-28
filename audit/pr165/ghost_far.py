import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
def run(a):
    from cases.r0p2_corner import potential, balance
    th, off, n = a
    m = potential(th, off, n, "mirror"); e = potential(th, off, n, "exact")
    o = balance(th, off, n, ghosts="plane", t_end=1.0, samples=(1.0,))
    h = o.get("history", [])
    far = max((min(((c["cell"][0] + .5 - cx) ** 2 + (c["cell"][1] + .5 - cy) ** 2) ** .5 for cx in (0, n) for cy in (0, n)) for c in m["bad"]), default=0)
    cols = sorted({c["column"] for c in m["bad"]})
    return "theta=%-6g off=%+.4f N=%-3d: mirror max|phi| %.1e (%d cells, up to %.1f cells from a corner, columns %s) | exact ghosts max|phi| %.1e | plane-ghost solver t=1 cells %s" % (th, off, n, m["max_phi"], len(m["bad"]), far, cols, e["max_phi"], ["%.1e" % r["cells"] for r in h] if h else o)
if __name__ == "__main__":
    J = [(37.5, 0.45, 64), (37.5, 0.0, 64), (30.0, 0.40, 64), (26.565, 0.40, 64), (42.5, -0.4415, 64)]
    with mp.get_context("spawn").Pool(5, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

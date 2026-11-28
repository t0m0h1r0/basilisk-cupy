"""N=128, nu=0.05, CFL 0.5, t=0.5 の mix。第2版の d1 の各部分を戻す。"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
sys.path.insert(0, "."); sys.path.insert(0, "audit/d179b")
from r0p1_d1v2 import run

def job(a):
    level, n, nu = a
    o = run(n, 0.5, flow="mix", nu=nu, boost=True, t_ends=(0.5,), variant="conserving", level=level, source="g+visc", visc="cn")
    return "%-22s N=%d nu=%g: max err %.3e" % (level, n, nu, o["errors"][-1]["max"])

if __name__ == "__main__":
    J = [(l, n, nu) for l in ("d1v2", "d1v2-nopredL", "d1v2-srcold", "d1v2-addold", "d1v2-srcold-addold", "pred") for n, nu in ((64, 0.05), (128, 0.05))]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        for line in pool.imap(job, J, chunksize=1):
            print(line, flush=True)

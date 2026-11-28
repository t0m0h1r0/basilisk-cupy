"""二相の既定（V-BE、源の項は g_P、予測に粘性なし）を単相で代理：大きい kappa = nu dt/h^2。出し入れに g_P か g^n。"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
sys.path.insert(0, "."); sys.path.insert(0, "audit/d179b")
from r0p1_d1v2 import run

def job(a):
    level, n, nu = a
    o = run(n, 0.5, flow="mix", nu=nu, boost=True, t_ends=(0.5,), variant="conserving", level=level, source="g", visc="be")
    e = o["errors"][-1]
    return "%-12s N=%d nu=%g kappa~%.2g: max err %.3e" % (level, n, nu, nu * (0.5 / e["steps"]) * n * n, e["max"])

if __name__ == "__main__":
    J = [(l, n, nu) for l in ("d1v2", "d1v2-addold", "pred", "n") for n, nu in ((64, 0.05), (64, 0.5), (64, 2.0), (128, 0.5))]
    with mp.get_context("spawn").Pool(16, maxtasksperchild=1) as pool:
        for line in pool.imap(job, J, chunksize=1):
            print(line, flush=True)

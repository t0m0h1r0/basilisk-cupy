"""二相の既定の形（V-BE、源の項 g_P、予測に粘性なし）を単相で代理：N=128・256 の大きい kappa。"""
import sys; sys.path.insert(0, "audit/d179c")
from kappa2 import *
CFG["d1-BE"] = dict(level="d1v2", source="g", visc="be")
CFG["d1-BE-srcold"] = dict(level="d1v2-srcold", source="g", visc="be")
CFG["c-BE"] = dict(level="n", source="g", visc="be")
if __name__ == "__main__":
    J = [(c, n, nu, 0.5) for c in ("d1-BE", "c-BE") for n, nu in ((128, 2.0), (128, 4.0), (128, 16.0), (256, 1.0))]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(k2, J, chunksize=1):
            print(line, flush=True)

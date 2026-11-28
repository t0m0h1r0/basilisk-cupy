import sys; sys.path.insert(0, "audit/d179c")
from kappa2 import *
CFG["d1-srcold"] = dict(level="d1v2-addold-srcold", source="g+visc", visc="cn")
CFG["d1-noL-srcold"] = dict(level="d1v2-nopredL-addold-srcold", source="g+visc", visc="cn")
if __name__ == "__main__":
    J = [(c, 128, nu, 0.5) for c in ("d1-srcold", "d1-noL-srcold") for nu in (1.0, 2.0, 4.0)]
    with mp.get_context("spawn").Pool(6, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(k2, J, chunksize=1):
            print(line, flush=True)

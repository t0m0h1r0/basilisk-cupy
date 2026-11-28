"""密度比 100 で、required.py の自由流（CFL 0.5）と運動量の収支を繰り返す。"""
import sys, multiprocessing as mp
sys.path.insert(0, "audit/c6b")
import required as R
from basilisk_cupy.interface import two_phase as T

def go(j):
    init = T.TwoPhase.__init__
    def patched(self, *a, **kw):
        for k in ("rho1", "rho2"):
            if kw[k] == 1e-3:
                kw[k] = 1e-2
        init(self, *a, **kw)
    T.TwoPhase.__init__ = patched
    return "ratio100 " + R.call(j)

if __name__ == "__main__":
    J = [(R.free, (g, 64, h, 0.5, mu)) for g in ("film05", "film15", "twodrops", "tiny", "walldrop", "wallfilm") for h in (True, False) for mu in (0.0, 1e-2)]
    J += [(R.mom, (g, 64, h, ns_, lid, 0.0)) for g in ("drop", "walldrop") for h in (True, False) for ns_, lid in ((False, 0.0), (True, 1.0))]
    with mp.get_context("spawn").Pool(14, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(go, J, chunksize=1):
            print(line, flush=True)

"""監修 C-9 の 2：振動子のモデルの予測を実際のソルバーで確かめる（反証できる形）。E1 の下で、表面張力のポテンシャルを
(σκ(f^n) + σκ(f^{n+1}))/2 から作れば、追加の減衰は消えるはず。E1 なしで同じことをすると逆符号の追加（負の減衰）になるはず。
厳密な線形の減衰率 0.7147（audit/pr177/capwave_exact.py）。"""
import os, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

def run(a):
    label, e1, mid, n = a
    import cases.r0_combined_tension as t
    from cases.r0_combined import combined
    from basilisk_cupy.interface import two_phase
    from basilisk_cupy.interface.heights import NODATA

    add = two_phase.TwoPhase.add_surface_tension

    def patched(self, sigma, *aa, **k):
        add(self, sigma, *aa, **k)
        if not mid:
            return
        num = self.potentials[-1]
        st = {}
        self.ns.on("vof", lambda s: st.__setitem__("old", num(self)))      # f^n（輸送の前）

        def avg(tp):
            new = num(tp)
            old = st.get("old", new)
            xp = tp.f.grid.xp
            a_, b_ = old < NODATA / 2, new < NODATA / 2
            return xp.where(a_ & b_, 0.5 * (old + new), xp.where(b_, new, old))
        self.potentials[-1] = avg

    two_phase.TwoPhase.add_surface_tension = patched
    t._ctx = lambda conf: combined(e1=True, e2=False, e3=False, band=None) if e1 else contextlib.nullcontext()
    r = t.capillary_wave("x", n)
    return "%-26s N=%-3d damping %.4f (extra over exact 0.7147: %+.4f)  period %.5f %s" % (label, n, r["damping"], r["damping"] - 0.7147, r["period"], r["failed"] or "")

if __name__ == "__main__":
    V = [("base (library)", False, False), ("E1", True, False), ("E1 + force at midpoint", True, True), ("no E1 + force at midpoint", False, True)]
    J = [(l, e, m, n) for l, e, m in V for n in (32, 64)]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(run, J, chunksize=1):
            print(line, flush=True)

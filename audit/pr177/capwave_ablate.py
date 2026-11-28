"""条件 9 の減衰率の悪化がどの修正から来るか（E3 は外した上で、1 つずつ）。厳密値 0.7147。"""
import os, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
def run(a):
    label, d2, kw, n = a
    import cases.r0_combined_tension as t
    from cases.r0_combined import combined
    from cases.r0_aligned_free_stream import scheme
    def ctx(conf):
        s = scheme(True, d2) if d2 is not None else contextlib.nullcontext()
        c = combined(**kw) if kw is not None else contextlib.nullcontext()
        class both:
            def __enter__(self): s.__enter__(); c.__enter__()
            def __exit__(self, *e): c.__exit__(*e); s.__exit__(*e)
        return both()
    t._ctx = ctx
    r = t.capillary_wave("x", n)
    return "%-28s N=%d: damping %.4f (%+.1f%% vs exact 0.7147)  period %.5f %s" % (label, n, r["damping"], 100 * (r["damping"] / 0.7147 - 1), r["period"], r["failed"] or "")
if __name__ == "__main__":
    V = [("library", None, None), ("D1+D2lim only", "limited", None), ("E1 only", None, dict(e1=True, e2=False, e3=False, band=1)),
         ("E2 only (band)", None, dict(e1=False, e2=True, e3=False, band=1)), ("source only (no E1,E2)", None, dict(e1=False, e2=False, e3=False, band=1)),
         ("noE3 without E2", "limited", dict(e1=True, e2=False, e3=False, band=1)), ("noE3 all", "limited", dict(e1=True, e2=True, e3=False, band=1))]
    J = [(l, d, k, n) for l, d, k in V for n in (32, 64)]
    with mp.get_context("spawn").Pool(14, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

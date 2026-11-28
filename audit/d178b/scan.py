"""監修 #178 改訂（5aae288）：設計の A-1（壁で閉じた列）の実装そのものを、監修の 1794 配置（t = 0 の静的な走査）と、追加の配置
（壁の 1 セル以内の平行な界面、ごく浅い角度、壁の近くの切片）に掛ける。N = 32・64（追加は 128 も）。"""
import sys, math, warnings, importlib.util, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
import multiprocessing as mp
spec = importlib.util.spec_from_file_location("d178", "design/checks/178_heights.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

def run(a):
    n, th, off = a
    grid, c, G, Z = D.plane(n, th, off)
    bc = D.BoundaryConditions()
    out = {}
    for var in ("library", "A-1", "A-1+A-2"):
        try:
            p = D.position(c, bc, grid, G, Z, variant=var, eps=D.EPS if var != "library" else None)
        except Exception as ex:
            out[var] = ("EXC " + repr(ex)[:80],)
            continue
        ok = p < D.H.NODATA / 2
        v = np.abs(np.where(ok, p, 0.0))
        miss = 0
        for ax in (0, 1):
            fl, fr = c.take(range(0, n - 1), ax), c.take(range(1, n), ax)
            ol, orr = ok.take(range(0, n - 1), ax), ok.take(range(1, n), ax)
            miss += int(((np.abs(fr - fl) > 1e-8) & ~ol & ~orr).sum())
        out[var] = (float(v.max()), D.face_force(np.where(ok, p, D.H.NODATA), c, False), miss)
    return a, out, int(((c > 1e-10) & (c < 1 - 1e-10)).sum())

def report(title, J):
    with mp.get_context("spawn").Pool(16) as pool:
        R = pool.map(run, J, chunksize=8)
    R = [r for r in R if r[2] > 0]
    print("%s: configurations with partial cells %d" % (title, len(R)))
    for var in ("library", "A-1", "A-1+A-2"):
        exc = [(a, o[var]) for a, o, _ in R if isinstance(o[var][0], str)]
        good = [(a, o[var]) for a, o, _ in R if not isinstance(o[var][0], str)]
        bad = [(a, o) for a, o in good if o[0] > 1e-8]
        print("   %-8s max|phi|>1e-8: %4d  face force>1e-8: %4d  worst %.2e  faces with f jump and no phi: %d  exceptions %d" % (
            var, len(bad), sum(o[1] > 1e-8 for _, o in good), max((o[0] for _, o in good), default=0), sum(o[2] for _, o in good), len(exc)))
        for a, o in sorted(bad, key=lambda x: -x[1][0])[:3]:
            print("        n=%d theta=%g off=%+.5f: max|phi| %.2e" % (*a, o[0]))
        for a, o in exc[:2]:
            print("        n=%d theta=%g off=%+.5f: %s" % (*a, o[0]))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    J = [(n, float(th), float(off)) for n in (32, 64) for th in np.arange(0, 90.01, 2.5) for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)])]
    report("audit 1850 scan (seed 0)", J)
    rng = np.random.default_rng(7)
    J = [(n, float(th), float(off)) for n in (32, 64, 128) for th in rng.uniform(0, 90, 60) for off in rng.uniform(-0.49, 0.49, 25)]
    report("random angles and offsets (seed 7)", J)
    J = []
    for n in (32, 64):
        h = 1.0 / n
        near = [s * (0.5 - d * h) for s in (1, -1) for d in (0.05, 0.3, 0.5, 0.9, 1.3, 1.7, 2.4)]
        J += [(n, th, off) for th in (0.0, 0.1, 0.5, 1.0, 2.0, 88.0, 89.0, 89.5, 90.0) for off in near]
    report("planes within 2.4 cells of a wall", J)

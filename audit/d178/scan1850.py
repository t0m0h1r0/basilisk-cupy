"""監修 #178：設計の A-1・A-2（design/checks/178_heights.py の実装そのもの）を、監修の #165 の 1850 配置（t = 0 の静的な走査）に
かける。反例探し。平面なので、候補が正しければ max|phi| は丸め。"""
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
    real = ((c > 1e-10) & (c < 1 - 1e-10))
    for var in ("library", "A-1", "A-2", "A-1+A-2"):
        p = D.position(c, bc, grid, G, Z, variant=var, eps=D.EPS if var != "library" else None)
        ok = p < D.H.NODATA / 2
        v = np.abs(np.where(ok, p, 0.0))
        # 力に現れるもの：f が跳ぶ面で片側にも φ が無い面の数（釣り合いは崩れないが力が欠ける）
        miss = 0
        for ax in (0, 1):
            fl, fr = c.take(range(0, n - 1), ax), c.take(range(1, n), ax)
            ol, orr = ok.take(range(0, n - 1), ax), ok.take(range(1, n), ax)
            miss += int(((np.abs(fr - fl) > 1e-8) & ~ol & ~orr).sum())
        out[var] = (float(v.max()), D.face_force(np.where(ok, p, D.H.NODATA), c, False), miss, int(ok.sum()))
    return a, out, int(real.sum())

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    J = []
    for n in (32, 64):
        for th in np.arange(0, 90.01, 2.5):
            for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)]):
                J.append((n, float(th), float(off)))
    with mp.get_context("spawn").Pool(16) as pool:
        R = pool.map(run, J, chunksize=8)
    R = [r for r in R if r[2] > 0]
    print("configurations with an interface:", len(R))
    for var in ("library", "A-1", "A-2", "A-1+A-2"):
        bad = [(a, o[var]) for a, o, _ in R if o[var][0] > 1e-8]
        force = [(a, o[var]) for a, o, _ in R if o[var][1] > 1e-8]
        miss = sum(o[var][2] for a, o, _ in R)
        print("%-8s max|phi|>1e-8: %4d   face |phi_f df|>1e-8: %4d   worst %.2e   faces with f jump and no phi on either side: %d" % (var, len(bad), len(force), max(o[var][0] for a, o, _ in R), miss))
        for a, o in sorted(bad, key=lambda x: -x[1][0])[:4]:
            print("            n=%d theta=%.1f off=%+.4f max|phi| %.2e force %.2e" % (a[0], a[1], a[2], o[0], o[1]))

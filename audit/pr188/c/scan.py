"""C 版の走査：帯（位置のずれ 6 通り × CFL 19 通り × 密度比 1e3・1e6、U=(1,0)）と、帯・正方形の U=(1,0.37)。θ = 1・3・無限大。"""
import subprocess, sys, itertools
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1]
cfls = [0.26, 0.3, 0.33, 0.36, 0.38, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.495, 0.498, 0.499, 0.4999]
J = [(0, r, 1, 0, c, o) for c in cfls for o in (0, 0.05, 0.13, 0.25, 0.4, 0.5) for r in (1e-3, 1e-6)]
J += [(s, r, 1, 0.37, c, 0) for s in (0, 1) for r in (1e-3, 1e-4, 1e-6) for c in (0.4, 0.45, 0.49, 0.5)]
J += [(1, r, 1, 0, c, 0) for r in (1e-3, 1e-6) for c in cfls]
def run(a):
    t, j = a
    p = subprocess.run(["%s/a_%s" % (D, t)] + [str(v) for v in j], capture_output=True, text=True, timeout=300)
    line = [l for l in p.stderr.splitlines() if l.startswith("RESULT")]
    return t, j, float(line[0].split()[2]) if line else float("inf")
with ThreadPoolExecutor(8) as ex:
    R = list(ex.map(run, [(t, j) for t in ("1", "3", "1e30") for j in J]))
for t in ("1", "3", "1e30"):
    rs = [r for r in R if r[0] == t]
    for ratio in (1e-3, 1e-4, 1e-6):
        sub = [r for r in rs if r[1][1] == ratio]
        w = max(sub, key=lambda r: r[2] if r[2] == r[2] else 1e99)
        print("theta=%-5s rho2=%g: %3d cases  worst max|u-U| %.2e  (shape,rho2,Ux,Uy,CFL,off)=%s  n(>1e-8)=%d" % (t, ratio, len(sub), w[2], w[1], sum(not r[2] <= 1e-8 for r in sub)))

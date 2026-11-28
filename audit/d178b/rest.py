import sys, math, warnings, importlib.util
warnings.filterwarnings("ignore")
exec(open("audit/d178b/scan.py").read().split('def report')[0])
rng = np.random.default_rng(0)
J = [(n, float(th), float(off)) for n in (32, 64) for th in np.arange(0, 90.01, 2.5) for off in np.concatenate([np.linspace(-0.45, 0.45, 13), rng.uniform(-0.45, 0.45, 12)])]
for n, th, off in J:
    if th != 45.0:
        continue
    grid, c, G, Z = D.plane(n, th, off)
    p = D.position(c, D.BoundaryConditions(), grid, G, Z, variant="A-1", eps=D.EPS)
    bad = np.argwhere((p < D.H.NODATA / 2) & (np.abs(np.where(p < D.H.NODATA / 2, p, 0)) > 1e-8))
    if len(bad):
        part = np.argwhere((c > 1e-10) & (c < 1 - 1e-10))
        print("n=%d off=%+.5f: bad cells %s  f=%s  phi=%s | partial cells in the box %d, touching walls: %s" % (
            n, off, [tuple(int(v) for v in b) for b in bad[:6]], ["%.4g" % c[tuple(b)] for b in bad[:6]], ["%.2e" % p[tuple(b)] for b in bad[:6]],
            len(part), sorted({tuple(int(v) for v in q) for q in part if min(q[0], q[1], n - 1 - q[0], n - 1 - q[1]) == 0})[:8]))

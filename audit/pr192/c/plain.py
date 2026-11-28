exec(open("audit/pr192/c/one.py").read().split("for t in")[0])
def lib(stale, kw):
    grid = CartesianGrid((N, N))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-12)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, conserving_stale_ghosts=stale, **kw)
    ns.cfl = 0.5; ns.DT = 0.007
    ns.u[0].interior = np.tile(ux, (N, 1)); ns.u[1].interior = np.ones((N, N))
    ns.start()
    for _ in range(NS):
        ns.step()
    return ns.u[0].interior, ns.u[1].interior, tp.f.interior
d = np.loadtxt("%s/out.txt" % sys.argv[1]); i, j = d[:, 0].astype(int), d[:, 1].astype(int)
C = [np.empty((N, N)) for _ in range(3)]
for a, col in zip(C, (2, 3, 4)):
    a[i, j] = d[:, col]
for stale in (False, True):
    L = lib(stale, {})
    print("plain C vs library default stale=%d after %d steps: ux %.1e uy %.1e f %.1e" % (stale, NS, *[np.abs(a - b).max() for a, b in zip(L, C)]))
    L2 = lib(stale, dict(conserving_vof_half=True))
    print("   same, conserving_vof_half=True (C's rule for c=1/2): ux %.1e uy %.1e f %.1e" % tuple(np.abs(a - b).max() for a, b in zip(L2, C)))

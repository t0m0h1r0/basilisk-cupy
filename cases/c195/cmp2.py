exec(open("cmp.py").read().split("kw = dict(")[0])
for stale in (False, True):
    for half in (False, True):
        grid = CartesianGrid((N, N)); ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-12)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, conserving_stale_ghosts=stale, conserving_vof_half=half)
        ns.cfl = 0.5; ns.DT = 0.007
        ns.u[0].interior = np.tile(ux, (N, 1)); ns.u[1].interior = np.ones((N, N))
        out = {}
        ns.on("tracer_advection", lambda s: out.__setitem__(s.clock.i, (s.u[0].interior.copy(), s.u[1].interior.copy())), after=True)
        ns.start()
        for _ in range(3): ns.step()
        for it in (1, 2):
            C = loadC("vof", it); a, b = out[it]
            print("stale=%d vof_half=%d step %d: |lib-C| ux %.1e uy %.1e | x-nonuniformity lib ux %.1e uy %.1e, C ux %.1e uy %.1e" % (stale, half, it, np.abs(a - C[0]).max(), np.abs(b - C[1]).max(), np.ptp(a, axis=0).max(), np.ptp(b, axis=0).max(), np.ptp(C[0], axis=0).max(), np.ptp(C[1], axis=0).max()))

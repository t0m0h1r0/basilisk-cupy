from audit.pr160.half import run
for rho1, rho2 in ((1.0, 1.0), (1.0, 1e-3), (1e-3, 1.0), (1.0, 1e-6), (1e-6, 1.0)):
    for kind in ("column",):
        print(rho1, rho2, "  ".join("steps=%d first=%d: %.2e" % (s, i0, run(kind, True, rho1, rho2, False, i0, True, steps=s)) for s in (1, 6, 30) for i0 in (0, 1)), flush=True)

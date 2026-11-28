exec(open("cmp.py").read().split("kw = dict(")[0])
import basilisk_cupy.interface.two_phase as T
orig = T._MomentumField.fill_ghosts
def run(mode):
    if mode == "H1":   # the ghosts made once (the first fill of the step), then kept for the second sweep
        def fill(self):
            if getattr(self, "_frozen", None) is not None:
                inner = self.interior.copy(); self.data[...] = self._frozen; self.interior = inner; return self
            orig(self); self._frozen = self.data.copy(); return self
        T._MomentumField.fill_ghosts = fill
    if mode == "H1p":  # frozen once per step, and made at the ADJACENT interior cell (not the periodic image)
        import numpy as _np
        def fill(self):
            if getattr(self, "_frozen", None) is not None:
                inner = self.interior.copy(); self.data[...] = self._frozen; self.interior = inner; return self
            g = self.grid.ghosts
            inner = self.interior.copy()
            self.data[...] = _np.pad(inner, g, mode="edge"); self._frozen = self.data.copy(); return self
        T._MomentumField.fill_ghosts = fill
    try:
        grid = CartesianGrid((N, N)); ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-12)
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, conserving_stale_ghosts=True, conserving_vof_half=True)
        ns.cfl = 0.5; ns.DT = 0.007
        ns.u[0].interior = np.tile(ux, (N, 1)); ns.u[1].interior = np.ones((N, N))
        out = {}
        ns.on("tracer_advection", lambda s: out.__setitem__(("vof", s.clock.i), [s.u[0].interior.copy(), s.u[1].interior.copy(), tp.f.interior.copy()]), after=True)
        ns.on("end_timestep", lambda s: out.__setitem__(("end", s.clock.i), [s.u[0].interior.copy(), s.u[1].interior.copy(), tp.f.interior.copy()]), after=True)
        ns.start()
        for _ in range(3): ns.step()
    finally:
        T._MomentumField.fill_ghosts = orig
    C = loadC("vof", 2); L = out[("vof", 2)]
    for nm, a, b in zip(("ux", "uy"), L[:2], C[:2]):
        d = np.abs(a - b); idx = np.argwhere(d > 1e-9)
        print(nm, "cells", len(idx), "i set", sorted(set(idx[:, 0].tolist()))[:8], "j set", sorted(set(idx[:, 1].tolist()))[:8])
    C0 = loadC("end", 1); L0 = out[("end", 1)]
    print("end of step 1, x-nonuniformity: C ux %.1e, lib ux %.1e" % (np.ptp(C0[0], axis=0).max(), np.ptp(L0[0], axis=0).max()))
    for it in (0, 1, 2):
        for tag in ("vof", "end"):
            C = loadC(tag, it); L = out[(tag, it)]
            print("%-3s step %d %-3s |lib-C| ux %.1e uy %.1e f %.1e" % (mode, it, tag, *[np.abs(a - b).max() for a, b in zip(L, C[:3])]))
def where(mode):
    pass
for m in ("H1", "H1p"): run(m)
# where does step 2 differ under H1?

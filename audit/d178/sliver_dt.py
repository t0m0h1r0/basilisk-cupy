"""監修 #178：sliver_periodic の時間刻み。DT なし（ノート120）と DT = 0.5/N を、Δt・CFL・f の変化と一緒に比べる。
雑音なし／雑音あり、DT ありで長く（400 ステップ）も。"""
import warnings, math
import numpy as np
warnings.filterwarnings("ignore")
from basilisk_cupy.time.basilisk_centered import CenteredNS
import cases.r0p2_corner as C

info = {}
orig_start, orig_step = CenteredNS.start, CenteredNS.step
def run(nv, n, DT, steps=40, noise=1e-16, off=0.0137):
    def start(self, *a, **k):
        if DT:
            self.DT = DT
        orig_start(self, *a, **k)
        info.clear(); info.update(dt=0.0, cfl=0.0, df=0.0, f0=None)
    def step(self):
        orig_step(self)
        dt = self.clock.dt
        info["dt"] = max(info["dt"], dt)
        info["cfl"] = max(info["cfl"], max(float(np.abs(c).max()) for c in self.uf.components) * dt * n)
        info["t"] = self.clock.t
    CenteredNS.start, CenteredNS.step = start, step
    try:
        from basilisk_cupy.interface.two_phase import TwoPhase
        keep = {}
        init = TwoPhase.__init__
        def patched(self, ns, f, **k):
            init(self, ns, f, **k)
            keep["tp"], keep["f0"] = self, np.array(f.interior)
        TwoPhase.__init__ = patched
        r = C.sliver_periodic(nv, off, n, steps, noise)
        TwoPhase.__init__ = init
    finally:
        CenteredNS.start, CenteredNS.step = orig_start, orig_step
    df = float(np.abs(np.asarray(keep["tp"].f.interior) - keep["f0"]).max())
    print("n=%s N=%-3d DT=%-6s noise=%g steps=%d: dev %.2e cells  slivers %d  max dt %.1e  max CFL %.2f  max|f-f0| %.1e  t_end %.2e"
          % (nv, n, "0.5/N" if DT else "none", noise, steps, r["devmax_cells"], r["slivers_max"], info["dt"], info["cfl"], df, info["t"]), flush=True)

for nv in ((1, 1), (1, 2), (2, 3)):
    for n in (32, 64):
        run(nv, n, None)
        run(nv, n, 0.5 / n)
run((1, 2), 32, 0.5 / 32, steps=1000)
run((1, 2), 32, 0.5 / 32, steps=200, noise=1e-12)
run((1, 2), 32, 0.5 / 32, steps=200, noise=1e-3)

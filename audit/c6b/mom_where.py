"""x 運動量の残差がどのステップ・どの段で生じるか（x 周期・滑り壁、非粘性/粘性）。初期速度が離散的に発散なし/ありの 2 通り。"""
import sys, math, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "audit/c6b")
from required import setup

def run(geom, heavy, divfree, n=64, steps=8, tol=1e-11, mu=1e-2):
    grid, ns, tp, f0, to_numpy = setup(n, geom, heavy, mu, True, tol=tol)
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    if divfree:   # 中心差分で離散的に発散なし、壁で v=0
        u0 = np.sin(np.pi * Y) * np.ones_like(X); v0 = np.zeros_like(X)
    else:
        u0 = np.sin(np.pi * Y) * (1.0 + 0.5 * np.sin(2 * np.pi * X)); v0 = 0.3 * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    ns.u[0].interior = u0; ns.u[1].interior = v0
    ns.DT = 0.25 / n
    ns.start()
    def P():
        f = np.clip(to_numpy(tp.f.interior), 0, 1)
        return float(((f * (tp.rho1 - tp.rho2) + tp.rho2) * to_numpy(ns.u[0].interior)).sum()) * h * h
    marks = []
    for stage in ("advection_term", "vof", "properties", "viscous_term", "acceleration", "projection", "end_timestep"):
        ns.on(stage, lambda s, st=stage: marks.append((st, P())))
    prev = P(); out = []; where = None
    for s in range(steps):
        marks.clear()
        ns.step()
        p = P(); d = p - prev
        out.append("%.1e" % d)
        if where is None or abs(d) > where[0]:
            q = prev; w = []
            for st, v in marks:
                w.append("%s %.1e" % (st, v - q)); q = v
            w.append("rest %.1e" % (p - q))
            where = (abs(d), s + 1, w)
        prev = p
    print("%-8s %s divfree=%d tol=%g mu=%g N=%d: dP per step %s\n      worst step %d, change by stage (hook = after preceding stage): %s" % (
        geom, "heavy" if heavy else "light", divfree, tol, mu, n, " ".join(out), where[1], "; ".join(where[2])))

for divfree in (False, True):
    run("drop", True, divfree)
run("drop", True, False, mu=0.0)
run("drop", True, False, tol=1e-13)

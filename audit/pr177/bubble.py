"""監修（C-2）：統合した構成を、R0 の試験の表の「上昇気泡」（Hysing の 2 つの場合。閉じた箱、上下は粘着壁、表面張力、
reduced の重力）で壊しにいく。base：ライブラリ／noE3：E3 を外した統合構成／full：全て。A の cases.r0_combined を使う。"""
import os, math, warnings, json, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
CASES = {1: dict(rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0, sigma=24.5), 2: dict(rho1=1000.0, rho2=1.0, mu1=10.0, mu2=0.1, sigma=1.96)}

def run(a):
    case, conf, n, T = a
    from cases.r0_combined import configuration
    ctx = configuration(conf) if conf != "base" else contextlib.nullcontext()
    with ctx:
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.boundary import Dirichlet
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        p = CASES[case]
        grid = CartesianGrid((n, 2 * n), length=(1.0, 2.0))
        walls = {w: 0.0 for w in ("left", "right", "bottom", "top")}
        ns = CenteredNS(grid, uf_normal=walls, tangential={"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}, tolerance=1e-6, mu=p["mu1"])
        f0 = to_numpy(fraction(grid, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.25 ** 2, refine=8))
        tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions()), rho1=p["rho1"], rho2=p["rho2"], mu1=p["mu1"], mu2=p["mu2"], conserving=True)
        tp.add_surface_tension(p["sigma"])
        tp.add_reduced_gravity(G=(0.0, -0.98), Z=(0.5, 0.5))
        y = (np.arange(2 * n) + 0.5) / n
        ns.DT = 0.5 / n
        ns.start(events=[T])
        vmax, failed, over = 0.0, "", 0.0
        m0 = float(to_numpy(tp.f.interior).sum())
        while ns.clock.t < T - 1e-12:
            try:
                ns.step()
            except Exception as exc:
                failed = repr(exc)[:80]; break
            f = to_numpy(tp.f.interior); b = 1.0 - np.clip(f, 0, 1)
            u = [to_numpy(c.interior) for c in ns.u.components]
            vb = float((b * u[1]).sum() / b.sum()); vmax = max(vmax, vb)
            yc = float((b * y[None, :]).sum() / b.sum())
            over = max(over, -float(f.min()), float(f.max()) - 1.0)
            um = max(float(np.abs(c).max()) for c in u)
            if not math.isfinite(um) or um > 20 or ns.clock.i > 20000:
                failed = "blow-up/step limit t=%.3f |u|=%.3g" % (ns.clock.t, um); break
        return {"case": case, "conf": conf, "n": n, "t": ns.clock.t, "steps": ns.clock.i, "failed": failed, "vrise_max": vmax,
                "yc": yc, "vb_end": vb, "mass_rel": abs(float(to_numpy(tp.f.interior).sum()) - m0) / m0, "overshoot": over}

if __name__ == "__main__":
    J = [(c, s, n, 3.0) for c in (1, 2) for s in ("base", "full-noE3", "d1+d2lim", "full") for n in (32, 64)]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool, open("audit/pr177/bubble.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(r, flush=True)

"""監修（C）C-6 の内容 1（#170、P0）：filtered=False × 表面張力 × 密度比 1000 の成立性。現行のライブラリ（main）。
  drop   静止液滴（閉じた箱 [-1/2, 1/2]^2、R = 0.25、sigma = 1、La = 12000）。曲率は数値（tension.h）または厳密 sigma/R。
  bubble 同じで内外の密度を入れ替える。
  wave   毛管波（capwave.c の配置：2×2、滑り壁、y = 0.01 cos 2 pi x、重い相が下）。
構成：cons（保存形・filtered=False、基準の候補）、cons-filt、plain（非保存形・filtered=False）、plain-filt、
cons-lim（A-1 の候補 D1+D2lim の差し替え。このファイルを PR #164 の版の cases で実行する）。"""
import os, sys, math, json, warnings, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
VARIANTS = {"cons": (True, False), "cons-filt": (True, True), "plain": (False, False), "plain-filt": (False, True), "cons-lim": (True, False)}

def run(a):
    case, variant, n, ratio, curv, mur, T = a
    cons, filt = VARIANTS[variant]
    ctx = contextlib.nullcontext()
    if variant == "cons-lim":
        from cases.r0_aligned_free_stream import scheme
        ctx = scheme(True, "limited")
    with ctx:
        from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
        from basilisk_cupy.backend import to_numpy
        from basilisk_cupy.interface.fractions import fraction
        from basilisk_cupy.interface.heights import NODATA, position
        from basilisk_cupy.interface.two_phase import TwoPhase
        from basilisk_cupy.time.basilisk_centered import CenteredNS
        walls = {w: 0.0 for w in ("left", "right", "bottom", "top")}
        sigma = 1.0
        if case == "wave":
            grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
            f = CellField.from_interior(grid, to_numpy(fraction(grid, lambda x, y: y - 0.01 * np.cos(2 * np.pi * x))), BoundaryConditions())
            rho1, rho2 = 1.0 / ratio, 1.0                     # f = 1 は上（軽い）
            muh = 0.0182571749236 / 10
            mu1, mu2 = muh / mur, muh
        else:
            grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
            R = 0.25
            f0 = to_numpy(fraction(grid, lambda x, y: R * R - x * x - y * y, refine=8))
            f = CellField.from_interior(grid, f0, BoundaryConditions())
            rho1, rho2 = (1.0, 1.0 / ratio) if case == "drop" else (1.0 / ratio, 1.0)
            muh = math.sqrt(2 * R / 12000.0)
            mu1, mu2 = (muh, muh / mur) if case == "drop" else (muh / mur, muh)
        ns = CenteredNS(grid, uf_normal=walls, tolerance=1e-8, mu=max(mu1, mu2))
        tp = TwoPhase(ns, f, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, conserving=cons, filtered=filt)
        tp.add_surface_tension(sigma)
        if curv == "exact" and case != "wave":
            num = tp.potentials.pop()
            def exact(t, num=num):
                k = num(t)                                      # sigma kappa（界面のセル以外は NODATA）
                xp = t.f.grid.xp
                ok = k < NODATA
                sign = float(xp.sign(xp.median(k[ok])))
                return xp.where(ok, sign * sigma / 0.25, NODATA)
            tp.potentials.append(exact)
        f_init = to_numpy(f.interior).copy()
        ns.start(events=[T])
        hist, failed, amp = [], "", []
        umax_run = 0.0
        nxt = 0.0
        while ns.clock.t < T - 1e-12:
            ns.step()
            u = max(float(np.abs(to_numpy(c.interior)).max()) for c in ns.u.components)
            umax_run = max(umax_run, u)
            if not math.isfinite(u) or u > 100:
                failed = "blow-up t=%.4f" % ns.clock.t; break
            if ns.clock.t >= nxt - 1e-12:
                rho = to_numpy(tp.rho(tp.f.interior))
                ke = 0.5 * float((rho * (to_numpy(ns.u[0].interior) ** 2 + to_numpy(ns.u[1].interior) ** 2)).mean())
                row = {"t": ns.clock.t, "u": u, "ke": ke}
                if case == "wave":
                    pos = to_numpy(position(f.interior, f.bc, grid, (0.0, 1.0)))
                    p = pos[pos < NODATA]
                    row["amp"] = float((p.max() - p.min()) / 2) if p.size else float("nan")
                hist.append(row); nxt += T / 40
            if ns.clock.i > 200000:
                failed = "step limit"; break
        return {"case": case, "variant": variant, "n": n, "ratio": ratio, "curv": curv, "mur": mur, "T": T, "steps": ns.clock.i,
                "t_end": ns.clock.t, "failed": failed, "umax": umax_run, "hist": hist,
                "df": float(np.abs(to_numpy(f.interior) - f_init).max())}

if __name__ == "__main__":
    J = []
    for variant in ("cons", "cons-filt", "plain", "plain-filt"):
        for n in (32, 64):
            for case in ("drop", "bubble"):
                for curv in ("numeric", "exact"):
                    J.append((case, variant, n, 1e3, curv, 55.0, 1.0))
            J.append(("wave", variant, n, 1e3, "numeric", 55.0, 2.0))
    for case in ("drop", "bubble"):
        for r in (1e1, 1e2):
            J.append((case, "cons", 64, r, "numeric", 55.0, 1.0))
        J.append((case, "cons", 64, 1e3, "numeric", 1.0, 1.0))
        J.append((case, "cons", 128, 1e3, "numeric", 55.0, 0.5))
        J.append((case, "cons", 128, 1e3, "exact", 55.0, 0.5))
    J.append(("wave", "cons", 128, 1e3, "numeric", 55.0, 1.0))
    J.sort(key=lambda a: a[2])
    out = sys.argv[1] if len(sys.argv) > 1 else "audit/c6/surface_tension.jsonl"
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()

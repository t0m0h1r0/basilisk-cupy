"""監修 C-6 の内容 2・3（必須の範囲：密度比 1000 まで）。ライブラリは main（c9bc278）、保存形 + filtered=False。
free : 一様流が保たれるか（薄い膜、接近した 2 液滴、小液滴、壁に接する界面）。
mom  : 壁と粘性のある x 運動量の収支（x 周期、y 壁）。圧力は x の壁の力を持たないので、残る外力は壁の粘性せん断だけ。
dam  : 閉じた箱のダム崩壊（重い液柱、密度比 1000、粘度比 55）。CFL 0.1 と 0.5 の比較。"""
import os, sys, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def setup(n, geom, heavy_in, mu, channel, noslip=False, lid=0.0, sigma=0.0, tol=1e-11):
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Dirichlet
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    h = 1.0 / n
    grid = CartesianGrid((n, n))
    kw = {}
    if channel:
        kw = dict(periodic=(0,), uf_normal={"bottom": 0.0, "top": 0.0})
        if noslip:
            kw["tangential"] = {"bottom": Dirichlet(0.0), "top": Dirichlet(lid)}
    else:
        kw = dict(periodic=(0, 1))
    ns = CenteredNS(grid, tolerance=tol, mu=mu or None, **kw)
    s = math.sin(0.3); c = math.cos(0.3)
    phi = {
        "film05": lambda x, y: 0.25 * h - np.abs(c * (y - 0.5) - s * (x - 0.5)),     # 傾いた膜、厚さ 0.5h
        "film15": lambda x, y: 0.75 * h - np.abs(c * (y - 0.5) - s * (x - 0.5)),
        "twodrops": lambda x, y: np.maximum(0.15 - np.hypot(x - 0.34, y - 0.5), 0.15 - np.hypot(x - 0.34 - 0.3 - 0.5 * h, y - 0.5 - 0.3 * h)),
        "tiny": lambda x, y: 1.3 * h - np.hypot(x - 0.5 - 0.21 * h, y - 0.5 - 0.37 * h),
        "walldrop": lambda x, y: 0.22 - np.hypot(x - 0.5, y - 0.013),
        "wallfilm": lambda x, y: 0.07 + 0.03 * np.sin(2 * np.pi * x) - y,
        "drop": lambda x, y: 0.2 - np.hypot(x - 0.5, y - 0.27),
    }[geom]
    f0 = to_numpy(fraction(grid, phi, refine=16))
    bc = ns.p.bc if channel else BoundaryConditions.periodic(2)
    r = (1.0, 1e-3) if heavy_in else (1e-3, 1.0)
    mu1 = mu if heavy_in else mu / 55
    mu2 = mu / 55 if heavy_in else mu
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=r[0], rho2=r[1], mu1=mu1, mu2=mu2, conserving=True, filtered=False)
    if sigma:
        tp.add_surface_tension(sigma)
    return grid, ns, tp, f0, to_numpy

def free(a):
    geom, n, heavy, cfl, mu = a
    channel = geom.startswith("wall")
    grid, ns, tp, f0, to_numpy = setup(n, geom, heavy, mu, channel)
    U = (1.0, 0.0) if channel else (1.0, 0.37)
    for k in range(2):
        ns.u[k].interior = np.full((n, n), U[k])
    ns.DT = cfl / n
    ns.start()
    du = 0.0; steps = int(round(1.0 * n / cfl)); r = None
    for s in range(steps):
        ns.step()
        du = max(du, max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2)))
        if not math.isfinite(du) or du > 10:
            break
    f = to_numpy(tp.f.interior)
    return "free %-8s N=%-3d %s cfl=%.2g mu=%-5g: max|u-U| %.1e  mass %.1e  f over %.1e  (step %d/%d)" % (
        geom, n, "heavy-in" if heavy else "light-in", cfl, mu, du, abs(f.sum() - f0.sum()) / f0.sum(), max(-f.min(), f.max() - 1), s + 1, steps)

def mom(a):
    geom, n, heavy, noslip, lid, sigma = a
    grid, ns, tp, f0, to_numpy = setup(n, geom, heavy, 1e-2, True, noslip=noslip, lid=lid, sigma=sigma)
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    # 中心差分で離散的に発散なし（流れ関数 psi = 0.1 sin(pi y) sin(2 pi x) の中心差分 + x によらないせん断）
    psi = lambda x, y: 0.1 * np.sin(np.pi * y) * np.sin(2 * np.pi * x)
    ns.u[0].interior = np.sin(np.pi * Y) + (psi(X, Y + h) - psi(X, Y - h)) / (2 * h)
    ns.u[1].interior = -(psi(X + h, Y) - psi(X - h, Y)) / (2 * h)
    from basilisk_cupy.linear.viscosity import viscosity
    store = {}
    class V:
        def solve(self, u, mu_, rho_, dt, tolerance=1e-3):
            out = viscosity(u, mu_, rho_, dt, tolerance=tolerance)
            m = to_numpy(mu_[1])
            u1 = to_numpy(u[0].interior)
            flux = 0.0
            if noslip:      # 壁の面の粘性流束（x 方向、長さ h を掛け、h/2 の差分で割ると 2 mu (U - u)）
                flux = float(np.sum(2 * m[:, 0] * (0.0 - u1[:, 0])) + np.sum(2 * m[:, -1] * (lid - u1[:, -1])))
            store["F"] = dt * flux
            return out
    ns.viscous = V()
    ns.DT = 0.25 / n
    ns.start()
    def P():
        f = np.clip(to_numpy(tp.f.interior), 0, 1)
        rho = f * (tp.rho1 - tp.rho2) + tp.rho2
        return float((rho * to_numpy(ns.u[0].interior)).sum()) * h * h, float((rho * np.abs(to_numpy(ns.u[0].interior))).sum()) * h * h
    p0, scale = P()
    prev = p0; res = 0.0; worst = 0.0; Fsum = 0.0
    steps = int(round(0.5 * n / 0.25))
    for s in range(steps):
        ns.step()
        p, _ = P()
        e = (p - prev) - store.get("F", 0.0)
        res += e; worst = max(worst, abs(e)); Fsum += store.get("F", 0.0); prev = p
    return "mom  %-8s N=%-3d %s %s lid=%g sigma=%g: P0 %.4e  dP %.4e  wall %.4e  cumulative residual %.1e (rel %.1e)  worst step %.1e" % (
        geom, n, "heavy-in" if heavy else "light-in", "noslip" if noslip else "slip", lid, sigma, p0, prev - p0, Fsum, res, abs(res) / scale, worst)

def dam(a):
    n, cfl, noslip = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.boundary import Dirichlet
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    grid = CartesianGrid((n, n))
    walls = ("left", "right", "bottom", "top")
    kw = {"tangential": {w: Dirichlet(0.0) for w in walls}} if noslip else {}
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-6, **kw)
    f0 = to_numpy(fraction(grid, lambda x, y: np.minimum(0.292 - x, 0.5833 - y), refine=16))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions()), rho1=1.0, rho2=1e-3, mu1=1e-4, mu2=1e-4 / 55, conserving=True, filtered=False)
    tp.add_reduced_gravity(G=(0.0, -1.0))
    ns.cfl = cfl; ns.DT = 0.01
    ns.start()
    umax = 0.0; out = []
    marks = [0.4, 0.8, 1.2, 1.6, 2.0]
    steps = 0
    while ns.clock.t < 2.0 and steps < 100000:
        ns.step(); steps += 1
        um = float(max(np.abs(to_numpy(ns.u[k].interior)).max() for k in range(2)))
        umax = max(umax, um)
        if not math.isfinite(um) or um > 50:
            out.append("BLOW-UP t=%.3f" % ns.clock.t); break
        if marks and ns.clock.t >= marks[0]:
            f = np.clip(to_numpy(tp.f.interior), 0, 1)
            col = np.where(f[:, 0] > 0.5)[0]
            out.append("t=%.1f front %.3f height %.3f" % (marks.pop(0), (col.max() + 1) / n if len(col) else 0, f[0, :].sum() / n))
    f = to_numpy(tp.f.interior)
    return "dam  N=%-3d cfl=%.2g %s: steps %d  max|u| %.2f  mass %.1e | %s" % (n, cfl, "noslip" if noslip else "slip", steps, umax, abs(f.sum() - f0.sum()) / f0.sum(), "; ".join(out))

def call(j):
    try:
        return j[0](j[1])
    except Exception as e:
        return "%s %s: EXCEPTION %r" % (j[0].__name__, j[1], e)

if __name__ == "__main__":
    J = []
    if "mom" in sys.argv[1:]:
        for geom in ("drop", "walldrop", "wallfilm"):
            for heavy in (True, False):
                for noslip, lid in ((False, 0.0), (True, 1.0)):
                    J.append((mom, (geom, 64, heavy, noslip, lid, 0.0)))
        J += [(mom, ("drop", n, True, True, 1.0, 0.0)) for n in (32, 128)]
        J += [(mom, ("drop", n, h_, False, 0.0, 0.1)) for n in (32, 64, 128) for h_ in (True, False)]
    else:
        for geom in ("film05", "film15", "twodrops", "tiny", "walldrop", "wallfilm"):
            for heavy in (True, False):
                for cfl, mu in ((0.5, 0.0), (0.1, 0.0), (0.5, 1e-2)):
                    J.append((free, (geom, 64, heavy, cfl, mu)))
        J += [(dam, (n, c, ns_)) for n in (64, 128) for c in (0.1, 0.5) for ns_ in (False, True)]
    with mp.get_context("spawn").Pool(14, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(call, J, chunksize=1):
            print(line, flush=True)

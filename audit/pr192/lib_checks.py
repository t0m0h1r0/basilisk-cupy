"""監修 PR #192：ライブラリの選択肢の独立の検査。実行は PR のツリー（PYTHONPATH に指定）で。
identity : 既定の挙動（main のツリーと配列を比べる。引数で出力ファイル名）
sweep    : 面に揃った帯 255 配置（監修の θ=1/3 の走査と同じ配置）を、ライブラリの選択肢で
mirror   : 滑り壁の箱 = 鏡像で 2 倍にした周期の計算の半分（D1 + D2 限定）
mom      : 壁と粘性の x 運動量の収支（発散なし／ありの初期速度、project_start）"""
import os, sys, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
R0 = dict(conserving_consistent=True, conserving_slope_limit=3.0)

def build(f0, ratio, n, periodic=True, kw=None, mu=0.0, shape=None, tol=1e-10, walls_x_periodic=False):
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    nx, ny = shape or (n, n)
    grid = CartesianGrid((nx, ny), length=(nx / n, ny / n))
    if periodic:
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=tol, mu=mu or None)
        bc = BoundaryConditions.periodic(2)
    elif walls_x_periodic:
        ns = CenteredNS(grid, periodic=(0,), uf_normal={"bottom": 0.0, "top": 0.0}, tolerance=tol, mu=mu or None)
        bc = ns.p.bc
    else:
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=tol, mu=mu or None)
        bc = BoundaryConditions()
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, bc), rho1=1.0, rho2=1.0 / ratio, mu1=mu, mu2=mu / 55, conserving=True, **(kw or {}))
    return grid, ns, tp

def sweep(a):
    cfl, off, ratio, U, n = a
    i = np.arange(n)[:, None] * np.ones((1, n))
    f0 = np.clip(np.minimum(i + 1, n // 2 + off) - np.maximum(i, n // 4 + off), 0, 1)
    grid, ns, tp = build(f0, ratio, n, kw=R0)
    ns.cfl = 0.5
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    ns.DT = cfl / n / max(abs(U[0]), abs(U[1]))
    ns.start()
    m = 0.0
    for s in range(600):
        ns.step()
        e = max(float(np.abs(ns.u[k].interior - U[k]).max()) for k in range(2))
        m = max(m, e)
        if not math.isfinite(e) or e > 10:
            break
    return (a, m, tp.rho_floor_hits)

def identity(path):
    out = {}
    n = 32
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy import CartesianGrid
    f0 = np.asarray(fraction(CartesianGrid((n, n)), lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.4) ** 2))
    for name, periodic, mu in (("per", True, 1e-2), ("box", False, 1e-2)):
        grid, ns, tp = build(f0, 1e3, n, periodic=periodic, mu=mu)
        tp.add_surface_tension(0.1)
        ns.u[0].interior = np.sin(np.pi * X) * np.cos(np.pi * Y); ns.u[1].interior = -np.cos(np.pi * X) * np.sin(np.pi * Y)
        ns.DT = 0.25 / n
        ns.start()
        for s in range(40):
            ns.step()
        out[name] = np.stack([ns.u[0].interior, ns.u[1].interior, tp.f.interior, ns.p.interior])
    np.savez(path, **out)

def mirror(a):
    ratio, mu, heavy = a
    n = 32
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy import CartesianGrid
    phi = lambda x, y: np.maximum(0.2 - np.hypot(x - 0.31, y - 0.12), 0.11 - np.hypot(x - 0.9, y - 0.93))
    f0 = np.asarray(fraction(CartesianGrid((n, n)), phi))
    if not heavy:
        f0 = 1 - f0
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(np.pi * X) * np.cos(np.pi * Y); v0 = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    res = []
    for periodic in (False, True):
        if periodic:   # 2n x 2n、鏡像
            F = np.block([[f0, f0[:, ::-1]], [f0[::-1, :], f0[::-1, ::-1]]])
            U0 = np.block([[u0, u0[:, ::-1]], [-u0[::-1, :], -u0[::-1, ::-1]]])
            V0 = np.block([[v0, -v0[:, ::-1]], [v0[::-1, :], -v0[::-1, ::-1]]])
            grid, ns, tp = build(F, ratio, n, periodic=True, kw=R0, mu=mu, shape=(2 * n, 2 * n), tol=1e-12)
            ns.u[0].interior = U0; ns.u[1].interior = V0
        else:
            grid, ns, tp = build(f0, ratio, n, periodic=False, kw=R0, mu=mu, tol=1e-12)
            ns.u[0].interior = u0; ns.u[1].interior = v0
        ns.DT = 0.2 / n
        ns.start()
        for s in range(60):
            ns.step()
        res.append((ns.u[0].interior[:n, :n].copy(), ns.u[1].interior[:n, :n].copy(), tp.f.interior[:n, :n].copy()))
    d = [float(np.abs(res[0][k] - res[1][k]).max()) for k in range(3)]
    return "mirror ratio=%g mu=%g heavy=%d: |du| %.1e |dv| %.1e |df| %.1e" % (ratio, mu, heavy, *d)

def mom(a):
    kw_name, divfree, proj, noslip = a
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy import CartesianGrid, BoundaryConditions, CellField
    from basilisk_cupy.boundary import Dirichlet
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    n = 64; h = 1.0 / n
    grid = CartesianGrid((n, n))
    tg = {"tangential": {"bottom": Dirichlet(0.0), "top": Dirichlet(1.0)}} if noslip else {}
    ns = CenteredNS(grid, periodic=(0,), uf_normal={"bottom": 0.0, "top": 0.0}, tolerance=1e-11, mu=1e-2, **tg)
    ns.project_start = proj
    f0 = np.asarray(fraction(grid, lambda x, y: 0.2 - np.hypot(x - 0.5, y - 0.27), refine=16))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, ns.p.bc), rho1=1.0, rho2=1e-3, mu1=1e-2, mu2=1e-2 / 55, conserving=True,
                  **(R0 if kw_name == "r0" else {}))
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    psi = lambda x, y: 0.1 * np.sin(np.pi * y) * np.sin(2 * np.pi * x)
    if divfree:
        ns.u[0].interior = np.sin(np.pi * Y) + (psi(X, Y + h) - psi(X, Y - h)) / (2 * h)
        ns.u[1].interior = -(psi(X + h, Y) - psi(X - h, Y)) / (2 * h)
    else:
        ns.u[0].interior = np.sin(np.pi * Y) * (1.0 + 0.5 * np.sin(2 * np.pi * X))
        ns.u[1].interior = 0.3 * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    from basilisk_cupy.linear.viscosity import viscosity
    st = {"F": 0.0}
    class V:
        def solve(self, u, mu_, rho_, dt, tolerance=1e-3):
            out = viscosity(u, mu_, rho_, dt, tolerance=tolerance)
            if noslip:
                m = np.asarray(mu_[1]); u1 = np.asarray(u[0].interior)
                st["F"] += dt * float(np.sum(2 * m[:, 0] * (0 - u1[:, 0])) + np.sum(2 * m[:, -1] * (1.0 - u1[:, -1])))
            return out
    ns.viscous = V()
    ns.DT = 0.25 / n
    ns.start()
    def P():
        f = tp.f.interior
        r = (f if kw_name == "r0" else np.clip(f, 0, 1)) * (1 - 1e-3) + 1e-3
        return float((r * ns.u[0].interior).sum()) * h * h, float((np.clip(f, 0, 1) * (1 - 1e-3) + 1e-3) @ np.ones(n) @ np.ones(n)) if False else 0.0
    def Pc():
        r = np.clip(tp.f.interior, 0, 1) * (1 - 1e-3) + 1e-3
        return float((r * ns.u[0].interior).sum()) * h * h
    p0, pc0 = P()[0], Pc()
    scale = float((np.abs((np.clip(tp.f.interior, 0, 1) * (1 - 1e-3) + 1e-3) * ns.u[0].interior)).sum()) * h * h
    for s in range(128):
        ns.step()
    return "mom %-4s divfree=%d project_start=%d noslip=%d: residual/scale (solver rho) %.1e   (clamped rho) %.1e  floor hits %d  f range [%.1e, 1%+.1e]" % (
        kw_name, divfree, proj, noslip, abs(P()[0] - p0 - st["F"]) / scale, abs(Pc() - pc0 - st["F"]) / scale, tp.rho_floor_hits, tp.f.interior.min(), tp.f.interior.max() - 1)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "identity":
        identity(sys.argv[2])
    else:
        with mp.get_context("spawn").Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
            if mode == "mirror":
                for line in pool.imap(mirror, [(r, mu, h) for r in (1e3,) for mu in (0.0, 1e-2) for h in (True, False)]):
                    print(line, flush=True)
            elif mode == "mom":
                for line in pool.imap(mom, [(k, d, p, ns_) for k in ("base", "r0") for d, p in ((1, 0), (0, 0), (0, 1)) for ns_ in (0, 1)]):
                    print(line, flush=True)
            else:
                cfls = [0.26, 0.3, 0.33, 0.36, 0.38, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.495, 0.498, 0.499, 0.4999]
                J = [(c, o, r, (1.0, 0.0), 32) for c in cfls for o in (0.0, 0.05, 0.13, 0.25, 0.4, 0.5) for r in (1e3, 1e6)]
                J += [(c, o, r, (1.0, 0.37), 32) for c in (0.4, 0.45, 0.49) for o in (0.0, 0.13, 0.4) for r in (1e3, 1e4, 1e6)]
                R = list(pool.imap_unordered(sweep, J, chunksize=1))
                for r in (1e3, 1e4, 1e6):
                    sub = [x for x in R if x[0][2] == r]
                    w = max(sub, key=lambda x: x[1])
                    print("sweep ratio %g: %d cases, worst %.2e at %s, floor hits total %d" % (r, len(sub), w[1], w[0], sum(x[2] for x in sub)))

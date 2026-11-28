"""監修（C）：ノート115 の総運動量と静水圧を、A が測っていない条件で確かめる。
momentum: 周期、渦の中の液滴。filtered=True（two-phase.h の FILTERED）、粘性あり（密度・粘性の跳び）、密度比 1e6。
hydro   : 別の角度（33°・60°）、密度比 1e6、filtered=True。
curl    : plain が釣り合わない理由。静止の条件は α_f (Δp)_f / h = g·e_f、つまり ρ_f g·e_f が離散の勾配であること。
          内部の頂点まわりの循環を直接計算する（0 でなければ、どんな p でも釣り合わない）。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def momentum(a):
    from basilisk_cupy import CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases.r0p2_pressure_momentum import _bc
    _, ratio, filtered, mu, n, T = a
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=(mu if mu else None))
    f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.3) ** 2, refine=8))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, _bc((0, 1))), rho1=1.0, rho2=1.0 / ratio, conserving=True,
                  filtered=filtered, mu1=mu, mu2=mu / 55.0)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    mom = lambda: np.array([float((to_numpy(tp.rho(tp.f.interior)) * to_numpy(ns.u[k].interior)).sum()) for k in range(2)])
    m0 = mom(); ns.start(events=[T]); worst = 0.0; failed = ""
    while ns.clock.t < T - 1e-12 and ns.clock.i < 40 * n:
        ns.step()
        d = float(np.hypot(*(mom() - m0)) / np.hypot(*m0))
        if not math.isfinite(d):
            failed = "nan"; break
        worst = max(worst, d)
    return "momentum ratio=%-7g filtered=%-5s mu=%-6g n=%d t=%g: max relative change %.2e %s" % (ratio, filtered, mu, n, ns.clock.t, worst, failed)

def hydro(a):
    from cases.r0p2_pressure_momentum import hydrostatic
    import cases.r0p2_pressure_momentum as m
    _, kind, th, grav, n, ratio, filtered = a
    if filtered:
        from basilisk_cupy.interface.two_phase import TwoPhase
        m._two_phase = lambda ns, f, k, r, c: TwoPhase(ns, f, rho1=1.0, rho2=1.0 / r, conserving=c, filtered=True)
    o = hydrostatic(kind, th, grav, n, ratio=ratio, t_end=1.0, samples=(0.1, 1.0))
    h = o["history"]
    return "hydro %-6s theta=%-5g %-7s n=%d ratio=%-7g filtered=%-5s: %s %s" % (kind, th, grav, n, ratio, filtered,
        " ".join("t=%g cells %.1e faces %.1e" % (r["t"], r["cells"], r["faces"]) for r in h), o["failed"] or "")

def run(a):
    return (momentum if a[0] == "m" else hydro)(a)

def curl():
    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    for n in (32, 128):
        for th in (0.0, 10.0, 26.565, 45.0):
            t = math.radians(th); nx, ny = math.sin(t), math.cos(t)
            f = to_numpy(fraction(CartesianGrid((n, n)), lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5137))))
            rho = lambda c: c + (1 - c) * 1e-3
            qx = rho(0.5 * (f[1:] + f[:-1])) * (-nx)          # 内部の x の面
            qy = rho(0.5 * (f[:, 1:] + f[:, :-1])) * (-ny)    # 内部の y の面
            c = (qy[1:] - qy[:-1]) - (qx[:, 1:] - qx[:, :-1])  # 内部の頂点の循環（x h）
            print("curl smooth n=%3d theta=%6.3f: max |circulation| = %.3e,  nonzero vertices %d" % (n, th, np.abs(c).max(), int((np.abs(c) > 1e-12).sum())))

if __name__ == "__main__":
    curl()
    J = []
    J += [("h", k, th, g, n, r, False) for k in ("smooth", "mean") for th in (33.0, 60.0) for g in ("reduced", "plain") for n in (32, 64) for r in (1e3, 1e6)]
    J += [("h", "smooth", th, "reduced", n, 1e3, True) for th in (26.565, 33.0) for n in (32, 64)]
    with mp.get_context("spawn").Pool(14, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(run, J, chunksize=1):
            print(line, flush=True)

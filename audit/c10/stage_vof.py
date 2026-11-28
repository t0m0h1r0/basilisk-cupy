"""監修 C-10：保存型の中点法・RK2 の段の作り方と、VOF の有界性・界面の厚み・自由流（輸送だけ、密度比 1000、周期）。

既存の保存形の輸送 TwoPhase._vof を部品にして、段の作り方を 3 通り比べる。速度（面）は与える。
 M: (f,u)^{n+1} = A_dt[uf^{n+1/2}] (f,u)^n          … f^n から中点の速度で運ぶ（輸送は現行と同じ形）
 R: 中間 (f,u)^h = A_{dt/2}[uf^n](f,u)^n を作り、流束（増分）を (f,u)^h から取って (f,u)^n に加える
 H: Heun：(f,u)^{**} = A_dt[uf^{n+1}] A_dt[uf^n] (f,u)^n、質量と運動量を (n と **) で平均
運ぶセル速度は最初に一様 U（自由流の検査。面の速度とは独立に、一様のまま保たれるべき）。"""
import os, math, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
U = (1.0, 0.37)

def run(a):
    case, form, n, cfl = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    h = 1.0 / n
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1))
    phi = {"translate": lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2,
           "vortex": lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.75) ** 2,
           "band": lambda x, y: 0.25 - np.abs(y - 0.5)}[case]
    f0 = to_numpy(fraction(grid, phi, refine=16))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True)
    ns.u[0].interior = np.full((n, n), U[0]); ns.u[1].interior = np.full((n, n), U[1])
    ns.start()
    xv = np.arange(n + 1) * h
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    T = 2.0
    def uf_at(t):
        if case == "vortex":                     # Rider-Kothe の渦（t = T で元に戻る）。流れ関数の差分で離散的に発散なし
            psi = np.sin(np.pi * X) ** 2 * np.sin(np.pi * Y) ** 2 / np.pi * math.cos(math.pi * t / T)
            return -(psi[:, 1:] - psi[:, :-1]) / h, (psi[1:, :] - psi[:-1, :]) / h
        V = (1.0, 0.5) if case == "translate" else (0.0, 0.5)     # t = 2 で周期の整数倍だけ動く
        return np.full((n + 1, n), V[0]), np.full((n, n + 1), V[1])
    umax = max(np.abs(c).max() for c in uf_at(0.0))
    steps = int(math.ceil(T / (cfl * h / umax)))
    dt = T / steps
    def rho(f):
        return f * (1.0 - 1e-3) + 1e-3
    def adv(f, u, t, d, i):
        tp.f.interior = f; tp.f.fill_ghosts()
        for k in range(2):
            ns.u[k].interior = u[k]
        ns.u.fill_ghosts()
        for k, w in enumerate(uf_at(t)):
            ns.uf[k] = w
        ns.clock.dt = d; ns.clock.i = i
        tp._vof(ns)
        return to_numpy(tp.f.interior).copy(), [to_numpy(ns.u[k].interior).copy() for k in range(2)]
    f, u = f0.copy(), [np.full((n, n), U[0]), np.full((n, n), U[1])]
    lo, hi, du = 0.0, 1.0, 0.0
    mixed0 = int(((f0 > 1e-6) & (f0 < 1 - 1e-6)).sum())
    fail = ""
    for s in range(steps):
        t = s * dt
        if form == "M":
            f, u = adv(f, u, t + dt / 2, dt, s)
        elif form == "R":
            fh, uh = adv(f, u, t, dt / 2, s)
            fs, us = adv(fh, uh, t + dt / 2, dt, s)
            fn = f + (fs - fh)
            m = [rho(f) * u[k] + rho(fs) * us[k] - rho(fh) * uh[k] for k in range(2)]
            if rho(fn).min() <= 0:
                fail = "rho<=0 at step %d (min f %.2e)" % (s + 1, fn.min())
                lo = min(lo, float(fn.min()))
                break
            f, u = fn, [m[k] / rho(fn) for k in range(2)]
        else:
            f1, u1 = adv(f, u, t, dt, s)
            f2, u2 = adv(f1, u1, t + dt, dt, s + 1)
            fn = 0.5 * (f + f2)
            m = [0.5 * (rho(f) * u[k] + rho(f2) * u2[k]) for k in range(2)]
            f, u = fn, [m[k] / rho(fn) for k in range(2)]
        lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
        du = max(du, max(float(np.abs(u[k] - U[k]).max()) for k in range(2)))
        if not math.isfinite(du) or du > 1e3:
            fail = "blow-up at step %d" % (s + 1)
            break
    mixed = int(((f > 1e-6) & (f < 1 - 1e-6)).sum())
    thick = float(np.minimum(np.clip(f, 0, 1), 1 - np.clip(f, 0, 1)).sum()) / max(float(np.minimum(f0, 1 - f0).sum()), h)
    return ("%-9s %s N=%-3d cfl=%.2g: f over [%.2e, 1%+.2e]  mass %.1e  free-stream %.1e  shape L1 %.3e  mixed cells %d->%d  thickness x%.3g %s"
            % (case, form, n, cfl, lo, hi - 1, abs(f.sum() - f0.sum()) / f0.sum(), du, np.abs(f - f0).mean(), mixed0, mixed, thick, fail))

if __name__ == "__main__":
    J = [(c, m, n, cfl) for c in ("translate", "vortex", "band") for n in (64, 128) for cfl in (0.4, 0.1) for m in ("M", "R", "H")]
    with mp.get_context("spawn").Pool(16, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

"""監修（C-2）：E3（Crank–Nicolson）が二相で崩れる機構の独立の確認。陽的な半分と陰的な半分で、粘性の作用素（μ・ρ の
時刻）が違うことが原因なら、両方の半分を同じ（輸送の後の）μ・ρ で作れば安定になるはず。A のフックを使わず自分で作る。
  be     : ライブラリ（後退 Euler）
  cn-old : 陽的な半分を、輸送の前の u^n・μ^n・ρ^n で作って後で足す（ノート113 の形）
  cn-same: 陽的な半分も、粘性の段で受け取った μ・ρ と、その時点の速度で作る"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    form, ratio, mur, n, steps, shape = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    grid = CartesianGrid((n, n))
    mu1 = 1e-2; mu2 = mu1 / mur
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=mu1)
    if shape == "drop":
        f0 = to_numpy(fraction(grid, lambda x, y: 0.15 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
    else:
        i = np.arange(n)[:, None] * np.ones((1, n)); f0 = np.clip(np.minimum(i + 1, n // 2 + 0.3) - np.maximum(i, n // 4 + 0.3), 0, 1)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, mu1=mu1, mu2=mu2, conserving=True)
    U = (1.0, 0.37)
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    st = {}
    def lu(mu, rho, dt):
        ns.u.fill_ghosts()
        prob = ViscosityProblem.reuse(ns.u, mu, rho, dt)
        res = [np.zeros(grid.n) for _ in range(2)]
        prob.residual(list(ns.u), [np.zeros(grid.n) for _ in range(2)], res)
        return [(r + c.interior) / dt for r, c in zip(res, ns.u.components)]
    class CN:
        def solve(self, u, mu, rho, dt, tolerance=1e-3):
            acc = st["old"] if form == "cn-old" else lu(mu, rho, dt)
            for c, x in zip(u.components, acc):
                c.interior = c.interior + 0.5 * dt * x
            u.fill_ghosts()
            return viscosity(u, mu, rho, 0.5 * dt, tolerance=tolerance)
    if form != "be":
        ns.viscous = CN()
    if form == "cn-old":
        ns.on("vof", lambda s: st.__setitem__("old", lu(s.mu, s.rho, s.clock.dt)))
    ns.start()
    m = 0.0
    for s in range(steps):
        ns.step()
        e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
        m = max(m, e)
        if not math.isfinite(e) or e > 10:
            break
    return "%-8s ratio=%-6g mu-ratio=%-4g N=%d %-6s: max|u-U| %.2e (%d steps)" % (form, ratio, mur, n, shape, m, s + 1)

if __name__ == "__main__":
    J = [(f, r, m, 32, 200, sh) for f in ("be", "cn-old", "cn-same") for r, m in ((1e3, 55), (1e3, 1), (10, 55), (1.0, 55)) for sh in ("drop", "column")]
    J += [(f, 1e3, 55, 64, 400, "drop") for f in ("be", "cn-old", "cn-same")]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

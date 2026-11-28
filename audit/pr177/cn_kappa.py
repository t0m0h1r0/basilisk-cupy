"""監修（C-2）：E3 の自由流の発散が「二相だから」か「粘性の硬さ κ = ν Δt/h² が大きいから」かを分ける。
単相（f = 0、密度一定）の一様流 + 丸めの大きさの擾乱で、ν を変えて cn-old（ノート113 の形）を走らせる。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    form, nu, n, noise, steps = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=nu)
    TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0, mu1=nu, mu2=nu, conserving=True)
    U = (1.0, 0.37)
    rng = np.random.default_rng(0)
    for k in range(2):
        ns.u[k].interior = U[k] + noise * rng.standard_normal(grid.n)
    st = {}
    def lu(mu, rho, dt):
        ns.u.fill_ghosts()
        prob = ViscosityProblem.reuse(ns.u, mu, rho, dt)
        res = [np.zeros(grid.n) for _ in range(2)]
        prob.residual(list(ns.u), [np.zeros(grid.n) for _ in range(2)], res)
        return [(r + c.interior) / dt for r, c in zip(res, ns.u.components)]
    class CN:
        def solve(self, u, mu, rho, dt, tolerance=1e-3):
            for c, x in zip(u.components, st["old"]):
                c.interior = c.interior + 0.5 * dt * x
            u.fill_ghosts()
            return viscosity(u, mu, rho, 0.5 * dt, tolerance=tolerance)
    if form == "cn-old":
        ns.viscous = CN()
        ns.on("vof", lambda s: st.__setitem__("old", lu(s.mu, s.rho, s.clock.dt)))
    ns.start()
    m, kap = 0.0, 0.0
    for s in range(steps):
        ns.step()
        kap = max(kap, nu * ns.clock.dt * n * n)
        e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
        m = e
        if not math.isfinite(e) or e > 10:
            break
    return "%-6s single phase nu=%-6g N=%d noise=%g kappa=%.3g: |u-U| after %d steps %.2e" % (form, nu, n, noise, kap, s + 1, m)

if __name__ == "__main__":
    J = [(f, nu, 32, 1e-8, 200) for f in ("be", "cn-old") for nu in (0.01, 0.2, 1.0, 10.0)]
    J += [("cn-old", nu, 64, 1e-8, 400) for nu in (0.2, 10.0)]
    with mp.get_context("spawn").Pool(6, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

"""監修 C-9 の 1：陽的な粘性の半分（E3）が密度比 1000 で崩れるのに、界面の移動は要るか。
静止した二相（u = 0 + 丸めより大きい雑音）と、一様流で、後退 Euler・CN（陽の半分は u^n）を比べる。θ を変えた形も。
さらに、粘性の段の中の圧力の扱いを変える："nog"：陰的な段に u* だけを入れる（dt g を足して引くのをやめる）。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    U, theta, nog, ratio, mur, frozen, steps, shape, cfl = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    n = 32
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-11, mu=1e-2)
    phi = {"circle": lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2,
           "slab": lambda x, y: 0.2 - np.abs(y - 0.51)}[shape]
    f0 = to_numpy(fraction(grid, phi, refine=8))
    TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, mu1=1e-2, mu2=1e-2 / mur, conserving=True)
    rng = np.random.default_rng(0)
    for k in range(2):
        ns.u[k].interior = U[k] + 1e-8 * rng.standard_normal((n, n))
    st = {}
    def lu(solver, dt):
        solver.u.fill_ghosts()
        prob = ViscosityProblem.reuse(solver.u, solver.mu, solver.rho, dt)
        res = [np.zeros(grid.n) for _ in range(2)]
        prob.residual(list(solver.u), [np.zeros(grid.n) for _ in range(2)], res)
        return [(r + c.interior) / dt for r, c in zip(res, solver.u.components)]
    class Theta:
        def solve(self, u, mu, rho, dt, tolerance=1e-3):
            g = [c.interior.copy() for c in ns.g.components]
            if nog:                                    # 受け取った u* + dt g から dt g を外してから解き、後で戻す
                for c, gg in zip(u.components, g):
                    c.interior = c.interior - dt * gg
            for c, x in zip(u.components, st["Lun"]):
                c.interior = c.interior + (1 - theta) * dt * x
            u.fill_ghosts()
            out = viscosity(u, mu, rho, theta * dt, tolerance=tolerance)
            if nog:
                for c, gg in zip(u.components, g):
                    c.interior = c.interior + dt * gg
                u.fill_ghosts()
            return out
    if theta < 1.0 or nog:
        ns.viscous = Theta()
    ns.on("vof", lambda s: st.__setitem__("Lun", lu(s, s.clock.dt)))
    if frozen:                                         # 界面を動かさない（輸送を止める）
        ns._hooks["vof"] = [h for h in ns._hooks["vof"] if getattr(h, "__self__", None) is None]
    ns.DT = cfl / n
    ns.start()
    e0 = None; e = 0.0
    for s in range(steps):
        ns.step()
        e = max(float(np.abs(to_numpy(ns.u[k].interior) - U[k]).max()) for k in range(2))
        if s == 9:
            e0 = e
        if not math.isfinite(e) or e > 10:
            break
    return "%-6s dt=%.3g/n U=%-10s theta=%-4g %-4s ratio=%-5g mur=%-3g %-6s: |u-U| step10 %.1e  step %d %.1e" % (shape, cfl, U, theta, "nog" if nog else "lib", ratio, mur, "frozen" if frozen else "moving", e0 or float("nan"), s + 1, e)

if __name__ == "__main__":
    J = []
    for theta in (1.0, 0.5):
        J.append(((0.0, 0.0), theta, False, 1e3, 55.0, False, 300, "circle", 0.4))
        for U in ((1.0, 0.0), (0.0, 0.37), (1.0, 0.37)):
            J.append((U, theta, False, 1e3, 55.0, False, 300, "slab", 0.4))
    for cfl in (0.2, 0.05, 0.01):
        J.append(((1.0, 0.37), 0.5, False, 1e3, 55.0, False, int(300 * 0.4 / cfl) if cfl > 0.02 else 4000, "circle", cfl))
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

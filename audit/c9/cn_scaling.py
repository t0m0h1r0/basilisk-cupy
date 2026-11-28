"""監修 C-9 の 1：E3 の崩れを決めるのは Courant 数か κ=μΔt/(ρ h²) か。
円形の滴、一様流 U=(1,0.37)、密度比 1000。陽の半分の作り方 old（旧い ρ、μ で輸送後の u）と same（新しい ρ、μ で u*）。
最初に |u-U| が 1e-5 を超えたときの最大点の f も記録する。物理時刻 T まで。"""
import os, math, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")

def run(a):
    form, theta, mus, cfl, ratio, T = a
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.fractions import fraction
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.linear.viscosity import ViscosityProblem, viscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    n = 32
    U = (1.0, 0.37)
    grid = CartesianGrid((n, n))
    mu = 1e-2 * mus
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-11, mu=mu)
    f0 = to_numpy(fraction(grid, lambda x, y: 0.2 ** 2 - (x - 0.5) ** 2 - (y - 0.5) ** 2, refine=8))
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio, mu1=mu, mu2=mu / 55, conserving=True)
    rng = np.random.default_rng(0)
    for k in range(2):
        ns.u[k].interior = U[k] + 1e-8 * rng.standard_normal((n, n))
    st = {}
    def lu(solver, u, mu_, rho_, dt):
        u.fill_ghosts()
        prob = ViscosityProblem.reuse(u, mu_, rho_, dt)
        res = [np.zeros(grid.n) for _ in range(2)]
        prob.residual(list(u), [np.zeros(grid.n) for _ in range(2)], res)
        return [(r + c.interior) / dt for r, c in zip(res, u.components)]
    class Theta:
        def solve(self, u, mu_, rho_, dt, tolerance=1e-3):
            L = st["Lun"] if form == "old" else lu(ns, u, mu_, rho_, dt)
            for c, x in zip(u.components, L):
                c.interior = c.interior + (1 - theta) * dt * x
            u.fill_ghosts()
            return viscosity(u, mu_, rho_, theta * dt, tolerance=tolerance)
    if theta < 1.0:
        ns.viscous = Theta()
        if form == "old":
            ns.on("vof", lambda s: st.__setitem__("Lun", lu(s, s.u, s.mu, s.rho, s.clock.dt)))
    ns.DT = cfl / n
    ns.start()
    steps = int(round(T * n / cfl))
    first = None
    for s in range(steps):
        ns.step()
        d = np.maximum(*[np.abs(to_numpy(ns.u[k].interior) - U[k]) for k in range(2)])
        e = float(d.max())
        if first is None and e > 1e-5:
            i = np.unravel_index(np.argmax(d), d.shape)
            f = to_numpy(tp.f.interior)
            first = "onset t=%.3f f=%.2e (nbr f %.2f..%.2f)" % ((s + 1) * cfl / n, f[i],
                     min(np.roll(f, k, ax)[i] for ax in (0, 1) for k in (1, -1)), max(np.roll(f, k, ax)[i] for ax in (0, 1) for k in (1, -1)))
        if not math.isfinite(e) or e > 10:
            break
    kl = (mu / 55) * (cfl / n) * n * n * ratio
    return "%-4s theta=%.2g mu=%-6g dt=%-5g/n ratio=%-5g kappa_light=%-7.3g : t=%.2f/%g err %.1e  %s" % (form, theta, mu, cfl, ratio, kl, (s + 1) * cfl / n, T, e, first or "")

if __name__ == "__main__":
    J = []
    for form in ("old", "same"):
        for mus in (1.0, 0.125, 0.015625):
            J.append((form, 0.5, mus, 0.4, 1e3, 2.0))
        for cfl, mus in ((0.1, 1.0), (0.1, 4.0), (0.05, 8.0)):
            J.append((form, 0.5, mus, cfl, 1e3, 2.0))
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)

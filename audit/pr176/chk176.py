"""監修：PR #176 の確認。(1) A の参照解と C-3、(2) A の参照解のセル平均を直接の求積と、(3) 現行のライブラリの誤差を
C-3 の初期値で独立に再現し、CFL を下げて 1 次の出どころ（時間か空間か）を分ける。(4) 空間の 1 次の候補：界面の面の粘性が
算術平均であること（調和平均に差し替える）。"""
import os, math, warnings, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
warnings.filterwarnings("ignore")
RHO, MU, U = (1.0, 1e-3), (1e-2, 1e-2 / 55.0), (1.0, 0.37)

def refs(n, boost):
    from cases.r0_shear import Shear
    import audit.c3ref as c3
    U_ = U if boost else (0.0, 0.0)
    A = Shear(n=n, U=U_)
    C = c3.Shear(n=n, U=U_, alpha=0.25, beta=0.75, rho=RHO, mu=MU)
    z = np.linspace(0, 1, 997, endpoint=False)
    c = np.zeros(len(C.lam))
    for k, amp in ((1, 1.0), (2, 0.3)):
        e = np.zeros(len(C.lam)); e[k] = 1.0
        phi = C.w_at(z, e, 0.0); W = A._W(A.modes[k - 1], (z - 0.25) % 1.0)
        c[k] = amp * float(phi @ W / (phi @ phi))
    return A, C, c

def run(a):
    n, boost, N, cfl, mu_mode, T = a
    A, C, c = refs(n, boost)
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    grid = CartesianGrid((N, N))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10, mu=max(MU))
    cls = TwoPhase
    if mu_mode == "harmonic":
        class cls(TwoPhase):
            def mu(self, f):
                xp = self.f.grid.xp
                fc = xp.minimum(xp.maximum(f, 0.0), 1.0)
                return 1.0 / (fc / self.mu1 + (1.0 - fc) / self.mu2)
    f0 = np.clip(C.fraction_cells(N, 0.0), 0, 1)
    cls(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=RHO[0], rho2=RHO[1], mu1=MU[0], mu2=MU[1], conserving=True)
    ns.cfl = cfl
    u0, v0 = C.velocity_cells(N, c, 0.0)
    ns.u[0].interior = u0; ns.u[1].interior = v0
    if boost is False:
        ns.DT = cfl / N / 1.0 * 0.5
    ns.start(events=[T])
    ux, uy = A.face_fluxes(N, 0.0)
    ns.uf.components[0][...] = ux; ns.uf.components[1][...] = uy
    while ns.clock.t < T - 1e-12:
        ns.step()
    ue, ve = C.velocity_cells(N, c, ns.clock.t)
    e = np.hypot(to_numpy(ns.u[0].interior) - ue, to_numpy(ns.u[1].interior) - ve)
    return a, float(np.sqrt((e * e).mean())), ns.clock.i

if __name__ == "__main__":
    from cases.r0_shear import compare_c3
    os.environ["C3_REFERENCE"] = "audit/c3ref.py"
    for r in compare_c3():
        print("compare n=%s lam %.1e shape %.1e cells %.1e frac %.1e" % (r["n"], max(r["lam_rel_diff"]), max(r["mode_shape_diff"]), r["cells_diff"], r["fraction_diff"]))
    for n in ((0, 1), (1, 2)):
        A, C, c = refs(n, True)
        N, m = 8, 600
        x = (np.arange(N * m) + 0.5) / (N * m)
        w = A.w_cells(N, 0.3)
        i, j = 2, 3
        X, Y = np.meshgrid(x[i * m:(i + 1) * m], x[j * m:(j + 1) * m], indexing="ij")
        print("A cell average n=%s vs midpoint quadrature: %.1e" % (n, abs(w[i, j] - A.w_at(n[0] * (X - U[0] * 0.3) + n[1] * (Y - U[1] * 0.3), 0.3).mean())))
    J = [(n, b, N, cfl, mm, 0.5) for n, b in (((0, 1), False), ((0, 1), True), ((1, 1), True)) for N in (32, 64, 128) for cfl, mm in ((0.5, "arith"), (0.1, "arith"), (0.5, "harmonic"), (0.1, "harmonic"))]
    with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
        R = {a: (e, s) for a, e, s in pool.imap_unordered(run, J, chunksize=1)}
    for n, b in (((0, 1), False), ((0, 1), True), ((1, 1), True)):
        for cfl, mm in ((0.5, "arith"), (0.1, "arith"), (0.5, "harmonic"), (0.1, "harmonic")):
            es = [R[(n, b, N, cfl, mm, 0.5)][0] for N in (32, 64, 128)]
            print("n=%s boost=%-5s cfl=%.2f mu=%-8s L2 %s  ratios %.2f %.2f  steps %s" % (n, b, cfl, mm, " ".join("%.3e" % v for v in es), es[0] / es[1], es[1] / es[2], [R[(n, b, N, cfl, mm, 0.5)][1] for N in (32, 64, 128)]), flush=True)

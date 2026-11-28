import sys, math, numpy as np, warnings
warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS
N, k = 64, 4 * math.pi; h = 1.0 / N
lo, hi = 0.25 + (1 - 0.033) * h, 0.75 + (1 - 0.033) * h
yl = np.arange(N) * h
f0 = np.tile(np.clip((np.minimum(yl + h, hi) - np.maximum(yl, lo)) / h, 0, 1), (N, 1))
ux = 0.5 * (np.cos(k * (yl - 0.25)) - np.cos(k * (yl + h - 0.25))) / (k * h)
def loadC(tag, it):
    d = np.loadtxt("%s_%d.txt" % (tag, it)); i, j = d[:, 0].astype(int), d[:, 1].astype(int)
    out = []
    for col in range(2, 8):
        a = np.empty((N, N)); a[i, j] = d[:, col]; out.append(a)
    return out  # ux uy f p gx gy
def loadCuf(tag, it):
    fx = np.zeros((N + 1, N)); fy = np.zeros((N, N + 1))
    for line in open("%s_%d_uf.txt" % (tag, it)):
        a, i, j, v = line.split(); i, j = int(i), int(j)
        if a == "x": fx[i, j] = float(v)
        else: fy[i, j] = float(v)
    return fx[:N], fy[:, :N]
kw = dict(conserving_stale_ghosts=True, conserving_vof_half=True)
grid = CartesianGrid((N, N)); ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-12)
tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1e-3, conserving=True, **kw)
ns.cfl = 0.5; ns.DT = 0.007
ns.u[0].interior = np.tile(ux, (N, 1)); ns.u[1].interior = np.ones((N, N))
snap = {}
def at(tag):
    def hook(s):
        snap[(tag, s.clock.i)] = [s.u[0].interior.copy(), s.u[1].interior.copy(), tp.f.interior.copy(), s.p.interior.copy(), s.g[0].interior.copy(), s.g[1].interior.copy()]
        snap[(tag + "_uf", s.clock.i)] = [s.uf.components[0][:N].copy(), s.uf.components[1][:, :N].copy()]
        snap[(tag + "_dt", s.clock.i)] = s.clock.dt
    return hook
ns.on("tracer_advection", at("vof"), after=True)
ns.on("end_timestep", at("end"), after=True)
ns.start()
for _ in range(3): ns.step()
names = ("ux", "uy", "f", "p", "gx", "gy")
for it in range(3):
    for tag in ("vof", "end"):
        C = loadC(tag, it); L = snap[(tag, it)]
        diffs = []
        for nm, a, b in zip(names, L, C):
            if nm == "p":
                a = a - a.mean(); b = b - b.mean()
            diffs.append("%s %.1e" % (nm, np.abs(a - b).max()))
        Cf = loadCuf(tag, it); Lf = snap[(tag + "_uf", it)]
        print("step %d %-3s dt=%.17g  " % (it, tag, snap[(tag + "_dt", it)]) + "  ".join(diffs) + "  ufx %.1e ufy %.1e" % (np.abs(Cf[0] - Lf[0]).max(), np.abs(Cf[1] - Lf[1]).max()))
C = loadC("vof", 1); L = snap[("vof", 1)]; f1 = L[2]
for nm, a, b in zip(("ux", "uy"), L[:2], C[:2]):
    d = np.abs(a - b); idx = np.argwhere(d > 1e-10)
    print(nm, "cells differing:", len(idx), "rows j:", sorted(set(idx[:, 1].tolist()))[:10], "uniform in i?", len(set(idx[:, 0].tolist())))
    j = idx[0][1]; print("  at j", j, "f(start)", snap[("end",0)][2][0, j-1:j+2], "f(after)", f1[0, j-1:j+2], "lib", a[0, j], "C", b[0, j])
    print("  C value varies along i (x)?", np.ptp(b[:, j]), " lib along i:", np.ptp(a[:, j]))
for nm, a, b in zip(("ux", "uy"), L[:2], C[:2]):
    d = np.abs(a - b); idx = np.argwhere(d > 1e-10)
    print(nm, [tuple(map(int, x)) for x in idx], ["%.2e" % d[tuple(x)] for x in idx])
    j = 17
    print("   lib row j=17 deviations from its median (i: value):", [(int(i), "%.1e" % (a[i, j] - np.median(a[:, j]))) for i in np.argsort(-np.abs(a[:, j] - np.median(a[:, j])))[:4]])
    print("   C   row j=17 deviations:", [(int(i), "%.1e" % (b[i, j] - np.median(b[:, j]))) for i in np.argsort(-np.abs(b[:, j] - np.median(b[:, j])))[:4]])

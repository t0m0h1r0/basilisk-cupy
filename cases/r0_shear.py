"""A-7 (#167, #145): R0's required test "oblique two-layer shear flow + Galilean boost" (docs/notes/111, section 6).

Problem (periodic unit square, n = (a, b) integers):
    z = a (x - Ux t) + b (y - Uy t)  (mod 1),   t_hat = (-b, a) / |n|,   u = U + t_hat w(z, t),   p = const
    rho w_t = |n|^2 (mu w_z)_z,   [w] = [mu w_z] = 0 at the interfaces;  phase 1 (rho1, mu1) in alpha <= z < beta
(the convention of the audit's independent reference, C-3: audit/c3/oblique_shear_reference.py).

Reference (independent of C-3's spectral elements): the eigenmodes in closed form. In each layer
W = A cos k s + B sin k s, k = sqrt(lambda rho / (|n|^2 mu)); the state (W, |n|^2 mu W') is carried across a layer by
[[cos kL, sin kL / (m k)], [-m k sin kL, cos kL]] (m = |n|^2 mu); the rates lambda are the roots of tr(Phi_L Phi_H) = 2
(``cases.viscous_stress.stokes_rate``, with Sturm's count) and the mode's start state is the null vector of
Phi_L Phi_H - I. Cell averages are exact: with G1' = W and G2' = G1 (closed forms per layer, continued over periods),
the average of g(a x + b y + z0) over a cell is a second difference of G2 / (a b h^2) (a first difference of G1 /
(b h) when a = 0). The same for the volume fraction (g the indicator).

  compare_c3()  the eigenvalues, the modes and the cell averages against C-3's reference.
  run()         the library's solver as it is (TwoPhase(conserving=True), filtered=False, the default options):
                the errors of the tangential velocity (cell averages, L1 / L2 / Linf, in all cells, in the interface
                band and in the pure phases), of f (L1), the phase mass, the total momentum, the discrete divergence
                and the gap between the face flux and the cells' velocity, at T/4, T/2, T.
  baseline()    the values before the R0 changes (recorded only, not a verdict: #167).

    python -m cases.r0_shear       # docs/figures/r0_shear.json (about 20 min, 16 processes)
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

RHO = (1.0, 1e-3)
MU = (1e-2, 1e-2 / 55.0)          # water / air ratio of the dynamic viscosities (as C-3 and docs/notes/100)
ALPHA, BETA = 0.25, 0.75
MODES = {1: 1.0, 2: 0.3}          # C-3's suggestion: a superposition of the two lowest modes (max |w| amplitudes)


class Shear:
    def __init__(self, n=(1, 1), U=(1.0, 0.37), alpha=ALPHA, beta=BETA, rho=RHO, mu=MU, nmodes=2):
        from cases.viscous_stress import stokes_rate

        self.n, self.U = (int(n[0]), int(n[1])), (float(U[0]), float(U[1]))
        self.alpha, self.beta, self.rho, self.mu = alpha, beta, rho, mu
        self.k2 = float(n[0] ** 2 + n[1] ** 2)
        self.LH, self.LL = beta - alpha, 1.0 - (beta - alpha)
        mH, mL = self.k2 * mu[0], self.k2 * mu[1]
        # stokes_rate: rho_H = 1, rho_L = 1 / ratio, heavy layer [a, b): rho must be (1, 1 / ratio)
        assert rho[0] == 1.0
        self.lam = list(stokes_rate(mH, mL, 1.0 / rho[1], alpha, beta, which=nmodes))
        self.modes = []
        for lam in self.lam:
            kH, kL = math.sqrt(lam * rho[0] / mH), math.sqrt(lam * rho[1] / mL)
            P = lambda k, L, m: np.array([[math.cos(k * L), math.sin(k * L) / (m * k)],  # noqa: E731
                                          [-m * k * math.sin(k * L), math.cos(k * L)]])
            PH, PL = P(kH, self.LH, mH), P(kL, self.LL, mL)
            v0 = np.linalg.svd(PL @ PH - np.eye(2))[2][-1]
            v1 = PH @ v0
            mode = {"lam": lam, "kH": kH, "kL": kL, "AH": v0[0], "BH": v0[1] / (mH * kH), "AL": v1[0],
                    "BL": v1[1] / (mL * kL), "scale": 1.0}
            s = np.linspace(0, 1, 20001)
            Wv = self._W(mode, s)
            mode["scale"] = 1.0 / Wv[np.argmax(np.abs(Wv))]          # max |W| = 1, the extreme value positive
            mode["closure"] = float(np.abs(PL @ PH @ v0 - v0).max())
            self.modes.append(mode)

    # --- one mode, s = z - alpha in [0, 1) ----------------------------------------------------------------------
    def _W(self, m, s):
        s = np.asarray(s, float)
        h = s < self.LH
        sl = s - self.LH
        out = np.where(h, m["AH"] * np.cos(m["kH"] * s) + m["BH"] * np.sin(m["kH"] * s),
                       m["AL"] * np.cos(m["kL"] * sl) + m["BL"] * np.sin(m["kL"] * sl))
        return m["scale"] * out

    @staticmethod
    def _F(A, B, k, s):
        """int_0^s and its integral of A cos k t + B sin k t."""
        F1 = A * np.sin(k * s) / k + B * (1 - np.cos(k * s)) / k
        F2 = A * (1 - np.cos(k * s)) / k ** 2 + B * (s - np.sin(k * s) / k) / k
        return F1, F2

    def _G(self, m, s):
        """G1p, G2p over one period (s in [0, 1)); m None: the heavy phase's indicator; "one": the constant."""
        s = np.asarray(s, float)
        h = s < self.LH
        sl = np.maximum(s - self.LH, 0.0)
        if m is None:
            return np.minimum(s, self.LH), np.where(h, s * s / 2, self.LH ** 2 / 2 + self.LH * sl)
        if m == "one":
            return s, s * s / 2
        H1, H2 = self._F(m["AH"], m["BH"], m["kH"], np.minimum(s, self.LH))
        E1, E2 = self._F(m["AH"], m["BH"], m["kH"], self.LH)
        L1, L2 = self._F(m["AL"], m["BL"], m["kL"], sl)
        G1 = np.where(h, H1, E1 + L1)
        G2 = np.where(h, H2, E2 + E1 * sl + L2)
        return m["scale"] * G1, m["scale"] * G2

    def _G2(self, m, z):
        """The second antiderivative on the whole line (z - alpha = k + s)."""
        r = np.asarray(z, float) - self.alpha
        k = np.floor(r)
        s = r - k
        I1, J = (float(v) for v in self._G(m, 1.0))
        G1p, G2p = self._G(m, s)
        return I1 * k * (k - 1) / 2 + k * J + k * I1 * s + G2p, k * I1 + G1p

    def _cells(self, m, N, t):
        a, b = self.n
        h = 1.0 / N
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        z0 = a * i * h + b * j * h - (a * self.U[0] + b * self.U[1]) * t
        if a != 0 and b != 0:
            G = lambda z: self._G2(m, z)[0]  # noqa: E731
            return (G(z0 + a * h + b * h) - G(z0 + a * h) - G(z0 + b * h) + G(z0)) / (a * b * h * h)
        c = a if a != 0 else b
        G = lambda z: self._G2(m, z)[1]  # noqa: E731
        return (G(z0 + c * h) - G(z0)) / (c * h)

    # --- the solution ------------------------------------------------------------------------------------------
    def w_cells(self, N, t, modes=MODES, mean=0.0):
        w = mean * self._cells("one", N, t)
        for k, amp in modes.items():
            w = w + amp * math.exp(-self.lam[k - 1] * t) * self._cells(self.modes[k - 1], N, t)
        return w

    def w_at(self, z, t, modes=MODES, mean=0.0):
        s = (np.asarray(z, float) - self.alpha) % 1.0
        return mean + sum(amp * math.exp(-self.lam[k - 1] * t) * self._W(self.modes[k - 1], s)
                          for k, amp in modes.items())

    def velocity_cells(self, N, t, modes=MODES, mean=0.0):
        w = self.w_cells(N, t, modes, mean)
        r = math.sqrt(self.k2)
        return self.U[0] - self.n[1] / r * w, self.U[1] + self.n[0] / r * w

    def face_fluxes(self, N, t, modes=MODES, mean=0.0):
        """Exact face averages of the normal velocity from the stream function psi = Ux y - Uy x - G1(z) / |n|
        (u = psi_y, v = -psi_x; G1' = w): discretely divergence-free. Returns (ux on x-faces (N+1, N),
        uy on y-faces (N, N+1))."""
        a, b = self.n
        r = math.sqrt(self.k2)
        h = 1.0 / N
        xv = np.arange(N + 1) * h
        X, Y = np.meshgrid(xv, xv, indexing="ij")
        z = a * (X - self.U[0] * t) + b * (Y - self.U[1] * t)
        G1 = mean * self._G2("one", z)[1]
        for k, amp in modes.items():
            G1 = G1 + amp * math.exp(-self.lam[k - 1] * t) * self._G2(self.modes[k - 1], z)[1]
        psi = self.U[0] * Y - self.U[1] * X - G1 / r
        return (psi[:, 1:] - psi[:, :-1]) / h, -(psi[1:, :] - psi[:-1, :]) / h

    def fraction_cells(self, N, t):
        return self._cells(None, N, t)

    def period(self):
        """The time in which the interface moves by one period of z (boost), else 1 / lambda_1."""
        s = abs(self.n[0] * self.U[0] + self.n[1] * self.U[1])
        return 1.0 / s if s > 0 else 1.0 / self.lam[0]


# --- the comparison with C-3 --------------------------------------------------------------------------------------
def compare_c3(ns_=((0, 1), (1, 1), (1, 2)), N=64, ts=(0.0, 0.3)):
    """Needs the audit branch's file (audit/c3/oblique_shear_reference.py, path in C3_REFERENCE or that default)."""
    import importlib.util

    path = os.environ.get("C3_REFERENCE", "audit/c3/oblique_shear_reference.py")
    spec = importlib.util.spec_from_file_location("c3ref", path)
    c3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c3)
    out = []
    for n in ns_:
        mine = Shear(n=n)
        ref = c3.Shear(n=n, U=mine.U, alpha=ALPHA, beta=BETA, rho=RHO, mu=MU)
        z = np.linspace(0, 1, 997, endpoint=False)
        row = {"n": list(n), "lam_mine": mine.lam, "lam_c3": [float(v) for v in ref.lam[1:3]],
               "closure": [m["closure"] for m in mine.modes]}
        row["lam_rel_diff"] = [abs(a_ - b_) / b_ for a_, b_ in zip(mine.lam, row["lam_c3"])]
        # the modes' normalisation differs (the sign of a mode whose max and min have the same size, and C-3 takes
        # max |phi| on its nodes): match each mode by its least-squares factor, then compare shapes and cells
        c = np.zeros(len(ref.lam))
        row["mode_shape_diff"], row["mode_factor"] = [], []
        for k in (1, 2):
            e = np.zeros(len(ref.lam))
            e[k] = 1.0
            phi = ref.w_at(z, e, 0.0)
            W = mine._W(mine.modes[k - 1], (z - ALPHA) % 1.0)
            sk = float(phi @ W / (phi @ phi))
            row["mode_factor"].append(sk)
            row["mode_shape_diff"].append(float(np.abs(sk * phi - W).max()))
            c[k] = MODES[k] * sk
        row["w_diff"] = {str(t): float(np.abs(mine.w_at(z, t) - ref.w_at(z, c, t)).max()) for t in ts}
        u1, v1 = mine.velocity_cells(N, ts[-1])
        u2, v2 = ref.velocity_cells(N, c, ts[-1])
        row["cells_diff"] = float(max(np.abs(u1 - u2).max(), np.abs(v1 - v2).max()))
        row["fraction_diff"] = float(np.abs(mine.fraction_cells(N, ts[-1]) - ref.fraction_cells(N, ts[-1])).max())
        out.append(row)
    return out


# --- the solver ---------------------------------------------------------------------------------------------------
def band_mask(ex, N, t, width=2.5):
    """The interface band: cell centres within ``width`` cells of an interface (the distance along the normal,
    |z - z_I| / |n|, periodic in z), from the exact interfaces at time t."""
    a, b = ex.n
    c = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(c, c, indexing="ij")
    z = a * (X - ex.U[0] * t) + b * (Y - ex.U[1] * t)
    d = np.minimum(np.abs((z - ex.alpha + 0.5) % 1.0 - 0.5), np.abs((z - ex.beta + 0.5) % 1.0 - 0.5))
    return d / math.sqrt(ex.k2) < width / N


def run(n=(1, 1), boost=True, N=64, cfl=0.5, samples=(0.25, 0.5, 1.0), tolerance=1e-10, backend="numpy",
        setup=None, save=None, exact_uf=True):
    """The library's solver (``setup(ns, tp)`` may add hooks, e.g. A-3's). Samples are fractions of T.
    exact_uf: the first step's face velocity is the exact face average (divergence-free); otherwise the library's
    ``start`` (the interpolated cell averages, not projected: the first VOF step then changes the phase mass, by
    up to 1e-4 for n = (1, 2) at N = 32; note A-7)."""
    import warnings

    warnings.filterwarnings("ignore")
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.backend import to_numpy
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    ex = Shear(n=n, U=(1.0, 0.37) if boost else (0.0, 0.0))
    T = ex.period()
    grid = CartesianGrid((N, N), backend=backend)
    xp = grid.xp
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=tolerance, mu=MU[0])
    f0 = ex.fraction_cells(N, 0.0)
    tp = TwoPhase(ns, CellField.from_interior(grid, xp.asarray(f0), BoundaryConditions.periodic(2)), rho1=RHO[0],
                  rho2=RHO[1], mu1=MU[0], mu2=MU[1], conserving=True)
    if setup is not None:
        setup(ns, tp)
    ns.cfl = cfl
    u0, v0 = ex.velocity_cells(N, 0.0)
    ns.u[0].interior = xp.asarray(u0)
    ns.u[1].interior = xp.asarray(v0)
    r = math.sqrt(ex.k2)
    th = np.array([-ex.n[1] / r, ex.n[0] / r])

    def momentum():
        f = to_numpy(tp.f.interior)
        rho = np.clip(f, 0, 1) * (RHO[0] - RHO[1]) + RHO[1]
        return np.array([(rho * to_numpy(ns.u[k].interior)).sum() / N ** 2 for k in range(2)])

    P0, V0 = momentum(), float(f0.sum())
    # the normalisation of the momentum's change: |P0|, or the total mass times the largest speed (no boost: P0 ~ 0)
    Pref = max(float(np.hypot(*P0)), (RHO[0] * ex.LH + RHO[1] * ex.LL) * float(np.hypot(u0, v0).max()))
    times = [s * T for s in samples]
    out = {"n": list(n), "boost": boost, "N": N, "cfl": cfl, "T": T, "lam": ex.lam, "rows": [], "failed": None,
           "gap_history": []}
    ns.start(events=times)
    if exact_uf:
        fx, fy = ex.face_fluxes(N, 0.0)
        ns.uf.components[0][...] = xp.asarray(fx)
        ns.uf.components[1][...] = xp.asarray(fy)
    out["exact_uf"] = exact_uf
    try:
        for te in times:
            while ns.clock.t < te - 1e-12:
                ns.step()
                ns.u.fill_ghosts()
                gap = max(float(xp.abs(ns.uf.components[k] - ns.ops.interp_f(ns.u[k], k)).max()) for k in range(2))
                out["gap_history"].append((float(ns.clock.t), gap))
                if not math.isfinite(gap) or gap > 1e3:
                    raise FloatingPointError("blow-up at t = %.4f" % ns.clock.t)
            u, v = to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior)
            ue, ve = ex.velocity_cells(N, te)
            f = to_numpy(tp.f.interior)
            et = (u - ue) * th[0] + (v - ve) * th[1]                      # the tangential velocity's error
            e = np.hypot(u - ue, v - ve)
            band = band_mask(ex, N, te)
            uf = [to_numpy(c) for c in ns.uf.components]
            div = (uf[0][1:, :] - uf[0][:-1, :] + uf[1][:, 1:] - uf[1][:, :-1])     # h div uf (h = 1/N)
            row = {"t": te, "steps": ns.clock.i}
            for name, sel in (("all", np.ones_like(band)), ("band", band), ("pure", ~band)):
                x = np.abs(et[sel])
                row[name] = ({"L1": float(x.mean()), "L2": float(np.sqrt((x * x).mean())), "Linf": float(x.max()),
                              "cells": int(x.size)} if x.size else {"L1": 0.0, "L2": 0.0, "Linf": 0.0, "cells": 0})
            row["speed_Linf"] = float(e.max())
            row["f_L1"] = float(np.abs(f - ex.fraction_cells(N, te)).mean())
            row["mass"] = abs(float(f.sum()) - V0) / V0
            row["momentum"] = float(np.hypot(*(momentum() - P0)) / Pref)
            row["divergence"] = float(np.abs(div).max()) / max(1.0, float(np.hypot(*ex.U)))
            row["gap"] = out["gap_history"][-1][1] if out["gap_history"] else 0.0
            out["rows"].append(row)
    except Exception as exc:  # noqa: BLE001  (a failed run is a result)
        out["failed"] = repr(exc)[:200]
    if save:
        np.save(save, np.stack([to_numpy(ns.u[0].interior), to_numpy(ns.u[1].interior), to_numpy(tp.f.interior)]))
    g = out.pop("gap_history")
    out["gap_samples"] = g[:: max(1, len(g) // 50)]
    return out


GRIDS = (32, 48, 64, 96, 128)
CASES = [((0, 1), True), ((0, 1), False), ((1, 1), True), ((1, 1), False), ((1, 2), True), ((1, 2), False)]


def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    (n, boost), N = a[0], a[1]
    return run(n=n, boost=boost, N=N, exact_uf=a[2] if len(a) > 2 else True)


def main() -> None:
    import multiprocessing as mp

    out = os.path.join("docs", "figures", "r0_shear.json")
    part = out + ".jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            d = json.loads(line)
            done[(tuple(d["n"]), d["boost"], d["N"]) + (() if d.get("exact_uf", True) else (False,))] = d
    J = [((n, b), N) for n, b in CASES for N in GRIDS if (n, b, N) not in done]
    J += [((n, b), N, False) for n, b in (((1, 2), True), ((1, 2), False)) for N in (32, 64, 128)
          if (n, b, N, False) not in done]
    J.sort(key=lambda a: -a[1])

    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "16")), maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, J, chunksize=1):
            with open(part, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            done[(tuple(r["n"]), r["boost"], r["N"]) + (() if r.get("exact_uf", True) else (False,))] = r
    res = {"baseline": list(done.values())}
    if os.path.exists(os.environ.get("C3_REFERENCE", "audit/c3/oblique_shear_reference.py")):
        res["compare_c3"] = compare_c3()
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()


def plot() -> None:
    """docs/figures/r0_shear.png: (a) L2 error of the tangential velocity at T against N (the library, before the R0
    changes); (b) the gap between the face flux and the cells' velocity in time (N = 64)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(os.path.join("docs", "figures", "r0_shear.json")))["baseline"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, b in CASES:
        rows = sorted((d for d in R if tuple(d["n"]) == n and d["boost"] == b and d.get("exact_uf", True)),
                      key=lambda d: d["N"])
        lab = "n = %s%s" % (n, ", boost" if b else "")
        ax[0].loglog([d["N"] for d in rows], [d["rows"][-1]["all"]["L2"] for d in rows], "o-" if b else "s--",
                     ms=4, label=lab)
        d64 = next(d for d in rows if d["N"] == 64)
        g = np.array(d64["gap_samples"])
        ax[1].semilogy(g[:, 0] / d64["T"], np.maximum(g[:, 1], 1e-16), "-" if b else "--", label=lab)
    Ns = np.array(GRIDS)
    ax[0].loglog(Ns, 0.4 / Ns, ":", color="0.5", label="1st order")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("L2 error of w (cell averages) at T")
    ax[0].set_title("(a) the library before R0 (density ratio 1000)")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("t / T")
    ax[1].set_ylabel("max |u_f - interp(u)|")
    ax[1].set_title("(b) face flux vs cell velocity, N = 64")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "r0_shear.png"), dpi=130)

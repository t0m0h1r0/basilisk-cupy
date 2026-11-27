"""docs/notes/100 (V3, V4): the minimum-viscous-energy reconstruction (the user's proposal) and the energy inner
products of advection and viscosity.

R_nu q = argmin { 1/2 int mu |v'|^2 : A_rho v = q }. Its Euler-Lagrange equation -(mu v')' = lambda_i rho on K_i makes
the stress tau = mu v' linear in the cell's mass coordinate: tau_h = (1 - s) tau_{i-1} + s tau_i, i.e. exactly
note 99's stress reconstruction, with H tau = D q. In cell i

    v(x) = q_i + w(x) - <w>_{rho,i},   w(x) = int_{x_i}^x tau_h / mu,

and  G_nu = R_nu^* R_nu (rho-weighted),  K_nu = D^T H^{-1} D,  G_nu q' = -K_nu q  (Ritz-Galerkin).
Compact form: G_nu = M + D^T H^{-1} Q H^{-1} D, Q_fg = int rho w_f w_g (w_f the mean-free correction of a unit
stress at face f), Q tridiagonal.

    python -m cases.viscous_minenergy
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.viscous_stress import X3, W3, Setup, mus, stokes_rate


class MinEnergy:
    def __init__(self, S: Setup, nodes=None):
        """``nodes``: (points, weights) on [0, 1] per phase piece (default 3-point Gauss: exact for G_nu)."""
        self.S = S
        XN, WN = nodes if nodes is not None else (X3, W3)
        N, h = S.N, S.h
        # per cell: quadrature points (x), rho-weights, and the values of the two face-stress responses
        # w_left(x), w_right(x) = int_{x_i}^x psi_{face} / mu, minus their rho-means
        self.cells = []
        for i, ps in enumerate(S.P):
            mi = S.m[i]
            X, Wr, wl, wr = [], [], [], []
            R = 0.0          # cumulative mass
            Il = Ir = 0.0    # cumulative integrals of (1 - s)/mu and s/mu
            for s0, s1, hv in ps:
                rho = 1.0 if hv else 1.0 / S.ratio
                mu = S.mu_H if hv else S.mu_L
                L = (s1 - s0) * h
                d = L * XN
                sA = R / mi                       # s at the piece start
                # s(x) = sA + rho d / mi ; int_0^d s = sA d + rho d^2 / (2 mi)
                int_s = sA * d + rho * d * d / (2 * mi)
                wr.append(Ir + int_s / mu)
                wl.append(Il + (d - int_s) / mu)
                X.append((i + s0) * h + d)
                Wr.append(rho * L * WN)
                full_s = sA * L + rho * L * L / (2 * mi)
                Ir += full_s / mu
                Il += (L - full_s) / mu
                R += rho * L
            X, Wr = np.concatenate(X), np.concatenate(Wr)
            wl, wr = np.concatenate(wl), np.concatenate(wr)
            wl -= (Wr @ wl) / Wr.sum()
            wr -= (Wr @ wr) / Wr.sum()
            self.cells.append((X, Wr, wl, wr))
        # the basis: v = q_i + tau_{i-1} wl + tau_i wr on cell i, tau = H^{-1} D q
        self.T = np.linalg.solve(S.H, S.D)           # face stresses per unit average (N faces x N data)
        G = np.diag(S.m).astype(float)
        Q = np.zeros((N, N))
        for i, (X, Wr, wl, wr) in enumerate(self.cells):
            fl, fr = (i - 1) % N, i
            Q[fl, fl] += Wr @ (wl * wl)
            Q[fr, fr] += Wr @ (wr * wr)
            Q[fl, fr] += Wr @ (wl * wr)
            Q[fr, fl] += Wr @ (wl * wr)
        self.Q = Q
        self.G = G + self.T.T @ Q @ self.T                 # the compact form M + D^T H^-1 Q H^-1 D
        self.K = S.D.T @ self.T                            # D^T H^-1 D

    def values(self, q):
        """(X, rho-weights, v) on every cell's quadrature points."""
        tau = self.T @ q
        N = self.S.N
        out = []
        for i, (X, Wr, wl, wr) in enumerate(self.cells):
            out.append((X, Wr, q[i] + tau[(i - 1) % N] * wl + tau[i] * wr))
        return out

    def gram_direct(self):
        """G from the reconstructed values of the unit averages (check of the compact form)."""
        N = self.S.N
        B = []
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.0
            B.append(np.concatenate([v for _, _, v in self.values(e)]))
        B = np.array(B).T
        W = np.concatenate([w for _, w, _ in self.values(np.zeros(N))])
        return B.T @ (W[:, None] * B)


def eigenfunction(mu_H, mu_L, ratio, a, b, lam):
    """The exact Stokes mode for the rate lam (u, tau continuous): (u0 at x = a, tau0 at a) from the null vector of
    Phi - I, propagated through the layers. Returns u(x)."""
    LH = (b - a) % 1.0

    def phi(L, rho, mu):
        k = math.sqrt(lam * rho / mu)
        return np.array([[math.cos(k * L), math.sin(k * L) / (mu * k)], [-mu * k * math.sin(k * L), math.cos(k * L)]])
    P = phi(1 - LH, 1.0 / ratio, mu_L) @ phi(LH, 1.0, mu_H)
    _, _, Vt = np.linalg.svd(P - np.eye(2))
    z0 = Vt[-1]

    def u(x):
        x = np.asarray(x, dtype=float)
        d = (x - a) % 1.0
        out = np.empty_like(d)
        for k, dd in enumerate(d):
            if dd <= LH:
                out[k] = (phi(dd, 1.0, mu_H) @ z0)[0]
            else:
                out[k] = (phi(dd - LH, 1.0 / ratio, mu_L) @ (phi(LH, 1.0, mu_H) @ z0))[0]
        return out
    return u


def methods(S, ME, GA=None):
    """The (mass matrix, viscous matrix) pairs: A (M, K4), B (G_A, K4), C (G_nu, K_nu), D (G_A, K_nu)."""
    M = np.diag(S.m)
    out = {"A: M, K4": (M, S.K), "C: Gnu, Knu": (ME.G, ME.K), "H4-ref": (M, S.K)}
    if GA is not None:
        out["B: GA, K4"] = (GA, S.K)
        out["D: GA, Knu"] = (GA, ME.K)
    out.pop("H4-ref")
    return out


def _gen_eig(Gm, Km):
    L = np.linalg.cholesky(Gm)
    Li = np.linalg.inv(L)
    A = Li @ Km @ Li.T
    ev, V = np.linalg.eigh(0.5 * (A + A.T))
    return ev, Li.T @ V


def stokes_test(N, ratio, kind, a=0.2613, b=0.7137, t=0.01, GA=None):
    """Slowest two rates (relative error) and the mode-1 decay at time t (max error of the averages / max |q0|)."""
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    ME = MinEnergy(S)
    ex = stokes_rate(mu_H, mu_L, ratio, a, b, which=2)
    u = eigenfunction(mu_H, mu_L, ratio, a, b, ex[0])
    q0 = S.averages(u)
    qe = q0 * math.exp(-ex[0] * t)
    res = {}
    for name, (Gm, Km) in methods(S, ME, GA).items():
        ev, V = _gen_eig(Gm, Km)
        evn = np.sort(ev[ev > 1e-9 * ev.max()])[:2]
        # q(t) = V exp(-ev t) V^T G q0  (V G-orthonormal)
        c = V.T @ (Gm @ q0)
        qt = V @ (np.exp(-ev * t) * c)
        res[name] = {"rate_err": [float(abs(x - y) / y) for x, y in zip(evn, ex)],
                     "decay_err": float(np.abs(qt - qe).max() / np.abs(q0).max())}
    return {"exact": ex, **res}


def robustness(N=32, ratio=1e6, kind="nu", cs=(1e-8, 1e-6, 1e-4, 1e-2, 0.5, 0.99)):
    """Gram lambda_max(M^-1/2 G_nu M^-1/2), stiffness of K_nu in the M and G_nu inner products (x h^2/nu_max), and
    the compact-form check, over heavy fractions c of the cut cell."""
    mu_H, mu_L = mus(ratio, kind)
    nu = max(mu_H, mu_L * ratio)
    out = []
    for c in cs:
        h = 1.0 / N
        a = (N // 4 + 1 - c) * h
        S = Setup(N, ratio, a, 0.7137, mu_H, mu_L)
        ME = MinEnergy(S)
        Mh = 1.0 / np.sqrt(S.m)
        gram = float(np.linalg.eigvalsh(ME.G * Mh[:, None] * Mh[None, :])[-1])
        stM = float(np.linalg.eigvalsh(ME.K * Mh[:, None] * Mh[None, :])[-1] * h * h / nu)
        stG = float(_gen_eig(ME.G, ME.K)[0][-1] * h * h / nu)
        st4 = float(np.linalg.eigvalsh(S.K * Mh[:, None] * Mh[None, :])[-1] * h * h / nu)
        Gd = ME.gram_direct()
        out.append({"c": c, "gram_max": gram, "stiff_M": stM, "stiff_Gnu": stG, "stiff_H4": st4,
                    "compact_vs_direct": float(np.abs(Gd - ME.G).max() / np.abs(Gd).max()),
                    "gram_min": float(np.linalg.eigvalsh(ME.G * Mh[:, None] * Mh[None, :])[0])})
    return out


def structure(N=32, ratio=1e3, kind="mu", a=0.2613, b=0.7137):
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    ME = MinEnergy(S)
    one = np.ones(N)
    # the mass-weighted means of the reconstruction equal the data
    q = np.random.default_rng(0).normal(size=N)
    means = np.array([(Wr @ v) / Wr.sum() for _, Wr, v in ME.values(q)])
    # viscous energy of the reconstruction equals q^T K q
    tau = ME.T @ q
    Evisc = float(tau @ S.H @ tau)
    return {"mean_err": float(np.abs(means - q).max()), "G1_m": float(np.abs(ME.G @ one - S.m).max() / S.m.max()),
            "K1": float(np.abs(ME.K @ one).max()), "energy_identity": float(abs(Evisc - q @ ME.K @ q) / Evisc),
            "G_minus_M_min_eig": float(np.linalg.eigvalsh(ME.G - np.diag(S.m))[0] / S.m.max()),
            "K_min_eig_rel": float(np.linalg.eigvalsh(ME.K)[0] / np.linalg.eigvalsh(ME.K)[-1])}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    res = {}
    with ProcessPoolExecutor(24) as ex:
        fs = {k: ex.submit(structure, 32, r, k) for k in ("nu", "mu", "water-air") for r in (1e3,)}
        ft = {(N, r, k): ex.submit(stokes_test, N, r, k) for k in ("nu", "mu", "water-air") for r in (1.0, 1e3)
              for N in (32, 64, 128, 256)}
        fr = {k: ex.submit(robustness, 32, 1e6, k) for k in ("nu", "mu", "water-air")}
        res["structure"] = {k: f.result() for k, f in fs.items()}
        res["stokes"] = [{"N": N, "ratio": r, "kind": k, **f.result()} for (N, r, k), f in ft.items()]
        res["robustness"] = {k: f.result() for k, f in fr.items()}
    json.dump(res, open("docs/figures/viscous_minenergy.json", "w"), indent=1, default=float)
    print(json.dumps(res["structure"], indent=1))
    for r in res["stokes"]:
        print(r["kind"], r["ratio"], r["N"], {k: v for k, v in r.items() if k not in ("kind", "ratio", "N", "exact")})
    for k, rows in res["robustness"].items():
        for x in rows:
            print("robust", k, x)





# --------------------------------------------------------------------------------------------- V4
def advection_basis(N, ratio, a, b, X):
    """Values at X of the two-phase PCCD advection reconstruction (partial-cell phase split, SplitBlend) of the unit
    averages: (len(X), N)."""
    from cases.pccd_local import SplitBlend

    cols = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        cols.append(SplitBlend(N, ratio, a, b, e, mixed_mode="partial")(X))
    return np.stack(cols, axis=1)


def v4_test(N, ratio, kind, a=0.2613, b=0.7137, t=0.01):
    """Methods A (M, K4), B (G_A, K4), C (G_nu, K_nu), D (G_A, K_nu), E (G_A, P^T K_nu P) on the Stokes mode."""
    from cases.pccd import WG, XG

    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    ME = MinEnergy(S, nodes=(XG, WG))                    # 8-point Gauss per piece: exact for the cross products
    X = np.concatenate([c[0] for c in ME.cells])
    W = np.concatenate([c[1] for c in ME.cells])
    BA = advection_basis(N, ratio, a, b, X)
    Bnu = np.stack([np.concatenate([v for _, _, v in ME.values(np.eye(N)[j])]) for j in range(N)], axis=1)
    GA = BA.T @ (W[:, None] * BA)
    Cna = Bnu.T @ (W[:, None] * BA)
    P = np.linalg.solve(ME.G, Cna)
    KE = P.T @ ME.K @ P
    ex = stokes_rate(mu_H, mu_L, ratio, a, b, which=2)
    u = eigenfunction(mu_H, mu_L, ratio, a, b, ex[0])
    q0 = S.averages(u)
    qe = q0 * math.exp(-ex[0] * t)
    M = np.diag(S.m)
    res = {"exact": ex}
    for name, (Gm, Km) in {"A": (M, S.K), "B": (GA, S.K), "C": (ME.G, ME.K), "D": (GA, ME.K), "E": (GA, KE)}.items():
        ev, V = _gen_eig(Gm, 0.5 * (Km + Km.T))
        evn = np.sort(ev[ev > 1e-9 * ev.max()])[:2]
        c = V.T @ (Gm @ q0)
        qt = V @ (np.exp(-ev * t) * c)
        Mh = 1.0 / np.sqrt(S.m)
        res[name] = {"rate_err": [float(abs(x - y) / y) for x, y in zip(evn, ex)],
                     "decay_err": float(np.abs(qt - qe).max() / np.abs(q0).max()),
                     "stiff_own": float(ev[-1] * S.h ** 2 / max(mu_H, mu_L * ratio)),
                     "gram_own": float(np.linalg.eigvalsh(Gm * Mh[:, None] * Mh[None, :])[-1])}
    res["PtGnuP_le_GA"] = float(np.linalg.eigvalsh(P.T @ ME.G @ P - GA)[-1] / np.linalg.eigvalsh(GA)[-1])
    return res


def main_v4() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(N, r, k) for k in ("nu", "mu", "water-air") for r in (1.0, 1e3) for N in (32, 64, 128)]
    with ProcessPoolExecutor(18) as ex:
        fs = [(j, ex.submit(v4_test, *j)) for j in jobs]
        res = [{"N": j[0], "ratio": j[1], "kind": j[2], **f.result()} for j, f in fs]
    json.dump(res, open("docs/figures/viscous_v4.json", "w"), indent=1, default=float)
    for r in res:
        print(r["kind"], r["ratio"], r["N"], " ".join(f"{m}: rate {max(r[m]['rate_err']):.2e} decay {r[m]['decay_err']:.2e}"
                                                      for m in "ABCDE"), f"PtGP-GA {r['PtGnuP_le_GA']:.1e}")


def d_stiffness(N, ratio, kind, c):
    """Method D's stiffness: lambda_max(K_nu, G_A) h^2 / nu_max, and lambda_max(G_A / M), the interface a in cell N/4
    with heavy fraction c."""
    from cases.pccd import WG, XG

    h = 1.0 / N
    a = (N // 4 + 1 - c) * h
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, 0.7137, mu_H, mu_L)
    ME = MinEnergy(S, nodes=(XG, WG))
    X = np.concatenate([cc[0] for cc in ME.cells])
    W = np.concatenate([cc[1] for cc in ME.cells])
    BA = advection_basis(N, ratio, a, 0.7137, X)
    GA = BA.T @ (W[:, None] * BA)
    Mh = 1.0 / np.sqrt(S.m)
    return {"stiff_D": float(_gen_eig(GA, ME.K)[0][-1] * h * h / max(mu_H, mu_L * ratio)),
            "gram_A": float(np.linalg.eigvalsh(GA * Mh[:, None] * Mh[None, :])[-1])}


if __name__ == "__main__":
    import sys

    main_v4() if "--v4" in sys.argv else main()

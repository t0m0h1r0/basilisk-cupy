"""docs/notes/101 (V0): the compact-stress viscous operators with walls (the user's proposal), 1-D two layers.

Domain [0, 1], heavy layer (rho = 1, mu_H) on (0, a), light (rho = 1/ratio, mu_L) on (a, 1); walls at 0 and 1 with
velocities u_L, u_R. Faces f = 0 .. N at x_f = f h (N + 1 stresses). Cell i = [x_i, x_{i+1}]. The hat functions psi_f
rise over cell f - 1 and fall over cell f (the wall faces have one side only), and integration by parts gives exactly

    face 0:  q_0 - u_L = int psi_0 tau / mu,   face N:  u_R - q_{N-1} = int psi_N tau / mu,
    interior:  q_f - q_{f-1} = int psi_f tau / mu,

i.e. H tau = D_b q + b, b = (-u_L, 0, .., 0, u_R). Operators:
* A: M q' = -D_b^T tau + F, tau = H_4^{-1}(D_b q + b), H_4 = (H + diag(H 1)) / 2;
* C: the minimum-viscous-energy reconstruction with the wall values (tau = H^{-1}(D_b q + b)), G_nu q' = -K_nu q + F
  for fixed walls, K_nu = D_b^T H^{-1} D_b.
Balances (A): d/dt sum m q = tau_N - tau_0 + sum F;  d/dt (q^T M q / 2) = -tau^T H_4 tau + u_R tau_N - u_L tau_0 + q.F.

    python -m cases.viscous_walls
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG
from cases.pccd_density import pieces
from cases.viscous_stress import X3, W3, mus


class Wall:
    def __init__(self, N, ratio, a, mu_H, mu_L, nodes=(X3, W3)):
        self.N, self.ratio, self.a, self.mu_H, self.mu_L = N, ratio, a, mu_H, mu_L
        h = self.h = 1.0 / N
        self.P = pieces(N, 0.0, a)                     # heavy on (0, a)
        XN, WN = nodes
        H = np.zeros((N + 1, N + 1))
        m = np.zeros(N)
        self.cells = []
        for i, ps in enumerate(self.P):
            mi = sum((s1 - s0) * h * (1.0 if hv else 1.0 / ratio) for s0, s1, hv in ps)
            m[i] = mi
            R = Il = Ir = 0.0
            rr = rf = rx = 0.0
            X, Wr, wl, wr = [], [], [], []
            for s0, s1, hv in ps:
                rho = 1.0 if hv else 1.0 / ratio
                mu = mu_H if hv else mu_L
                L = (s1 - s0) * h
                # H entries (exact with 3 points: products of linear functions)
                d3 = L * X3
                up = (R + rho * d3) / mi
                rr += (L * W3) @ (up * up) / mu
                rf += (L * W3) @ ((1 - up) ** 2) / mu
                rx += (L * W3) @ (up * (1 - up)) / mu
                # reconstruction responses at the nodes (for C): w_right = int s/mu, w_left = int (1 - s)/mu
                d = L * XN
                sA = R / mi
                int_s = sA * d + rho * d * d / (2 * mi)
                wr.append(Ir + int_s / mu)
                wl.append(Il + (d - int_s) / mu)
                X.append(i * h + s0 * h + d)
                Wr.append(rho * L * WN)
                full = sA * L + rho * L * L / (2 * mi)
                Ir += full / mu
                Il += (L - full) / mu
                R += rho * L
            fl, fr = i, i + 1                          # left face (falling psi), right face (rising psi)
            H[fr, fr] += rr
            H[fl, fl] += rf
            H[fl, fr] += rx
            H[fr, fl] += rx
            X, Wr, wl, wr = np.concatenate(X), np.concatenate(Wr), np.concatenate(wl), np.concatenate(wr)
            wl -= (Wr @ wl) / Wr.sum()
            wr -= (Wr @ wr) / Wr.sum()
            self.cells.append((X, Wr, wl, wr))
        self.H, self.m = H, m
        self.H4 = 0.5 * (H + np.diag(H.sum(1)))
        Db = np.zeros((N + 1, N))
        Db[0, 0] = 1.0
        Db[N, N - 1] = -1.0
        for f in range(1, N):
            Db[f, f], Db[f, f - 1] = 1.0, -1.0
        self.Db = Db
        # C: G_nu = M + T^T Q T, T = H^{-1} D_b (homogeneous walls); K_nu = D_b^T H^{-1} D_b
        Q = np.zeros((N + 1, N + 1))
        for i, (X, Wr, wl, wr) in enumerate(self.cells):
            Q[i, i] += Wr @ (wl * wl)
            Q[i + 1, i + 1] += Wr @ (wr * wr)
            Q[i, i + 1] += Wr @ (wl * wr)
            Q[i + 1, i] += Wr @ (wl * wr)
        T = np.linalg.solve(H, Db)
        self.Gnu = np.diag(m) + T.T @ Q @ T
        self.Knu = Db.T @ T
        self.K4 = Db.T @ np.linalg.solve(self.H4, Db)

    def averages(self, u):
        h = self.h
        out = np.zeros(self.N)
        for i, ps in enumerate(self.P):
            num = den = 0.0
            for s0, s1, hv in ps:
                rho = 1.0 if hv else 1.0 / self.ratio
                x = (i + s0 + (s1 - s0) * XG) * h
                num += rho * (s1 - s0) * h * (WG @ u(x))
                den += rho * (s1 - s0) * h
            out[i] = num / den
        return out

    def forcing(self, kind, value):
        """F_i = int_{K_i} f: "volume" (f per unit volume, a pressure gradient) or "accel" (f = rho g)."""
        return value * (self.h * np.ones(self.N) if kind == "volume" else self.m)


def exact_steady(mu_H, mu_L, ratio, a, uL, uR, force, value):
    """u(x) for the steady two-layer flow: tau' = -f (f = value or rho value), u(0) = uL, u(1) = uR."""
    def rho(x):
        return np.where(x < a, 1.0, 1.0 / ratio)

    def mu(x):
        return np.where(x < a, mu_H, mu_L)

    def F(x):                                          # int_0^x f
        if force == "volume":
            return value * x
        return value * np.where(x < a, x, a + (x - a) / ratio)

    def integ(g, x):
        out = np.zeros_like(x)
        for k, xx in enumerate(x):
            cuts = sorted({0.0, xx, *([a] if 0 < a < xx else [])})
            out[k] = sum((hi - lo) * (WG @ g(lo + (hi - lo) * XG)) for lo, hi in zip(cuts[:-1], cuts[1:]))
        return out
    I1 = integ(lambda s: 1.0 / mu(s), np.array([1.0]))[0]
    IF = integ(lambda s: F(s) / mu(s), np.array([1.0]))[0]
    tc = (uR - uL + IF) / I1                           # tau = tc - F(x)
    return lambda x: uL + integ(lambda s: (tc - F(s)) / mu(s), np.asarray(x, dtype=float))


def steady(N, ratio, kind, case, a=0.3613):
    """Couette (u_L = 0, u_R = 1, no force) or Poiseuille ("volume" / "accel" force, fixed walls): max error of the
    averages relative to max |q| for H_4 (A) and H (C, D share K_nu: the steady solution depends on K only)."""
    mu_H, mu_L = mus(ratio, kind)
    W = Wall(N, ratio, a, mu_H, mu_L)
    if case == "couette":
        uL, uR, force, val = 0.0, 1.0, "volume", 0.0
    else:
        uL, uR, force, val = 0.0, 0.0, case, 1.0
    b = np.zeros(N + 1)
    b[0], b[-1] = -uL, uR
    F = W.forcing(force, val)
    qe = W.averages(exact_steady(mu_H, mu_L, ratio, a, uL, uR, force, val))
    out = {}
    for name, Hm in (("H4", W.H4), ("H", W.H)):
        Hi_D = np.linalg.solve(Hm, W.Db)
        Hi_b = np.linalg.solve(Hm, b)
        q = np.linalg.solve(W.Db.T @ Hi_D, F - W.Db.T @ Hi_b)
        out[name] = float(np.abs(q - qe).max() / np.abs(qe).max())
    return out


def wall_rates(mu_H, mu_L, ratio, a, which=2):
    """Exact decay rates with u = 0 at both walls: Phi_12(lambda) = 0, Phi = Phi_L(1 - a) Phi_H(a)."""
    def phi(L, rho, mu, lam):
        k = math.sqrt(lam * rho / mu)
        return np.array([[math.cos(k * L), math.sin(k * L) / (mu * k)], [-mu * k * math.sin(k * L), math.cos(k * L)]])

    def f(lam):
        return float((phi(1 - a, 1.0 / ratio, mu_L, lam) @ phi(a, 1.0, mu_H, lam))[0, 1])
    cH, cL = math.sqrt(1.0 / mu_H) * a, math.sqrt(1.0 / ratio / mu_L) * (1 - a)
    ds = 0.02 / max(cH, cL)
    roots, sq, fp = [], ds, f(ds * ds)
    while len(roots) < which:
        s2 = sq + ds
        f2 = f(s2 * s2)
        if fp * f2 < 0:
            lo, hi, flo = sq, s2, fp
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid * mid)
                if fm * flo > 0:
                    lo, flo = mid, fm
                else:
                    hi = mid
            roots.append((0.5 * (lo + hi)) ** 2)
        sq, fp = s2, f2
    return roots, phi


def wall_stokes(N, ratio, kind, a=0.3613, t=0.01):
    """The slowest two wall rates and the decay of mode 1 at time t: A (M, K_4) and C (G_nu, K_nu)."""
    mu_H, mu_L = mus(ratio, kind)
    W = Wall(N, ratio, a, mu_H, mu_L)
    ex, phi = wall_rates(mu_H, mu_L, ratio, a)

    def u(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        for k, xx in enumerate(x):
            if xx <= a:
                out[k] = (phi(xx, 1.0, mu_H, ex[0]) @ np.array([0.0, 1.0]))[0]
            else:
                z = phi(a, 1.0, mu_H, ex[0]) @ np.array([0.0, 1.0])
                out[k] = (phi(xx - a, 1.0 / ratio, mu_L, ex[0]) @ z)[0]
        return out
    q0 = W.averages(u)
    qe = q0 * math.exp(-ex[0] * t)
    res = {"exact": ex}
    for name, (Gm, Km) in (("A", (np.diag(W.m), W.K4)), ("C", (W.Gnu, W.Knu))):
        L = np.linalg.cholesky(Gm)
        Li = np.linalg.inv(L)
        ev, V = np.linalg.eigh(Li @ Km @ Li.T)
        V = Li.T @ V
        qt = V @ (np.exp(-ev * t) * (V.T @ (Gm @ q0)))
        res[name] = {"rate_err": [float(abs(x - y) / y) for x, y in zip(np.sort(ev)[:2], ex)],
                     "decay_err": float(np.abs(qt - qe).max() / np.abs(q0).max())}
    return res


def balances(N=32, ratio=1e3, kind="water-air", a=0.3613, seed=0):
    """Momentum and energy balances of A with moving walls and a force: the discrete identities checked on random
    data (they hold algebraically; the check guards the wall rows)."""
    mu_H, mu_L = mus(ratio, kind)
    W = Wall(N, ratio, a, mu_H, mu_L)
    rng = np.random.default_rng(seed)
    q = rng.normal(size=N)
    uL, uR = 0.3, -0.7
    b = np.zeros(N + 1)
    b[0], b[-1] = -uL, uR
    F = W.forcing("accel", 0.5)
    tau = np.linalg.solve(W.H4, W.Db @ q + b)
    qdot = (-W.Db.T @ tau + F) / W.m
    dP = W.m @ qdot
    dE = q @ (W.m * qdot)
    return {"momentum_balance": float(abs(dP - (tau[-1] - tau[0] + F.sum()))),
            "energy_balance": float(abs(dE - (-tau @ W.H4 @ tau + uR * tau[-1] - uL * tau[0] + q @ F))),
            "dissipation": float(tau @ W.H4 @ tau)}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    res = {"steady": [], "stokes": [], "balances": balances()}
    with ProcessPoolExecutor(24) as ex:
        fs = [((N, r, k, c), ex.submit(steady, N, r, k, c)) for c in ("couette", "volume", "accel")
              for k in ("nu", "mu", "water-air") for r in (1.0, 1e3) for N in (32, 64, 128, 256)]
        ft = [((N, r, k), ex.submit(wall_stokes, N, r, k)) for k in ("nu", "mu", "water-air") for r in (1.0, 1e3)
              for N in (32, 64, 128, 256)]
        for (N, r, k, c), f in fs:
            res["steady"].append({"N": N, "ratio": r, "kind": k, "case": c, **f.result()})
        for (N, r, k), f in ft:
            res["stokes"].append({"N": N, "ratio": r, "kind": k, **f.result()})
    json.dump(res, open("docs/figures/viscous_walls.json", "w"), indent=1, default=float)
    print("balances", res["balances"])
    for c in ("couette", "volume", "accel"):
        for k in ("nu", "mu", "water-air"):
            for r in (1.0, 1e3):
                rows = sorted([x for x in res["steady"] if x["case"] == c and x["kind"] == k and x["ratio"] == r],
                              key=lambda x: x["N"])
                print(c, k, r, "H4", " ".join(f"{x['H4']:.2e}" for x in rows), "| H", " ".join(f"{x['H']:.2e}" for x in rows))
    for k in ("nu", "mu", "water-air"):
        for r in (1.0, 1e3):
            rows = sorted([x for x in res["stokes"] if x["kind"] == k and x["ratio"] == r], key=lambda x: x["N"])
            print("stokes", k, r, "exact", [f"{v:.5f}" for v in rows[0]["exact"]],
                  "A", " ".join(f"{max(x['A']['rate_err']):.1e}/{x['A']['decay_err']:.1e}" for x in rows),
                  "C", " ".join(f"{max(x['C']['rate_err']):.1e}/{x['C']['decay_err']:.1e}" for x in rows))


if __name__ == "__main__":
    main()

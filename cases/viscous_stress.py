"""docs/notes/99 (V0, V1): the compact-stress viscous operator for two phases (the user's proposal), 1-D periodic.

Cells K_i = [x_i, x_{i+1}], mass-weighted averages ubar_i, masses m_i; face f = x_{f+1} (right of cell f). The
function psi_f rises with the cumulative mass across K_f and falls across K_{f+1}:

    psi_f = (1/m_f) int_{x_f}^x rho  on K_f,   (1/m_{f+1}) int_x^{x_{f+2}} rho  on K_{f+1},   0 elsewhere,

and integration by parts gives exactly  ubar_{f+1} - ubar_f = int psi_f u' = int psi_f tau / mu  (tau = mu u'). With
the face stresses tau_g and tau_h = sum_g psi_g tau_g (sum_g psi_g = 1 in every cell):

    D ubar = H tau,   H_fg = int psi_f psi_g / mu   (symmetric positive definite, cyclic tridiagonal)
    H_L = diag(H 1),  H_4 = (H + H_L) / 2,  K = D^T H_4^{-1} D,  M d ubar/dt = -K ubar.

Constant rho, mu, uniform grid: H_4 = (h/mu) tridiag(1/12, 5/6, 1/12): the 4th-order compact relation
(tau_{f-1} + 10 tau_f + tau_{f+1})/12 = mu (ubar_{f+1} - ubar_f)/h.

    python -m cases.viscous_stress
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG
from cases.pccd_density import pieces

_x3, _w3 = np.polynomial.legendre.leggauss(3)
X3, W3 = 0.5 * (_x3 + 1.0), 0.5 * _w3


class Setup:
    """Periodic two-phase setup: heavy (rho = 1, mu = mu_H) on the arc a -> b, light (rho = 1/ratio, mu = mu_L)."""

    def __init__(self, N, ratio, a, b, mu_H, mu_L):
        self.N, self.ratio, self.a, self.b, self.mu_H, self.mu_L = N, ratio, a, b, mu_H, mu_L
        h = self.h = 1.0 / N
        self.P = pieces(N, a, b)
        H = np.zeros((N, N))
        m = np.zeros(N)
        for i, ps in enumerate(self.P):
            mi = sum((s1 - s0) * h * (1.0 if hv else 1.0 / ratio) for s0, s1, hv in ps)
            m[i] = mi
            R = 0.0
            rr = rf = rx = 0.0
            for s0, s1, hv in ps:
                rho = 1.0 if hv else 1.0 / ratio
                mu = mu_H if hv else mu_L
                x = (s1 - s0) * h * X3
                w = (s1 - s0) * h * W3
                Rq = R + rho * x
                up = Rq / mi
                dn = 1.0 - up
                rr += w @ (up * up) / mu
                rf += w @ (dn * dn) / mu
                rx += w @ (up * dn) / mu
                R += rho * (s1 - s0) * h
            f, g = i, (i - 1) % N                 # face right of cell i (rising), face left (falling)
            H[f, f] += rr
            H[g, g] += rf
            H[f, g] += rx
            H[g, f] += rx
        self.H, self.m = H, m
        # docs/notes/99 V3 trial: "H3", each row exact for the stresses tau = 1, x, x^2 (about the face), the targets
        # int psi_f p / mu; non-symmetric where rho or mu jump; energy dissipation needs only sym(H3) > 0
        H3 = np.zeros((N, N))
        for f in range(N):
            xf = (f + 1) * h
            t = np.zeros(3)
            for c in (f, f + 1):
                ps = self.P[c % N]
                mc = m[c % N]
                R = 0.0
                for s0, s1, hv in ps:
                    rho = 1.0 if hv else 1.0 / ratio
                    mu = mu_H if hv else mu_L
                    x = (c + s0 + (s1 - s0) * X3) * h
                    w = (s1 - s0) * h * W3
                    Rq = R + rho * (x - (c + s0) * h)
                    psi = Rq / mc if c == f else 1.0 - Rq / mc
                    for p_ in range(3):
                        t[p_] += w @ (psi * ((x - xf) / h) ** p_ / mu)
                    R += rho * (s1 - s0) * h
            V = np.array([[((g - f) * 1.0) ** p_ for g in (f - 1, f, f + 1)] for p_ in range(3)])
            coef = np.linalg.solve(V, t)
            for k, g in enumerate((f - 1, f, f + 1)):
                H3[f, g % N] += coef[k]
        self.H3 = H3
        self.HL = np.diag(H.sum(1))
        self.H4 = 0.5 * (H + self.HL)
        D = np.zeros((N, N))
        for f in range(N):
            D[f, f], D[f, (f + 1) % N] = -1.0, 1.0
        self.D = D
        self.K = D.T @ np.linalg.solve(self.H4, D)

    def rho(self, x):
        return np.where(((x - self.a) % 1.0) < ((self.b - self.a) % 1.0), 1.0, 1.0 / self.ratio)

    def mu(self, x):
        return np.where(((x - self.a) % 1.0) < ((self.b - self.a) % 1.0), self.mu_H, self.mu_L)

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


def mus(ratio, kind):
    """(mu_H, mu_L): "nu" constant kinematic viscosity (mu = rho), "mu" constant dynamic viscosity, "water-air"
    (mu_H / mu_L = 55, with the density ratio given)."""
    return {"nu": (1.0, 1.0 / ratio), "mu": (1.0, 1.0), "water-air": (1.0, 1.0 / 55.0)}[kind]


# --------------------------------------------------------------------------------------------- V0: exact solutions
def _piecewise_integral(f, mu_H, mu_L, a, b, x):
    """int_0^x f(s)/mu(s) ds for f smooth (Gauss on the pieces cut at a, b)."""
    out = np.zeros_like(x)
    for k, xx in enumerate(x):
        cuts = sorted({0.0, xx, *[c for c in (a, b) if 0 < c < xx]})
        tot = 0.0
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            s = lo + (hi - lo) * XG
            mu = np.where((s > a) & (s < b), mu_H, mu_L)
            tot += (hi - lo) * (WG @ (f(s) / mu))
        out[k] = tot
    return out


def steady_profile(mu_H, mu_L, a, b, k=1):
    """A periodic steady two-layer flow: tau = sin 2 pi k x + tau_c (continuous), tau_c such that u is periodic,
    u = int_0^x tau/mu (continuous, its gradient jumps). Returns (u, tau)."""
    base = _piecewise_integral(lambda s: np.sin(2 * np.pi * k * s), mu_H, mu_L, a, b, np.array([1.0]))[0]
    inv = (b - a) / mu_H + (1 - (b - a)) / mu_L
    tc = -base / inv
    tau = lambda x: np.sin(2 * np.pi * k * x) + tc  # noqa: E731

    def u(x):
        x = np.asarray(x, dtype=float)
        xm = x % 1.0
        # cumulative integral by sorting
        return _piecewise_integral(tau, mu_H, mu_L, a, b, xm)
    return u, tau


def steady_error(N, ratio, kind, a=0.2613, b=0.7137, H="H4"):
    """K ubar = D^T tau_faces (the steady forced problem): max error of the averages relative to max |ubar|, the
    total momentum fixed."""
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    u, tau = steady_profile(mu_H, mu_L, a, b)
    ub = S.averages(u)
    tf = tau((np.arange(N) + 1) * S.h)
    Hm = {"H4": S.H4, "H": S.H, "H3": S.H3}[H]
    K = S.D.T @ np.linalg.solve(Hm, S.D)
    rhs = S.D.T @ tf
    # K is singular (constants): add the mass constraint
    A = np.block([[K, S.m[:, None]], [S.m[None, :], np.zeros((1, 1))]])
    sol = np.linalg.solve(A, np.concatenate([rhs, [S.m @ ub]]))
    return float(np.abs(sol[:-1] - ub).max() / np.abs(ub).max())        # relative to max |ubar|


def dirichlet_count(mu_H, mu_L, ratio, LH, LL, lam):
    """Sturm's count (audit #126): the number of zeros in the open interval (0, LH + LL) of the solution of
    (mu u')' + lam rho u = 0 with u(0) = 0 (layer H of length LH first, then layer L of length LL; u and mu u'
    continuous). It equals the number of eigenvalues below lam of the problem with u = 0 at both ends. The zeros
    are counted from the phases in each layer (no scan)."""
    kH, kL = math.sqrt(lam / mu_H), math.sqrt(lam / ratio / mu_L)
    nH = math.floor(kH * LH / math.pi)                           # zeros of sin(kH x) in (0, LH]
    A, B = math.sin(kH * LH) / (mu_H * kH), math.cos(kH * LH)    # u and mu u' at the interface
    th = math.atan2(B / (mu_L * kL), A)                          # layer L: u = R cos(kL s - th), s in (0, LL)
    lo, hi = (-th - math.pi / 2) / math.pi, (kL * LL - th - math.pi / 2) / math.pi
    return nH + max(0, math.ceil(hi) - math.floor(lo) - 1)       # integers m with lo < m < hi


def periodic_count(mu_H, mu_L, ratio, LH, LL, lam, discriminant):
    """The number of positive periodic rates (with multiplicity) up to lam, from Sturm's count of the Dirichlet
    problem on one period and the sign of ``discriminant`` = tr(Phi) - 2 at lam (audit #126). The n-th Dirichlet
    eigenvalue lies in the n-th gap of Hill's problem; the even gaps are bounded by the periodic pairs
    (lambda_{2m-1}, lambda_{2m}), where the discriminant is positive."""
    nD = dirichlet_count(mu_H, mu_L, ratio, LH, LL, lam)
    return 2 * math.ceil(nD / 2) - 1 if discriminant > 0 else 2 * (nD // 2)


def count_guided_roots(f, count, lo, hi, depth=0):
    """All the roots (in s = sqrt(lambda), with multiplicity) of f(s^2) in (lo, hi], guided by ``count(s)`` = the
    number of roots up to s (Sturm): an interval is halved until it holds one root (a sign change: bisection) or is
    of round-off width (a multiple root). Unlike a sign-change scan it cannot drop a close pair (audit #126)."""
    n = count(hi) - count(lo)
    if n <= 0:
        return []
    if n == 1:
        flo = f(lo * lo)
        if flo * f(hi * hi) < 0:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid * mid)
                if fm * flo > 0:
                    lo, flo = mid, fm
                else:
                    hi = mid
            return [0.5 * (lo + hi)]
    if hi - lo < 1e-13 * hi or depth > 70:
        return [0.5 * (lo + hi)] * n
    mid = 0.5 * (lo + hi)
    return count_guided_roots(f, count, lo, mid, depth + 1) + count_guided_roots(f, count, mid, hi, depth + 1)


def stokes_rate(mu_H, mu_L, ratio, a, b, nu_guess=None, which=1):
    """The exact decay rates lambda of the periodic two-layer Stokes problem rho u_t = (mu u_x)_x, in increasing order:
    tr(Phi_L Phi_H) = 2 with the transfer matrices [[cos kL, sin kL /(mu k)], [-mu k sin kL, cos kL]],
    k = sqrt(lambda rho / mu). docs/notes/100 correction (the user's review): the scan is uniform in sqrt(lambda) with
    a phase step of at most 0.02 rad in either layer (the earlier coarse scan skipped the lowest rates), sign changes
    and touching zeros (double rates) are both refined; double rates are returned twice. Audit #126: a sign-change
    scan silently drops two rates that fall inside one step, so the number found is checked against Sturm's count
    (``periodic_count``); on a mismatch the rates are found again with ``count_guided_roots``."""
    LH = (b - a) % 1.0
    LL = 1.0 - LH
    cH, cL = math.sqrt(1.0 / mu_H) * LH, math.sqrt(1.0 / ratio / mu_L) * LL     # phase = sqrt(lambda) * c

    def f(lam):
        def phi(L, rho, mu):
            k = math.sqrt(lam * rho / mu)
            return np.array([[math.cos(k * L), math.sin(k * L) / (mu * k)],
                             [-mu * k * math.sin(k * L), math.cos(k * L)]])
        return float(np.trace(phi(LL, 1.0 / ratio, mu_L) @ phi(LH, 1.0, mu_H)) - 2.0)
    ds = 0.02 / max(cH, cL)
    roots = []
    sq, fprev, fprev2 = ds, f(ds * ds), None
    while len(roots) < which:
        sq2 = sq + ds
        f2 = f(sq2 * sq2)
        if fprev * f2 < 0:                                   # a simple root: bisection in sqrt(lambda)
            lo, hi, flo = sq, sq2, fprev
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid * mid)
                if fm * flo > 0:
                    lo, flo = mid, fm
                else:
                    hi = mid
            roots.append((0.5 * (lo + hi)) ** 2)
        elif fprev2 is not None and abs(fprev) < abs(fprev2) and abs(fprev) <= abs(f2) and abs(fprev) < 1e-3:
            # a touching zero (a double rate): |f| ~ (lambda - r)^2 fixes r only to sqrt(eps); bisect the derivative
            g = lambda z: (f((z * (1 + 1e-7)) ** 2) - f((z * (1 - 1e-7)) ** 2))  # noqa: E731
            lo, hi = sq - ds, sq2
            if g(lo) * g(hi) < 0:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if g(mid) * g(lo) > 0:
                        lo = mid
                    else:
                        hi = mid
                r = (0.5 * (lo + hi)) ** 2
                if abs(f(r)) < 1e-9:
                    roots += [r, r]
        fprev2, fprev, sq = fprev, f2, sq2
    count = lambda z: periodic_count(mu_H, mu_L, ratio, LH, LL, z * z, f(z * z))  # noqa: E731
    if count(sq) != len(roots):                                  # audit #126: two rates inside one scan step
        roots = [z * z for z in count_guided_roots(f, count, 1e-3 * ds, sq)]
        if count(sq) != len(roots):
            raise RuntimeError("stokes_rate: found %d rates up to lambda = %.6g, Sturm's count is %d"
                               % (len(roots), sq * sq, count(sq)))
    return roots[:which]


def stokes_error(N, ratio, kind, a=0.2613, b=0.7137, nmodes=2, H="H4"):
    """The nmodes slowest exact rates (with multiplicity) against the nmodes smallest nonzero discrete rates of
    M u' = -K u (generalised eigenproblem), sorted; for non-symmetric K the imaginary parts are reported."""
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    K = S.K if H == "H4" else S.D.T @ np.linalg.solve({"H": S.H, "H3": S.H3}[H], S.D)
    ex = stokes_rate(mu_H, mu_L, ratio, a, b, which=nmodes)
    Mh = 1.0 / np.sqrt(S.m)
    A = K * Mh[:, None] * Mh[None, :]
    if H == "H4" or H == "H":
        ev = np.sort(np.linalg.eigvalsh(A)).astype(complex)
    else:
        ev = np.linalg.eigvals(A)
        ev = ev[np.argsort(ev.real)]
    ev = ev[np.abs(ev) > 1e-9 * np.abs(ev).max()][:nmodes]
    return {"exact": ex, "discrete": [float(e.real) for e in ev], "imag": [float(e.imag) for e in ev],
            "rel_err": [float(abs(e - x) / x) for e, x in zip(ev, ex)]}


# --------------------------------------------------------------------------------------------- V1: structure
def structure(N, ratio, kind, a=0.2613, b=0.7137):
    mu_H, mu_L = mus(ratio, kind)
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    one = np.ones(N)
    # constant stress tau0 on an open stretch: u' = tau0/mu (piecewise linear, kinks at a and b); the faces whose
    # two cells lie in [0, 1) without wrapping: H_4 tau0 1 = D ubar exactly
    tau0 = 0.7
    u = lambda x: _piecewise_integral(lambda s: tau0 + 0 * s, mu_H, mu_L, a, b, np.asarray(x) % 1.0)  # noqa: E731
    ub = S.averages(u)
    r = S.H4 @ (tau0 * one) - S.D @ ub
    ev = np.linalg.eigvalsh(S.K)
    Mh = 1.0 / np.sqrt(S.m)
    nu = max(mu_H, mu_L * ratio)
    stiff = float(np.linalg.eigvalsh(S.K * Mh[:, None] * Mh[None, :])[-1] * S.h ** 2 / nu)
    K3 = S.D.T @ np.linalg.solve(S.H3, S.D)
    sym3 = np.linalg.eigvalsh(0.5 * (K3 + K3.T))
    return {"K3_sym_min_rel": float(sym3[0] / sym3[-1]),
            "const_stress_residual": float(np.abs(r[:-1]).max()), "momentum": float(np.abs(one @ S.K).max()),
            "K_min_eig_rel": float(ev[0] / ev[-1]), "stiffness_h2_over_nu": stiff,
            "H4_rowsum_vs_H": float(np.abs(S.H4.sum(1) - S.H.sum(1)).max())}


def stiffness_scan(N=32, ratios=(1.0, 1e3, 1e6), cs=(1e-8, 1e-6, 1e-4, 1e-2, 0.5, 0.99)):
    out = []
    for r in ratios:
        for c in cs:
            h = 1.0 / N
            a = (N // 4 + 1 - c) * h
            st = structure(N, r, "nu", a=a)
            out.append({"ratio": r, "c": c, **st})
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    res = {"structure": [], "steady": [], "stokes": []}
    with ProcessPoolExecutor(24) as ex:
        fs = [(k, ex.submit(structure, 32, r, k)) for k in ("nu", "mu", "water-air") for r in (1.0, 1e3)]
        fsc = ex.submit(stiffness_scan)
        fst = [((N, r, k, Hn), ex.submit(steady_error, N, r, k, H=Hn)) for k in ("nu", "mu", "water-air")
               for r in (1.0, 1e3) for N in (32, 64, 128, 256) for Hn in ("H4", "H", "H3")]
        fsk = [((N, r, k, Hn), ex.submit(stokes_error, N, r, k, H=Hn)) for k in ("nu", "mu") for r in (1.0, 1e3)
               for N in (32, 64, 128, 256) for Hn in ("H4", "H3")]
        for k, f in fs:
            res["structure"].append({"kind": k, **f.result()})
        res["stiffness_scan"] = fsc.result()
        for (N, r, k, Hn), f in fst:
            res["steady"].append({"N": N, "ratio": r, "kind": k, "H": Hn, "err": f.result()})
        for (N, r, k, Hn), f in fsk:
            try:
                res["stokes"].append({"N": N, "ratio": r, "kind": k, "H": Hn, **f.result()})
            except Exception as e:  # noqa: BLE001
                res["stokes"].append({"N": N, "ratio": r, "kind": k, "H": Hn, "error": repr(e)})
    json.dump(res, open("docs/figures/viscous_stress.json", "w"), indent=1, default=float)
    for r in res["steady"]:
        print("steady", r)
    for r in res["stokes"]:
        print("stokes", {k: v for k, v in r.items() if k != "discrete"})
    print("stiffness", max(x["stiffness_h2_over_nu"] for x in res["stiffness_scan"]))


if __name__ == "__main__":
    main()

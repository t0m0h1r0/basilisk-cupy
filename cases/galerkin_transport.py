"""P0-4b-3 (docs/notes/80): the user's geometry-consistent Galerkin transport, G^{n+1} u^{n+1} = C u^n, in 1-D with
a discontinuous density (ratio 1000) and an exact interface, local polynomial reconstructions (no CCD yet).

Periodic [0, 1), N cells, a heavy band rho = 1 on (0.213, 0.647) and 0.001 elsewhere, carried with U = 1 together with
q = sin 2 pi x (exact: rho(x - t), q(x - t)). The unknowns are the mass-weighted cell averages u_i = int_{K_i} rho q
/ m_i. Reconstruction: on cell i the polynomial of degree p whose mass-weighted averages over the stencil's cells are
the stencil's u (cubic: cells i-1 .. i+2; quartic: i-2 .. i+2), hence a basis phi_j with sum_j phi_j = 1 and
int_{K_i} rho phi_j = m_i delta_ij. Then

    G_ij = int rho^{n+1} phi_i^{n+1} phi_j^{n+1},    C_ij = int rho^n(x) phi_i^{n+1}(x + U dt) phi_j^n(x) dx,

integrated exactly (Gauss on the pieces between the cell boundaries before and after the move and the interfaces).

    python -m cases.galerkin_transport
"""

from __future__ import annotations

import math

import numpy as np

A0, B0, RHO_IN, RHO_OUT = 0.213, 0.647, 1.0, 1e-3
STENCILS = {3: (-1, 0, 1, 2), 4: (-2, -1, 0, 1, 2)}


def rho_at(x, t):
    s = np.mod(x - t, 1.0)
    return np.where((s > A0) & (s < B0), RHO_IN, RHO_OUT)


def interfaces(t):
    return [np.mod(A0 + t, 1.0), np.mod(B0 + t, 1.0)]


def _pieces(lo, hi, cuts):
    pts = sorted({lo, hi, *[c for c in cuts if lo < c < hi]})
    return list(zip(pts[:-1], pts[1:]))


def _integrate(f, lo, hi, cuts, order):
    """int_lo^hi f(x) dx with f smooth between the cuts (Gauss-Legendre, ``order`` points per piece)."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    s = 0.0
    for a, b in _pieces(lo, hi, cuts):
        x = 0.5 * (b - a) * xg + 0.5 * (a + b)
        s = s + 0.5 * (b - a) * np.tensordot(wg, f(x), axes=(0, 0))
    return s


class Reconstruction:
    """The basis at time t: on cell i, u_h(x) = sum_k coef[i][r, k] xi^r u_{i + o_k}, xi = (x - x_i)/h."""

    def __init__(self, n: int, p: int, t: float):
        self.n, self.p, self.t = n, p, t
        self.h = 1.0 / n
        self.offsets = STENCILS[p]
        cuts = interfaces(t)
        self.mass = np.array([_integrate(lambda x: rho_at(x, t), k * self.h, (k + 1) * self.h,
                                         cuts, 4) for k in range(n)])
        self.coef = []
        for i in range(n):
            xi0 = (i + 0.5) * self.h
            A = np.zeros((len(self.offsets), p + 1))
            for kk, o in enumerate(self.offsets):
                c = i + o  # unwrapped cell index: x in [c h, (c + 1) h], density from the periodic profile
                mom = _integrate(lambda x: rho_at(x, t)[:, None] * ((x[:, None] - xi0) / self.h) ** np.arange(p + 1),
                                 c * self.h, (c + 1) * self.h, [z + m for z in cuts for m in (-1, 0, 1)], p + 2)
                A[kk] = mom / self.mass[c % n]
            self.coef.append(np.linalg.solve(A, np.eye(len(self.offsets))))  # (p + 1, stencil)

    def basis(self, x):
        """For points x (unwrapped, any period): the cell of each, and the values of its stencil's basis functions."""
        cell = np.floor(x / self.h).astype(int)
        xi = (x - (cell + 0.5) * self.h) / self.h
        vals = np.empty((x.size, len(self.offsets)))
        for k in range(x.size):
            vals[k] = (xi[k] ** np.arange(self.p + 1)) @ self.coef[cell[k] % self.n]
        idx = (cell[:, None] + np.array(self.offsets)[None, :]) % self.n
        return idx, vals


def matrices(n: int, p: int, t: float, dt: float, U: float = 1.0):
    """G^{n+1} and C for one step from t to t + dt, and the reconstructions."""
    old, new = Reconstruction(n, p, t), Reconstruction(n, p, t + dt)
    h = 1.0 / n
    order = 2 * p + 2
    xg, wg = np.polynomial.legendre.leggauss(order)
    cut_new = [k * h for k in range(n + 2)] + [z + m for z in interfaces(t + dt) for m in (-1, 0, 1)]
    cut_old = [k * h - U * dt for k in range(n + 2)] + [z + m for z in interfaces(t) for m in (-1, 0, 1)]
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        for a, b in _pieces(i * h, (i + 1) * h, cut_new):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, t + dt)
            idx, v = new.basis(x)
            for q in range(order):
                G[np.ix_(idx[q], idx[q])] += w[q] * np.outer(v[q], v[q])
        for a, b in _pieces(i * h, (i + 1) * h, cut_old):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, t)
            io, vo = old.basis(x)
            inew, vn = new.basis(x + U * dt)
            for q in range(order):
                C[np.ix_(inew[q], io[q])] += w[q] * np.outer(vn[q], vo[q])
    return G, C, old, new


def profile(s, kind: str = "smooth"):
    """The transported q at the co-moving coordinate s: "smooth" sin 2 pi s; "kink": + 3 (s - A0)(B0 - s) inside the
    band (continuous, its gradient jumps at both interfaces)."""
    s = np.mod(s, 1.0)
    q = np.sin(2 * np.pi * s)
    if kind == "kink":
        q = q + np.where((s > A0) & (s < B0), 3.0 * (s - A0) * (B0 - s), 0.0)
    return q


def exact_averages(n: int, t: float, mass, kind: str = "smooth"):
    h = 1.0 / n
    cuts = [z + m for z in interfaces(t) for m in (-1, 0, 1)]
    return np.array([_integrate(lambda x: rho_at(x, t) * profile(x - t, kind), k * h, (k + 1) * h, cuts, 8)
                     for k in range(n)]) / mass


def one_step(n: int, p: int, cfl: float = 0.37):
    dt = cfl / n
    G, C, old, new = matrices(n, p, 0.0, dt)
    u0 = exact_averages(n, 0.0, old.mass)
    u1 = np.linalg.solve(G, C @ u0)
    err = float(np.abs(u1 - exact_averages(n, dt, new.mass)).max())
    # structure: free stream, total momentum, the energy-weighted amplification with G^n
    G0, _, _, _ = matrices(n, p, -dt, dt)  # G at time 0 (the step from -dt to 0)
    ones = np.ones(n)
    fs = float(np.abs(np.linalg.solve(G, C @ ones) - 1.0).max())
    mom = abs(new.mass @ u1 - old.mass @ u0)
    w0, V0 = np.linalg.eigh(G0)
    w1, V1 = np.linalg.eigh(G)
    G0h = V0 @ np.diag(np.sqrt(w0)) @ V0.T
    G0ih = V0 @ np.diag(1 / np.sqrt(w0)) @ V0.T
    G1h = V1 @ np.diag(np.sqrt(w1)) @ V1.T
    amp = np.linalg.norm(G1h @ np.linalg.solve(G, C) @ G0ih, 2)
    E0, E1 = u0 @ G0 @ u0, u1 @ G @ u1
    return {"error": err, "free_stream": fs, "momentum": float(mom), "amplification": float(amp),
            "dE": float(E1 / E0 - 1), "min_eig_G": float(w1.min())}


def run(n: int, p: int, t_end: float = 0.137, cfl: float = 0.37, kind: str = "smooth"):
    steps = int(math.ceil(t_end / (cfl / n)))
    dt = t_end / steps
    u = None
    for s in range(steps):
        G, C, old, new = matrices(n, p, s * dt, dt)
        if u is None:
            u = exact_averages(n, 0.0, old.mass, kind)
        u = np.linalg.solve(G, C @ u)
    return float(np.abs(u - exact_averages(n, t_end, new.mass, kind)).max()), steps


def locality(n: int, p: int, cfl: float = 0.37):
    """P0-4b-3c: the largest |(G^-1 C)_ij| at each cell distance |i - j| (periodic), and the face momentum fluxes the
    step implies, P_{i+1/2} - P_{i-1/2} = -(m^{n+1}_i u^{n+1}_i - m^n_i u^n_i) for every u^n = e_j."""
    dt = cfl / n
    G, C, old, new = matrices(n, p, 0.0, dt)
    T = np.linalg.solve(G, C)
    d = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)])
    return [float(np.abs(T[d == k]).max()) for k in range(n // 2 + 1)]


def main() -> None:
    ns = (16, 32, 64, 128)
    print("-- one step, CFL 0.37: max error of the mass-weighted cell averages; structure")
    for p in (3, 4):
        rs = [one_step(n, p) for n in ns]
        print(f"   degree {p}: " + " ".join(f"{r['error']:.2e}" for r in rs) + "; free stream "
              f"{max(r['free_stream'] for r in rs):.1e}, momentum {max(r['momentum'] for r in rs):.1e}, amplification "
              f"{max(r['amplification'] for r in rs):.12f}, dE/E {min(r['dE'] for r in rs):.1e} .. "
              f"{max(r['dE'] for r in rs):.1e}, min eig G {min(r['min_eig_G'] for r in rs):.1e}", flush=True)
    for kind in ("smooth", "kink"):
        print(f"-- t = 0.137 (no end-of-period cancellation), CFL 0.37, q {kind}")
        for p in (3, 4):
            res = [run(n, p, kind=kind) for n in ns]
            e = [r[0] for r in res]
            print(f"   degree {p}: " + " ".join(f"{x:.2e} ({r[1]} steps)" for x, r in zip(e, res))
                  + "  orders " + " ".join(f"{math.log2(e[k] / e[k + 1]):.2f}" for k in range(len(e) - 1)), flush=True)
    print("-- P0-4b-3c: max |(G^-1 C)_ij| against the cell distance |i - j| (N = 64, CFL 0.37)")
    for p in (3, 4):
        loc = locality(64, p)
        print(f"   degree {p}: " + " ".join(f"{k}:{v:.1e}" for k, v in enumerate(loc[:13])) + f" ... 32:{loc[32]:.1e}",
              flush=True)


if __name__ == "__main__":
    main()

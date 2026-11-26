"""P0-3 / P0-4 (docs/notes/78): the energy-consistent implicit momentum transport of the user's design, tested on small
1-D periodic matrices, with no flow solver and no VOF.

Cell masses m^n > 0, face mass fluxes M_f over one step (face f = i + 1/2), m^{n+1} = m^n - (M_{i+1/2} - M_{i-1/2}).
With the square-root-mass weighted time average

    w_i = (sqrt(m_i^{n+1}) u_i^{n+1} + sqrt(m_i^n) u_i^n) / (sqrt(m_i^{n+1}) + sqrt(m_i^n)),

the exact identity  Delta E_i = w_i Delta p_i - |w_i|^2 Delta m_i / 2  holds, and a pair flux of momentum
P_ij = M_ij (w_i + w_j)/2 - |M_ij| lambda_ij (w_j - w_i)/2 between cells i and j (M_ij = -M_ji the mass that pair
moves from i to j) gives  Delta E = -1/2 sum_pairs |M_ij| lambda_ij |w_j - w_i|^2.  The step solves

    (B_+ + L) w = B_- u^n,   B_+ = diag(m^{n+1} + sqrt(m^{n+1} m^n)),   B_- = diag(m^n + sqrt(m^{n+1} m^n)),

with (L w)_i = sum_j P_ij, then u^{n+1} from w.

Pair structures: ``pair2`` (the faces' own flux between neighbours), ``pair4`` (the user's redistribution: a pair flux
between cells two apart, b_i = -(M_{i+1/2} + M_{i+3/2})/12, the neighbours' pair fluxes adjusted so that the mass
crossing every face is still M_f: 4/3 and -1/6 of a constant flux, the fourth-order centred difference).

    python -m cases.energy_transport [--semi | --shear]    # P0-4a (the space error alone, variable density) /
                                                           # P0-4b (deform and return: the discrete energy)
"""

from __future__ import annotations

import math

import numpy as np


def pairs(M, structure: str):
    """Pair mass transfers (i, j, M_ij) on the periodic ring whose net flux through every face is ``M``."""
    n = M.size
    out = []
    if structure == "pair2":
        for i in range(n):
            out.append((i, (i + 1) % n, M[i]))
        return out
    if structure != "pair4":
        raise ValueError(structure)
    b = -(M + np.roll(M, -1)) / 12.0  # pair (i, i + 2), crossing faces i + 1/2 and i + 3/2
    for i in range(n):
        # face i + 1/2 is crossed by (i, i + 1), (i - 1, i + 1) and (i, i + 2)
        a = M[i] - b[(i - 1) % n] - b[i]
        out.append((i, (i + 1) % n, a))
        out.append((i, (i + 2) % n, b[i]))
    return out


def step(m0, M, u0, structure: str = "pair2", lam=0.0):
    """One step: returns m1, u1, w and the matrix of the step."""
    n = m0.size
    m1 = m0 - (M - np.roll(M, 1))
    if np.any(m1 <= 0):
        raise ValueError("a cell mass became non-positive")
    s = np.sqrt(m1 * m0)
    Bp, Bm = np.diag(m1 + s), np.diag(m0 + s)
    L = np.zeros((n, n))
    for k, (i, j, Mij) in enumerate(pairs(M, structure)):
        lij = lam if np.isscalar(lam) else lam[k]
        # P_ij = Mij (w_i + w_j)/2 - |Mij| lij (w_j - w_i)/2 leaves i and enters j
        ci, cj = 0.5 * Mij + 0.5 * abs(Mij) * lij, 0.5 * Mij - 0.5 * abs(Mij) * lij
        L[i, i] += ci
        L[i, j] += cj
        L[j, i] -= ci
        L[j, j] -= cj
    A = Bp + L
    w = np.linalg.solve(A, Bm @ u0)
    u1 = (w * (np.sqrt(m1) + np.sqrt(m0)) - np.sqrt(m0) * u0) / np.sqrt(m1)
    R = np.diag((np.sqrt(m1) + np.sqrt(m0)) / np.sqrt(m1)) @ np.linalg.solve(A, Bm) - np.diag(np.sqrt(m0 / m1))
    return m1, u1, w, R, A


def identities(seed: int, n: int = 24, structure: str = "pair2", lam_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    m0 = rng.uniform(0.5, 2.0, n) * (10.0 ** rng.uniform(-3, 0, n))  # up to a 1000:1 spread (two-phase cells)
    M = rng.normal(size=n) * 0.2 * np.minimum(m0, np.roll(m0, -1))
    u0 = rng.normal(size=n)
    pr = pairs(M, structure)
    lam = rng.uniform(0.0, lam_scale, len(pr))
    m1, u1, w, R, A = step(m0, M, u0, structure, lam)
    E0, E1 = 0.5 * np.sum(m0 * u0 ** 2), 0.5 * np.sum(m1 * u1 ** 2)
    dE_pred = -0.5 * sum(abs(Mij) * lk * (w[j] - w[i]) ** 2 for (i, j, Mij), lk in zip(pr, lam))
    sym = 0.5 * (A + A.T)
    # free stream
    _, uc, _, _, _ = step(m0, M, np.full(n, 0.7), structure, lam)
    amp = np.linalg.norm(np.diag(np.sqrt(m1)) @ R @ np.diag(1 / np.sqrt(m0)), 2)
    return {"momentum": abs(np.sum(m1 * u1) - np.sum(m0 * u0)) / np.sum(np.abs(m0 * u0)),
            "energy_balance": abs((E1 - E0) - dE_pred) / E0, "dE": (E1 - E0) / E0,
            "free_stream": float(np.abs(uc - 0.7).max()), "sym_min_eig": float(np.linalg.eigvalsh(sym).min()),
            "amplification": float(amp)}


def accuracy(structure: str, n: int, cfl: float, lam: float = 0.0, t_end: float = 1.0):
    """u_t + U u_x = 0 (U = 1) with constant density: m = h, M = U dt (a mass flux of density 1); u0 = sin 2 pi x as
    cell averages; the error at t = 1 (one period)."""
    h = 1.0 / n
    steps = int(math.ceil(t_end / (cfl * h)))
    dt = t_end / steps
    x = np.arange(n) * h
    u = (np.cos(2 * np.pi * x) - np.cos(2 * np.pi * (x + h))) / (2 * np.pi * h)
    u0 = u.copy()
    m = np.full(n, h)
    M = np.full(n, dt)
    _, _, _, R, _ = step(m, M, u, structure, lam)
    for _ in range(steps):
        u = R @ u
    return float(np.abs(u - u0).max())


CASES = {
    # rho(x, t), face mass flux rate F(x, t), transported q(x, t): exact solutions of (rho)_t + F_x = 0 and
    # (rho q)_t + (F q)_x = 0 on the periodic [0, 1)
    "constant": (lambda x, t: np.ones_like(x), lambda x, t: np.ones_like(x),
                 lambda x, t: np.sin(2 * np.pi * (x - t))),
    # a density wave carried by U = 1: F = rho varies in space and time (the discrete product rule)
    "density wave": (lambda x, t: 1.0 + 0.5 * np.sin(2 * np.pi * (x - t)),
                     lambda x, t: 1.0 + 0.5 * np.sin(2 * np.pi * (x - t)),
                     lambda x, t: np.sin(2 * np.pi * (x - t))),
}


def _steady_case(rho0=lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x), F=1.0):
    """Steady rho(x), F constant, U = F / rho(x): q is carried along the travel time tau(x) = int_0^x rho / F."""
    xs = np.linspace(0.0, 1.0, 200001)
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (rho0(xs[1:]) + rho0(xs[:-1])) * np.diff(xs))]) / F
    period = tau[-1]

    def q(x, t):
        tx = np.interp(np.mod(x, 1.0), xs, tau)
        return np.sin(2 * np.pi * (tx - t) / period)

    return (lambda x, t: rho0(x), lambda x, t: F + 0.0 * x, q)


CASES["steady variable velocity"] = _steady_case()


def _cell_integrals(f, n, t, order: int = 8):
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(order)
    c = (np.arange(n) + 0.5) * h
    return sum(0.5 * h * w * f(c + 0.5 * h * a, t) for a, w in zip(xg, wg))


def semi_discrete(structure: str, n: int, case: str, t_end: float = 1.0, cfl: float = 0.02):
    """The pair transport without its time discretisation: d(m q)/dt = -sum_j P_ij, P_ij = M_ij (q_i + q_j)/2 with the
    exact cell masses m(t) and face mass-flux rates M(t) (consistent: dm/dt = -D M exactly), integrated by classical
    RK4 at a tiny step; max error of the mass-weighted cell averages int rho q / int rho at t_end."""
    rho, F, q = CASES[case]
    h = 1.0 / n
    xf = np.arange(n) * h + h  # face i + 1/2

    def mass(t):
        return _cell_integrals(rho, n, t)

    def rhs(t, y):
        m = mass(t)
        qq = y / m
        out = np.zeros(n)
        for i, j, Mij in pairs(F(xf, t), structure):
            P = Mij * 0.5 * (qq[i] + qq[j])
            out[i] -= P
            out[j] += P
        return out

    y = _cell_integrals(lambda x, t: rho(x, t) * q(x, t), n, 0.0)
    steps = int(math.ceil(t_end / (cfl * h)))
    dt = t_end / steps
    t = 0.0
    for _ in range(steps):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    exact = _cell_integrals(lambda x, s: rho(x, s) * q(x, s), n, t_end) / mass(t_end)
    return float(np.abs(y / mass(t_end) - exact).max())


def main() -> None:
    print("-- identities (random masses with a 1000:1 spread, random fluxes and velocities, n = 24, 20 seeds): max")
    for st in ("pair2", "pair4"):
        for ls in (0.0, 1.0):
            rs = [identities(s, structure=st, lam_scale=ls) for s in range(20)]
            print(f"   {st} lambda in [0, {ls:g}]: momentum {max(r['momentum'] for r in rs):.1e}, energy balance "
                  f"{max(r['energy_balance'] for r in rs):.1e}, dE/E in [{min(r['dE'] for r in rs):.1e}, "
                  f"{max(r['dE'] for r in rs):.1e}], free stream {max(r['free_stream'] for r in rs):.1e}, "
                  f"min eig of (A + A^T)/2 {min(r['sym_min_eig'] for r in rs):.2e}, mass-weighted amplification "
                  f"{max(r['amplification'] for r in rs):.12f}", flush=True)
    print("-- accuracy: one period of sin 2 pi x at U = 1, constant density (max error)")
    for cfl in (0.5, 0.05):
        for st in ("pair2", "pair4"):
            errs = [accuracy(st, n, cfl) for n in (16, 32, 64, 128)]
            orders = [math.log2(errs[k] / errs[k + 1]) for k in range(3)]
            print(f"   CFL {cfl:4.2f} {st}: " + " ".join(f"{e:.2e}" for e in errs) + "  order "
                  + " ".join(f"{o:.2f}" for o in orders), flush=True)


def shear_return(structure: str, n: int, A: float = 0.25, cfl: float = 0.02, samples: int = 8):
    """P0-4b: u_x = a'(t) sin 2 pi y, a = A sin^2 (pi t) (deformed most at t = 1/2, back at t = 1), constant density;
    q = sin 2 pi (x - a sin 2 pi y) exactly. The pair transport row by row along x (face mass-flux rate of row j:
    a'(t) times the row's mean of sin 2 pi y) conserves E_h = 1/2 sum m qbar^2 exactly; the exact cell averages'
    E_h dips and returns. Returns the times, the exact and the scheme's E_h / E_h(0), and the max errors at 1/2, 1."""
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(8)
    c = (np.arange(n) + 0.5) * h
    srow = sum(0.5 * w * np.sin(2 * np.pi * (c + 0.5 * h * b)) for b, w in zip(xg, wg))  # row means of sin 2 pi y

    def exact(t):
        a = A * math.sin(math.pi * t) ** 2
        out = np.zeros((n, n))
        for p_, wp in zip(xg, wg):
            for q_, wq in zip(xg, wg):
                X, Y = np.meshgrid(c + 0.5 * h * p_, c + 0.5 * h * q_, indexing="ij")
                out += 0.25 * wp * wq * np.sin(2 * np.pi * (X - a * np.sin(2 * np.pi * Y)))
        return out

    plist = None

    def rhs(t, Q):
        ad = A * math.pi * math.sin(2 * math.pi * t)
        out = np.zeros_like(Q)
        for j in range(n):
            M = np.full(n, ad * srow[j] * h)  # mass-flux rate through the row's x-faces (density 1, face length h)
            for i, k, Mij in pairs(M, structure):
                P = Mij * 0.5 * (Q[i, j] + Q[k, j])
                out[i, j] -= P / (h * h)
                out[k, j] += P / (h * h)
        return out

    Q = exact(0.0)
    E0 = 0.5 * h * h * np.sum(Q ** 2)
    steps = int(math.ceil(1.0 / (cfl * h)))
    dt = 1.0 / steps
    t = 0.0
    rec, errs = [], {}
    for s in range(steps):
        k1 = rhs(t, Q)
        k2 = rhs(t + dt / 2, Q + dt / 2 * k1)
        k3 = rhs(t + dt / 2, Q + dt / 2 * k2)
        k4 = rhs(t + dt, Q + dt * k3)
        Q = Q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if (s + 1) % (steps // samples) == 0:
            ex = exact(t)
            rec.append((t, 0.5 * h * h * np.sum(ex ** 2) / E0, 0.5 * h * h * np.sum(Q ** 2) / E0))
            if abs(t - 0.5) < 1e-9 or abs(t - 1.0) < 1e-9:
                errs[round(t, 3)] = float(np.abs(Q - ex).max())
    return rec, errs


def main_semi() -> None:
    print("-- P0-4a: the space error alone (semi-discrete, RK4 at CFL 0.02), t = 1, max error")
    for case in CASES:
        for st in ("pair2", "pair4"):
            errs = [semi_discrete(st, n, case) for n in (16, 32, 64, 128)]
            orders = [math.log2(errs[k] / errs[k + 1]) for k in range(3)]
            print(f"   {case:24s} {st}: " + " ".join(f"{e:.2e}" for e in errs) + "  order "
                  + " ".join(f"{o:.2f}" for o in orders), flush=True)


def exact_energies(n: int, t: float, A: float = 0.25):
    """E_2 = 1/2 sum m qbar^2 and E_4 = E_2 + 1/2 sum m h^2/12 |D qbar|^2 (D spectral, standing in for CCD) of the
    exact cell averages of the deform-and-return solution at time t."""
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(8)
    c = (np.arange(n) + 0.5) * h
    a = A * math.sin(math.pi * t) ** 2
    Q = np.zeros((n, n))
    for p_, wp in zip(xg, wg):
        for q_, wq in zip(xg, wg):
            X, Y = np.meshgrid(c + 0.5 * h * p_, c + 0.5 * h * q_, indexing="ij")
            Q += 0.25 * wp * wq * np.sin(2 * np.pi * (X - a * np.sin(2 * np.pi * Y)))
    k = np.fft.fftfreq(n, d=h) * 2 * np.pi
    F = np.fft.fft2(Q)
    dx, dy = np.real(np.fft.ifft2(1j * k[:, None] * F)), np.real(np.fft.ifft2(1j * k[None, :] * F))
    E2 = 0.5 * h * h * np.sum(Q ** 2)
    return E2, E2 + 0.5 * h ** 4 / 12.0 * np.sum(dx ** 2 + dy ** 2)


def main_shear() -> None:
    print("-- P0-4b: the exact cell averages' energies, relative change from t = 0 to t = 1/2")
    for n in (16, 32, 64, 128):
        e0, e1 = exact_energies(n, 0.0), exact_energies(n, 0.5)
        print(f"   N {n:3d}: E_2 {e1[0] / e0[0] - 1:+.2e}, E_4 {e1[1] / e0[1] - 1:+.2e}", flush=True)
    print("-- P0-4b: deform and return (A = 0.25), row-wise pair transport, RK4 at CFL 0.02")
    for st in ("pair2", "pair4"):
        for n in (16, 32, 64):
            rec, errs = shear_return(st, n)
            dip = min(r[1] for r in rec)
            print(f"   {st} N {n:3d}: exact E_h min {dip:.6f} (1 - {1 - dip:.2e}), scheme E_h max |1 - E| "
                  f"{max(abs(1 - r[2]) for r in rec):.1e}; max error t = 1/2 {errs[0.5]:.2e}, t = 1 {errs[1.0]:.2e}",
                  flush=True)


if __name__ == "__main__":
    import sys

    if "--semi" in sys.argv:
        main_semi()
    elif "--shear" in sys.argv:
        main_shear()
    else:
        main()

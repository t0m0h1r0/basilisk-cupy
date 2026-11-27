"""docs/notes/91 (P0): the primitive-function combined compact difference (PCCD, the user's working name).

Cell averages q_i of u over [x_i, x_{i+1}] are the differences of the primitive F (F_{i+1} - F_i = h q_i). The 6th-order
Chu-Fan CCD applied to F, with v_i = F'_i = u(x_i) and g_i = h F''_i = h u_x(x_i) on the faces, reads

    v_i + 7/16 (v_{i-1} + v_{i+1}) - 1/16 (g_{i+1} - g_{i-1}) = 15/16 (q_{i-1} + q_i)
    g_i - 1/8 (g_{i-1} + g_{i+1}) + 9/8 (v_{i+1} - v_{i-1}) = 3 (q_i - q_{i-1})

(a 2 x 2 block tridiagonal system; periodic: block circulant). In cell i (s = (x - x_i)/h in [0, 1]) the cubic Hermite
interpolant of (v_i, g_i, v_{i+1}, g_{i+1}) plus the bubble 30 s^2 (1 - s)^2 (q_i - mean H_i) is the reconstruction:
C^1 across faces, the cell mean exact.

Transports of the same reconstruction (constant density, constant velocity, CFL C, periodic):
* "remap": q_i^{n+1} = int over the cell of the translated reconstruction (the swept-volume flux form);
* "galerkin": G q^{n+1} = C q^n, G = <phi_i, phi_j>, C = <phi_i, S phi_j>;
* "corr": the adjoint correction T (3/2 - T^dagger T / 2), T^dagger = G^{-1} C^T.

    python -m cases.pccd
"""

from __future__ import annotations

import json
import math

import numpy as np

_xg, _wg = np.polynomial.legendre.leggauss(8)
XG, WG = 0.5 * (_xg + 1.0), 0.5 * _wg

# s-monomial coefficients (s^0..s^4) of the Hermite basis and the bubble
H_V0 = np.array([1.0, 0.0, -3.0, 2.0, 0.0])
H_G0 = np.array([0.0, 1.0, -2.0, 1.0, 0.0])
H_V1 = np.array([0.0, 0.0, 3.0, -2.0, 0.0])
H_G1 = np.array([0.0, 0.0, -1.0, 1.0, 0.0])
BUB = 30.0 * np.array([0.0, 0.0, 1.0, -2.0, 1.0])
MEAN = np.array([1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5])          # int_0^1 s^p ds


# --------------------------------------------------------------------------------------------- the CCD system
def ccd_matrices(N, periodic=True):
    """(A, B): A [v; g] = B q, unknowns interleaved (v_0, g_0, v_1, g_1, ...), faces i = 0..N-1 (periodic)."""
    A = np.zeros((2 * N, 2 * N))
    B = np.zeros((2 * N, N))
    for i in range(N):
        im, ip = (i - 1) % N, (i + 1) % N
        r = 2 * i
        A[r, 2 * i] += 1.0
        A[r, 2 * im] += 7 / 16
        A[r, 2 * ip] += 7 / 16
        A[r, 2 * ip + 1] += -1 / 16
        A[r, 2 * im + 1] += 1 / 16
        B[r, im] += 15 / 16
        B[r, i] += 15 / 16
        r += 1
        A[r, 2 * i + 1] += 1.0
        A[r, 2 * im + 1] += -1 / 8
        A[r, 2 * ip + 1] += -1 / 8
        A[r, 2 * ip] += 9 / 8
        A[r, 2 * im] += -9 / 8
        B[r, i] += 3.0
        B[r, im] += -3.0
    return A, B


def faces(q):
    """Face values v and scaled gradients g = h u_x of the periodic averages q."""
    A, B = ccd_matrices(q.size)
    x = np.linalg.solve(A, B @ q)
    return x[0::2], x[1::2]


def cell_coefficients(q):
    """s-monomial coefficients (N, 5) of every cell's reconstruction."""
    v, g = faces(q)
    v1, g1 = np.roll(v, -1), np.roll(g, -1)
    H = (np.outer(v, H_V0) + np.outer(g, H_G0) + np.outer(v1, H_V1) + np.outer(g1, H_G1))
    return H + np.outer(q - H @ MEAN, BUB / (BUB @ MEAN))


def reconstruction_operator(N):
    """R (N, 5, N): cell coefficients as a linear map of q (one solve with the identity's columns)."""
    A, B = ccd_matrices(N)
    X = np.linalg.solve(A, B)                       # (2N, N): faces of every unit average
    v, g = X[0::2], X[1::2]
    v1, g1 = np.roll(v, -1, axis=0), np.roll(g, -1, axis=0)
    R = (v[:, None, :] * H_V0[None, :, None] + g[:, None, :] * H_G0[None, :, None]
         + v1[:, None, :] * H_V1[None, :, None] + g1[:, None, :] * H_G1[None, :, None])
    mean = np.einsum("p,ipj->ij", MEAN, R)
    R += (np.eye(N) - mean)[:, None, :] * (BUB / (BUB @ MEAN))[None, :, None]
    return R


# --------------------------------------------------------------------------------------------- transports
def _mono(s):
    return np.stack([s ** p for p in range(5)], axis=-1)


def transport_matrices(N, C):
    """G, Cm (Galerkin) and Tr (direct remap) for the shift by C cells (0 <= C < 1); integrals in units of h.
    New cell i: s in [0, C) holds old cell i-1 at s' = s + 1 - C, s in [C, 1) old cell i at s' = s - C."""
    R = reconstruction_operator(N)
    G = np.zeros((N, N))
    Cm = np.zeros((N, N))
    Tr = np.zeros((N, N))
    Mn = _mono(XG)
    for i in range(N):
        Pn = Mn @ R[i]
        G += Pn.T @ (WG[:, None] * Pn)
    for a, b, o, shift in ((0.0, C, -1, 1.0 - C), (C, 1.0, 0, -C)):
        if b - a <= 0:
            continue
        s = a + (b - a) * XG
        w = (b - a) * WG
        Mn, Mo = _mono(s), _mono(s + shift)
        for i in range(N):
            Pn = Mn @ R[i]                # phi_j on new cell i
            Po = Mo @ R[(i + o) % N]      # the translated old basis there
            Tr[i] += w @ Po
            Cm += Pn.T @ (w[:, None] * Po)
    return G, Cm, Tr


def operators(N, C):
    G, Cm, Tr = transport_matrices(N, C)
    T = np.linalg.solve(G, Cm)
    Td = np.linalg.solve(G, Cm.T)
    return {"remap": Tr, "galerkin": T, "corr": T @ (1.5 * np.eye(N) - 0.5 * Td @ T), "G": G, "C": Cm}


# --------------------------------------------------------------------------------------------- Fourier symbols
def face_symbol(theta):
    """(V, Gam): v_i = V e^{i i theta}, g_i = Gam e^{i i theta} for q_j = e^{i j theta} (face i left of cell i)."""
    e = np.exp(-1j * theta)
    M = np.array([[1 + 7 / 8 * math.cos(theta), -1j * math.sin(theta) / 8],
                  [9 / 4 * 1j * math.sin(theta), 1 - math.cos(theta) / 4]])
    rhs = np.array([15 / 16 * (1 + e), 3 * (1 - e)])
    return np.linalg.solve(M, rhs)


def cell0(theta):
    """s-coefficients of cell 0 for q_j = e^{i j theta}."""
    V, Gm = face_symbol(theta)
    E = np.exp(1j * theta)
    H = V * H_V0 + Gm * H_G0 + V * E * H_V1 + Gm * E * H_G1
    return H + (1.0 - H @ MEAN) * BUB / (BUB @ MEAN)


def phi(theta, x):
    """The reconstruction at x (in cells; cell j = [j, j + 1])."""
    x = np.asarray(x, dtype=float)
    c = cell0(theta)
    m = np.floor(x)
    return np.exp(1j * m * theta) * (_mono(x - m) @ c)


def symbols(theta, C):
    """Amplification factors of remap, Galerkin, adjoint-corrected for the mode theta and CFL C (0 < C < 1)."""
    f = lambda x: phi(theta, x)  # noqa: E731

    def integ(fun, a, b):
        x = a + (b - a) * XG
        return (b - a) * np.sum(WG * fun(x))
    remap = integ(lambda x: f(x - C), 0.0, C) + integ(lambda x: f(x - C), C, 1.0)
    num = integ(lambda x: np.conj(f(x)) * f(x - C), 0.0, C) + integ(lambda x: np.conj(f(x)) * f(x - C), C, 1.0)
    den = integ(lambda x: np.abs(f(x)) ** 2, 0.0, 1.0)
    T = num / den
    corr = T * (1.5 - 0.5 * abs(T) ** 2)
    return {"remap": remap, "galerkin": T, "corr": corr, "gram": den}


# --------------------------------------------------------------------------------------------- measurements
def accuracy(N, k=1):
    """Errors of the faces (v, g) and of the reconstruction for u = sin(2 pi k x) + 0.3 cos(6 pi k x) on [0, 1]."""
    h = 1.0 / N
    x = np.arange(N) * h
    U = lambda x: np.sin(2 * np.pi * k * x) + 0.3 * np.cos(6 * np.pi * k * x)  # noqa: E731
    dU = lambda x: 2 * np.pi * k * (np.cos(2 * np.pi * k * x) - 0.9 * np.sin(6 * np.pi * k * x))  # noqa: E731
    Fp = lambda x: -np.cos(2 * np.pi * k * x) / (2 * np.pi * k) + 0.3 * np.sin(6 * np.pi * k * x) / (6 * np.pi * k)  # noqa
    q = (Fp(x + h) - Fp(x)) / h
    v, g = faces(q)
    c = cell_coefficients(q)
    s = np.linspace(0, 1, 21)
    rec = (c @ _mono(s).T)
    ex = U(x[:, None] + s[None, :] * h)
    return {"v": float(np.abs(v - U(x)).max()), "g": float(np.abs(g - h * dU(x)).max()),
            "rec": float(np.abs(rec - ex).max()), "mean": float(np.abs(c @ MEAN - q).max()),
            "c1_jump_value": float(np.abs(c @ np.ones(5) - np.roll(c[:, 0], -1)).max()),
            "c1_jump_slope": float(np.abs(c @ np.arange(5) - np.roll(c[:, 1], -1)).max())}


def run(N, transport, C=0.37, t_end=1.0, profile="smooth", samples=(0.25, 0.5)):
    """Translate the periodic profile at unit speed to t_end (errors of the cell averages at the sample times)."""
    h = 1.0 / N
    steps = int(math.ceil(t_end / (C * h)))
    Cs = t_end / steps / h
    T = operators(N, Cs)[transport]
    x = np.arange(N) * h
    from math import erf

    sig = 0.05
    verf = np.vectorize(erf)

    def gauss_prim(y):                  # primitive of the periodic sum of exp(-(x - 1/2)^2 / (2 sig^2))
        return sum(sig * math.sqrt(math.pi / 2) * verf((y - 0.5 - k) / (sig * math.sqrt(2))) for k in (-2, -1, 0, 1, 2))
    prim = {"smooth": lambda x: -np.cos(2 * np.pi * x) / (2 * np.pi) + 0.3 * np.sin(6 * np.pi * x) / (6 * np.pi),
            "narrow": gauss_prim}[profile]
    avg = lambda t: (prim(x + h - t) - prim(x - t)) / h  # noqa: E731
    q = avg(0.0)
    G = operators(N, Cs)["G"]
    E0 = q @ G @ q
    errs = {}
    at = {int(round(t / (Cs * h))): t for t in samples}
    for k in range(steps):
        q = T @ q
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(q - avg((k + 1) * Cs * h)).max())
    errs[t_end] = float(np.abs(q - avg(t_end)).max())
    return {"errors": errs, "energy": float(q @ G @ q / E0) - 1.0, "steps": steps, "C": Cs,
            "free_stream": float(np.abs(T @ np.ones(N) - 1).max()), "mass": float(abs(np.ones(N) @ T @ q - q.sum()))}


def main() -> None:
    out = {"accuracy": {N: accuracy(N) for N in (16, 32, 64, 128)}}
    thetas = np.linspace(1e-3, math.pi, 400)
    out["fourier"] = {}
    for C in (0.01, 0.1, 0.37, 0.5, 0.9):
        rows = [symbols(t, C) for t in thetas]
        out["fourier"][C] = {k: [[float(abs(r[k])), float(np.angle(r[k]))] for r in rows]
                             for k in ("remap", "galerkin", "corr")}
        out["fourier"][C]["gram"] = [float(r["gram"].real) for r in rows]
    out["fourier_theta"] = thetas.tolist()
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(N, tr, pr) for pr in ("smooth", "narrow") for N in (16, 32, 64, 128, 256)
            for tr in ("remap", "galerkin", "corr")]
    with ProcessPoolExecutor(15) as ex:
        futs = [(j, ex.submit(run, j[0], j[1], profile=j[2])) for j in jobs]
        out["runs"] = [{"N": N, "transport": tr, "profile": pr, **f.result()} for (N, tr, pr), f in futs]
    for r in out["runs"]:
        print(r["profile"], r["N"], r["transport"], r["errors"], "E", r["energy"], flush=True)
    json.dump(out, open("docs/figures/pccd.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

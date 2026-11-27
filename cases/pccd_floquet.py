"""docs/notes/95 (P2): Floquet analysis of the two-phase PCCD transports (the user's proposal).

A two-phase band translated at C = 1/4: after four steps the interfaces sit at the same position relative to the
grid, one cell further. With T_k the step map at step k and S the one-cell shift,

    P_rel = S^{-1} T_3 T_2 T_1 T_0

is the map over one period in the relative frame: its spectral radius is the growth per period of a disturbance.
Also reported: the largest singular value of P_rel in the mass norm (transient growth over one period) and, for the
Galerkin transport, in its own G0 norm (<= 1 by construction, note 87 D).

Transports: "remap" (the new mass-weighted average is the integral of rho u_h over the swept old pieces) and
"galerkin" (G1 T = C on the common pieces). Reconstructions: as in cases.pccd_gram ("none", "joint", "local",
"split", "split_opt" = the optimised closure of cases.pccd_closure).

    python -m cases.pccd_floquet
"""

from __future__ import annotations

import json

import numpy as np

from cases.pccd import WG, XG
from cases.pccd_jump import B_
from cases.pccd_twophase import _pieces


def _basis(N, ratio, a, b, mode):
    if mode in ("blend_partial", "split_partial"):
        from cases.pccd_local import Split, SplitBlend
        fs = []
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.0
            fs.append(SplitBlend(N, ratio, a, b, e, mixed_mode="partial") if mode == "blend_partial"
                      else Split(N, ratio, a, b, e, mixed_mode="partial"))
        return lambda j, x: fs[j](x)
    if mode in ("split_opt", "blend_opt"):
        from cases.pccd_closure import closure
        from cases.pccd_local import Split, SplitBlend
        cl = closure(np.array(json.load(open("docs/figures/pccd_closure.json"))["best"]))
        cls = Split if mode == "split_opt" else SplitBlend
        fs = []
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.0
            fs.append(cls(N, ratio, a, b, e, closure=cl))
        return lambda j, x: fs[j](x)
    from cases.pccd_gram import basis
    return basis(N, ratio, a, b, mode)


def step(N, ratio, a0, b0, D, mode, transport, phi0=None):
    """(T, phi1, G0, G1, m1): the map of one step from the geometry (a0, b0) to (a0 + D, b0 + D)."""
    h = 1.0 / N
    a1, b1 = (a0 + D) % 1.0, (b0 + D) % 1.0
    phi0 = phi0 or _basis(N, ratio, a0, b0, mode)
    phi1 = _basis(N, ratio, a1, b1, mode)
    X, W, cell = [], [], []
    for i in range(N):
        for p, q, hv in _pieces(i * h, (i + 1) * h, a1, b1, extra=[i * h + D]):
            for k in range(2):                       # each piece halved: smoother integrands per part
                pp, qq = p + (q - p) * k / 2, p + (q - p) * (k + 1) / 2
                X.append(pp + (qq - pp) * XG)
                W.append((qq - pp) * WG * (1.0 if hv else 1.0 / ratio))
                cell.append(np.full(XG.size, i))
    X, W, cell = np.concatenate(X), np.concatenate(W), np.concatenate(cell)
    B0 = np.stack([phi0(j, X - D) for j in range(N)], axis=1)
    B1 = np.stack([phi1(j, X) for j in range(N)], axis=1)
    m1 = np.bincount(cell, weights=W, minlength=N)
    G0 = B0.T @ (W[:, None] * B0)
    G1 = B1.T @ (W[:, None] * B1)
    if transport == "remap":
        R = np.zeros((N, X.size))
        R[cell, np.arange(X.size)] = W
        T = (R @ B0) / m1[:, None]
    else:
        T = np.linalg.solve(G1, B1.T @ (W[:, None] * B0))
    return T, phi1, G0, G1, m1


def floquet(N, ratio, c, mode, transport, a_cell=None):
    """The interface a starts in cell N/4 with heavy fraction c (b far away)."""
    h = 1.0 / N
    m = N // 4 if a_cell is None else a_cell
    a = (m + 1 - c) * h
    D = h / 4
    P = np.eye(N)
    phi = None
    G0 = None
    for k in range(4):
        T, phi, Ga, Gb, m1 = step(N, ratio, (a + k * D) % 1.0, (B_ + k * D) % 1.0, D, mode, transport, phi)
        if k == 0:
            G0, M0 = Ga, None
        P = T @ P
    S = np.roll(np.eye(N), 1, axis=0)                 # (S u)_i = u_{i-1}
    Prel = S.T @ P
    rho = float(np.max(np.abs(np.linalg.eigvals(Prel))))
    # the mass norm of the start geometry (= of the end geometry in the relative frame)
    T0, _, _, _, m0 = step(N, ratio, (a - D) % 1.0, (B_ - D) % 1.0, D, mode, "remap")
    sm = np.sqrt(m0)
    sv_mass = float(np.linalg.norm((sm[:, None] * Prel) / sm[None, :], 2))
    L = np.linalg.cholesky(G0 + 1e-300 * np.eye(N))
    sv_G = float(np.linalg.norm(L.T @ Prel @ np.linalg.inv(L.T), 2))
    return {"spectral_radius": rho, "sv_mass": sv_mass, "sv_G": sv_G}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for mode in ("none", "joint", "local", "split", "split_opt", "blend", "blend_opt"):
        for transport in ("remap", "galerkin"):
            for ratio in (1.0, 1e3, 1e6):
                for c in (0.5, 0.05, 1e-3):
                    jobs.append((32, ratio, c, mode, transport))
    res = []
    with ProcessPoolExecutor(30) as ex:
        futs = [(j, ex.submit(floquet, *j)) for j in jobs]
        for j, f in futs:
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            res.append({"N": j[0], "ratio": j[1], "c": j[2], "mode": j[3], "transport": j[4], **r})
            print(j, r, flush=True)
    json.dump(res, open("docs/figures/pccd_floquet.json", "w"), indent=1)




def continuity(N, ratio, spacing, mode, span=0.2, seed=0):
    """Largest change of u_h (fixed random averages) between interface positions ``spacing`` cells apart while the
    interface a crosses the right face of cell N/4 (a jump does not shrink with the spacing)."""
    h = 1.0 / N
    u = np.random.default_rng(seed).normal(size=N)
    X = ((np.arange(N)[:, None] + np.linspace(0.01, 0.99, 25)[None, :]) * h).ravel()
    prev, worst, where = None, 0.0, None
    for t in np.arange(-span / 2, span / 2 + 1e-12, spacing):
        a = (N // 4 + 1 + t) * h
        phi = _basis(N, ratio, a, B_, mode)
        v = sum(u[j] * phi(j, X) for j in range(N))
        if prev is not None:
            d = float(np.abs(v - prev).max())
            if d > worst:
                worst, where = d, float(t)
        prev = v
    return {"max_change": worst, "at": where}


def floquet_general(N, ratio, c, mode, transport, p=1, q=4, powers=40):
    """CFL = p/q: after q steps the configuration has moved p cells. P_rel = S^{-p} T_{q-1}..T_0. Returns the spectral
    radius, the largest |eigenvalue| other than the free stream's 1 (the dissipation of the rest), the largest mass-norm
    singular value of P_rel (one period) and of P_rel^k for k <= ``powers`` (long-time transient growth), and for every
    starting step the one-period growth (the transient from any start)."""
    h = 1.0 / N
    m = N // 4
    a = (m + 1 - c) * h
    D = p * h / q
    Ts, masses = [], []
    phi = None
    for k in range(q):
        T, phi, _, _, m1 = step(N, ratio, (a + k * D) % 1.0, (B_ + k * D) % 1.0, D, mode, transport, phi)
        Ts.append(T)
        masses.append(m1)
    S = np.roll(np.eye(N), p, axis=0)
    # the start geometry's masses: the end geometry shifted back by p cells
    mstart = np.roll(masses[-1], -p)
    sm = np.sqrt(mstart)
    def per(start):
        P = np.eye(N)
        for k in range(q):
            P = Ts[(start + k) % q] @ P
        return P
    P = S.T @ per(0)
    ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    Pm = (sm[:, None] * P) / sm[None, :]
    sv1 = float(np.linalg.norm(Pm, 2))
    Pk, svk = np.eye(N), 0.0
    for _ in range(powers):
        Pk = Pm @ Pk
        svk = max(svk, float(np.linalg.norm(Pk, 2)))
    return {"spectral_radius": float(ev[0]), "second": float(ev[1]), "sv_one_period": sv1, "sv_max_powers": svk}


CFLS = ((1, 16), (1, 8), (1, 4), (1, 3), (1, 2), (3, 4), (9, 10))


def main_general() -> None:
    """docs/notes/97 (P2): every CFL of CFLS, density ratios 1, 1e3, 1e6, heavy fractions c down to 1e-6 (zeta ~ 1 at
    ratio 1e6), remap and Galerkin, for the partial-cell phase split (mode "blend_partial")."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(32, r, c, "blend_partial", tr, p, q) for (p, q) in CFLS for r in (1.0, 1e3, 1e6)
            for c in (0.5, 1e-2, 1e-4, 1e-6) for tr in ("remap", "galerkin")]
    res = []
    with ProcessPoolExecutor(30) as ex:
        futs = [(j, ex.submit(floquet_general, *j)) for j in jobs]
        for j, f in futs:
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            res.append({"ratio": j[1], "c": j[2], "mode": j[3], "transport": j[4], "cfl": f"{j[5]}/{j[6]}", **r})
            print(res[-1], flush=True)
    json.dump(res, open("docs/figures/pccd_floquet_general.json", "w"), indent=1)


def run_periodic(N, ratio, mode, transport, p=1, q=4, t_end=1.0, samples=(0.25, 0.5)):
    """Translate note 93's two-phase profile at CFL p/q to t_end. The q step maps of one period are built once: in
    period j the configuration is shifted by j p cells, T_{jq+k} = S^{jp} T_k S^{-jp}. Errors of the mass-weighted
    averages against the exact ones at the sample times."""
    from cases.pccd_jump import A_, exact
    from cases.pccd_twophase import averages

    h = 1.0 / N
    D = p * h / q
    Ts, phi = [], None
    for k in range(q):
        T, phi, _, _, _ = step(N, ratio, (A_ + k * D) % 1.0, (B_ + k * D) % 1.0, D, mode, transport, phi)
        Ts.append(T)
    u0 = lambda x: exact(x % 1.0)  # noqa: E731
    u = averages(N, A_, B_, ratio, u0)
    steps = int(round(t_end / D))
    at = {int(round(t / D)): t for t in samples}
    errs = {}
    ex = lambda t: averages(N, (A_ + t) % 1.0, (B_ + t) % 1.0, ratio, lambda x: u0(x - t))  # noqa: E731
    for n in range(steps):
        j, k = divmod(n, q)
        u = np.roll(Ts[k] @ np.roll(u, -j * p), j * p)
        if n + 1 in at:
            errs[at[n + 1]] = float(np.abs(u - ex((n + 1) * D)).max())
    errs[t_end] = float(np.abs(u - ex(steps * D)).max())
    return errs


if __name__ == "__main__":
    import sys

    main_general() if "--general" in sys.argv else main()

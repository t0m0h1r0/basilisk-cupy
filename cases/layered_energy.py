"""Why the layered CCD viscosity (D1 o D1) increases the energy with walls (docs/notes/37): ``layered_energy.png``.

The implicit viscous step is ``A = (I - kappa L)^{-1}``, ``L = h^2 L_H`` (mu = 1, unit density),
``kappa = mu dt / h^2 = 0.426`` (the value of the decay test of docs/notes/33 at N = 64). The operator is
assembled column by column from :class:`~basilisk_cupy.terms.viscous.CCDStress` (2 n^2 unknowns).

Reported for each operator (split / layered, walls / periodic):

* ``max Re lambda(L)`` and the largest eigenvalue of the symmetric part ``(L + L^T)/2``
  (energy can grow in one step only if it is positive);
* ``||L - L^T|| / ||L||``, the nullity of ``L`` and ``L^2`` (``L^2`` larger: a Jordan block at 0, growth
  in powers of k);
* ``||A||_2``, ``rho(A)`` and the energy of white noise after k steps of ``A`` (and ``P A``, ``P`` the
  approximate projection of CenteredNS, walls only);
* 1-D: the same for ``D1_ext D1_dir`` (layered: layer A with the Dirichlet closure, layer C with the
  extrapolating one), ``D2_dir`` (split) and ``-D1_dir^T D1_dir``.

The time-stepping numbers (``DECAY``) are from :func:`cases.coupled_stability.viscous_decay`.

    python -m cases.layered_energy [--n 16] [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

KAPPA = 0.426

#: viscous_decay(form, n=N): E/E0 after 1, 10, 20, 40 steps (measured)
DECAY = {("layered", 16): (6.16e-01, 3.66e+00, 6.73e+01, 2.49e+03), ("split", 16): (4.92e-01, 2.14e-02, 6.01e-03, 1.45e-03),
         ("layered", 32): (5.66e-01, 9.31e-01, 3.88e+00, 8.39e+01), ("split", 32): (4.61e-01, 1.36e-02, 4.11e-03, 1.58e-03),
         ("layered", 64): (5.26e-01, 2.28e-01, 1.49e+00, 7.84e+00), ("split", 64): (4.02e-01, 1.26e-02, 4.06e-03, 1.21e-03)}


def stress_matrix(n: int, form: str, periodic: bool) -> np.ndarray:
    """``h^2 L_H`` (mu = 1) of :class:`CCDStress` with the velocity conditions of CenteredNS."""
    from basilisk_cupy import CartesianGrid
    from basilisk_cupy.terms.viscous import CCDStress
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    ns = (CenteredNS(grid, periodic=(0, 1)) if periodic else
          CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}))
    op, u, m = CCDStress(form=form), ns.u, 2 * n * n
    L = np.zeros((m, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = 1.0
        for a, c in enumerate(u):
            c.interior = e[a * n * n:(a + 1) * n * n].reshape(n, n)
        u.fill_ghosts()
        L[:, j] = np.concatenate([o.ravel() for o in op(u, 1.0)])
    return L / n ** 2


def projection_matrix(n: int) -> np.ndarray:
    import warnings

    from cases.operator_commutator import Ops

    o, m = Ops(n=n), 2 * n * n
    P = np.zeros((m, m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 1e-14 multigrid tolerance is met to ~1e-13
        for j in range(m):
            e = np.zeros(m)
            e[j] = 1.0
            P[:, j] = np.concatenate([c.ravel() for c in o.P([e[:n * n].reshape(n, n), e[n * n:].reshape(n, n)])])
    return P


def analyse(L: np.ndarray, steps: int = 80, P: np.ndarray | None = None, seed: int = 0) -> dict:
    m = L.shape[0]
    A = np.linalg.inv(np.eye(m) - KAPPA * L)
    M = A if P is None else P @ A
    sv = np.linalg.svd(L, compute_uv=False)
    sv2 = np.linalg.svd(L @ L, compute_uv=False)
    x0 = np.random.default_rng(seed).standard_normal(m)
    if P is not None:
        x0 = P @ x0
    x, E = x0.copy(), []
    for _ in range(steps):
        x = M @ x
        E.append(float(x @ x / (x0 @ x0)))
    ev = np.linalg.eigvals(L)
    sym = np.linalg.eigvalsh(0.5 * (L + L.T))
    return {"max_re": float(ev.real.max()), "sym_max": float(sym.max()), "sym": sym,
            "asym": float(np.linalg.norm(L - L.T) / np.linalg.norm(L)),
            "null": (int((sv < 1e-9 * sv[0]).sum()), int((sv2 < 1e-9 * sv2[0]).sum())),
            "norm": float(np.linalg.norm(M, 2)), "rho": float(abs(np.linalg.eigvals(M)).max()),
            "energy": np.array(E), "eig": ev}


def one_d(n: int) -> dict:
    from basilisk_cupy.discretization.ccd import ccd_matrices

    h = 1.0 / n
    D1d, D2d = ccd_matrices(n, h, "dirichlet", "dirichlet")[:2]
    D1e = ccd_matrices(n, h, "extrapolate", "extrapolate")[0]
    D1p = ccd_matrices(n, h, "periodic", "periodic")[0]
    ops = {"periodic D1 D1": D1p @ D1p, "walls D1_ext D1_dir (layered)": D1e @ D1d,
           "walls D2_dir (split)": D2d, "walls -D1_dir^T D1_dir": -D1d.T @ D1d}
    return {k: analyse(v * h * h, steps=40) for k, v in ops.items()}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    n = args.n
    for nn in (16, 32, 64):
        for k, r in one_d(nn).items():
            print(f"1-D n={nn:3d} {k:30s} maxRe {r['max_re']:+.2e} sym max {r['sym_max']:+.2e} ||A|| {r['norm']:.3f} "
                  f"null L/L^2 {r['null'][0]}/{r['null'][1]} E40 {r['energy'][39]:.2e}")
    res, P = {}, None
    for periodic in (True, False):
        for form in ("split", "layered"):
            L = stress_matrix(n, form, periodic)
            r = res[(form, periodic)] = analyse(L)
            print(f"2-D N={n} {'periodic' if periodic else 'walls':8s} {form:8s} maxRe {r['max_re']:+.1e} sym max "
                  f"{r['sym_max']:+.2e} asym {r['asym']:.1e} null L/L^2 {r['null'][0]}/{r['null'][1]} ||A|| {r['norm']:.3f} "
                  f"rho {r['rho']:.5f} E k=1,40,80 {r['energy'][0]:.2e} {r['energy'][39]:.2e} {r['energy'][79]:.2e}",
                  flush=True)
            if not periodic:
                P = projection_matrix(n) if P is None else P
                rp = res[(form, "P")] = analyse(L, P=P)
                print(f"     with the projection, P A: ||PA|| {rp['norm']:.3f} rho {rp['rho']:.5f} E k=1,40,80 "
                      f"{rp['energy'][0]:.2e} {rp['energy'][39]:.2e} {rp['energy'][79]:.2e}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    k = np.arange(1, 81)
    for (form, bc), r in res.items():
        lab = {True: "periodic", False: "walls", "P": "walls, P A"}[bc]
        ax[0].semilogy(k, r["energy"], "-" if form == "layered" else "--", label=f"{form}, {lab}")
    ax[0].set_xlabel("steps k")
    ax[0].set_ylabel("E_k / E_0 (white noise)")
    ax[0].set_title(f"(a) matrix A = (I - kappa L)^-1, N = {n}, kappa = {KAPPA}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    for (form, bc), st in ((("layered", False), "-"), (("layered", True), ":"), (("split", False), "--"),
                           (("split", True), "-.")):
        sym = np.sort(res[(form, bc)]["sym"])[::-1][:60]
        ax[1].plot(np.arange(1, sym.size + 1), sym, st, marker=".", label=f"{form}, {'periodic' if bc else 'walls'}")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("index (largest first)")
    ax[1].set_ylabel("eigenvalue of (L + L^T)/2, L = h^2 L_H")
    ax[1].set_title("(b) symmetric part: > 0 lets the energy grow in one step")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    steps = [1, 10, 20, 40]
    for (form, nn), e in DECAY.items():
        ax[2].semilogy(steps, e, "o-" if form == "layered" else "s--", color={16: "C0", 32: "C1", 64: "C3"}[nn],
                       label=f"{form}, N = {nn}")
    ax[2].set_xlabel("steps")
    ax[2].set_ylabel("E / E0")
    ax[2].set_title("(c) CenteredNS, viscous only (viscous_decay)")
    ax[2].legend(fontsize=8, ncol=2)
    ax[2].grid(alpha=0.3, which="both")
    fig.savefig(f"{args.out}/layered_energy.png", dpi=100)
    print("wrote", f"{args.out}/layered_energy.png")


if __name__ == "__main__":
    main()

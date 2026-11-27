"""docs/notes/103 (Gate 1-A, audit issue #117 and the user's proposal): the pressure projection on PCCD cell averages.

Divergence B = D_f I (face values of the reconstruction, differenced), energy G_A (the advection Gram), projection

    Pi = I - G^{-1} B^T (B G^{-1} B^T)^+ B          (G-orthogonal: Pi^T G Pi <= G for any B)

With the plain PCCD face interpolation the face-centred symbol I(theta) is real and vanishes at theta = pi, so the
checkerboard is in the null space of B and of the Schur complement L = B G^{-1} B^T. The user's correction

    v_f = v_f^PCCD + alpha / 32 delta^4 (q_i - q_{i-1}),   I_alpha = I_PCCD + i alpha sin^5(theta / 2)

adds an O(h^5) skew part that is large only near the grid scale.

    python -m cases.projection_gate
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import ccd_matrices, face_symbol, symbols

ALPHAS = (0.0, 1.0, 1.1, 1.2, 1.3, 1.5)


# --------------------------------------------------------------------------------------------- single phase: symbols
def face_interp(theta, alpha=0.0):
    """Face-centred symbol of the face value from the averages q_j = e^{i j theta} (the face at the origin)."""
    V, _ = face_symbol(theta)
    return V * np.exp(0.5j * theta) + 1j * alpha * math.sin(theta / 2) ** 5


def _tables(n=2049):
    th = np.linspace(0.0, math.pi, n)
    g = np.array([1.0] + [symbols(t, 0.37)["gram"].real for t in th[1:]])
    I0 = np.array([1.0] + [face_interp(t).real for t in th[1:]])
    return th, g, I0


def schur_ratios(alphas=ALPHAS, n=2049):
    """min and max of L(theta) / |theta|^2 (h = rho = 1), 1-D and 2-D (n points per direction on [0, pi]; the
    symbols are even in theta). L = sum_d |2 sin(theta_d / 2) I_alpha(theta_d)|^2 / prod_d g(theta_d)."""
    th, g, I0 = _tables(n)
    s = np.sin(th / 2)
    out = []
    for al in alphas:
        b2 = (2 * s) ** 2 * (I0 ** 2 + (al * s ** 5) ** 2)       # |b|^2 (I_PCCD real)
        k = th > 0
        r1 = b2[k] / g[k] / th[k] ** 2
        L2 = (b2[:, None] + b2[None, :]) / (g[:, None] * g[None, :])
        den = th[:, None] ** 2 + th[None, :] ** 2
        m = den > 0
        r2 = L2[m] / den[m]
        # the same against the 2nd-order Laplacian symbol 4 sin^2 (the MAC operator)
        lap2 = 4 * s[:, None] ** 2 + 4 * s[None, :] ** 2
        out.append({"alpha": al, "1d_min": float(r1.min()), "1d_max": float(r1.max()),
                    "2d_min": float(r2.min()), "2d_max": float(r2.max()),
                    "2d_vs_mac_min": float((L2[m] / lap2[m]).min()), "2d_vs_mac_max": float((L2[m] / lap2[m]).max()),
                    "theta_min_2d": [float(x) for x in np.unravel_index(np.argmin(np.where(m, L2 / np.where(m, den, 1), np.inf)), L2.shape)]})
        out[-1]["theta_min_2d"] = [float(th[int(i)] / math.pi) for i in out[-1]["theta_min_2d"]]
    mac = 4 * s[1:] ** 2 / th[1:] ** 2
    return {"pccd": out, "mac_1d_min": float(mac.min()), "mac_2d_min": float(4 / math.pi ** 2)}


def projection_accuracy(alpha, thetas=(0.4, 0.2, 0.1, 0.05), angles=24):
    """2-D single phase, periodic: the projection of the exact averages of a divergence-free plane wave (should be
    unchanged) and of a gradient plane wave (should vanish). G is a scalar per mode for both components, so
    Pi = I - b^H b / |b|^2 with b = 2i (s_x I(theta_x), s_y I(theta_y)); the exact b is i (theta_x, theta_y).
    Returns the max over directions of |Pi u_df - u_df| / |u| and |Pi u_cf| / |u| per |theta|."""
    out = []
    for r in thetas:
        edf = ecf = 0.0
        for t in np.linspace(0.0, math.pi / 2, angles):
            tx, ty = r * math.cos(t), r * math.sin(t)
            b = np.array([2j * math.sin(tx / 2) * face_interp(tx, alpha), 2j * math.sin(ty / 2) * face_interp(ty, alpha)])
            P = np.eye(2) - np.outer(np.conj(b), b) / np.vdot(b, b).real
            udf = np.array([-ty, tx]) / r
            ucf = np.array([tx, ty]) / r
            edf = max(edf, float(np.linalg.norm(P @ udf - udf)))
            ecf = max(ecf, float(np.linalg.norm(P @ ucf)))
        out.append({"theta": r, "div_free_err": edf, "grad_err": ecf})
    return out


# --------------------------------------------------------------------------------------------- matrices (periodic)
def face_matrix(N, alpha=0.0):
    """(N faces x N cells): face f (left of cell f) values of the periodic PCCD, plus alpha/32 delta^4 (q_f - q_{f-1})."""
    A, Bm = ccd_matrices(N)
    I = np.linalg.solve(A, Bm)[0::2]
    return I + alpha * correction_matrix(N)


def correction_matrix(N):
    Dm = np.eye(N) - np.roll(np.eye(N), -1, axis=1)            # row f: q_f - q_{f-1}
    d2 = -2 * np.eye(N) + np.roll(np.eye(N), 1, axis=1) + np.roll(np.eye(N), -1, axis=1)
    return d2 @ d2 @ Dm / 32.0


def divergence(Iface):
    """(B u)_i = v_{i+1} - v_i (h = 1)."""
    N = Iface.shape[0]
    return np.roll(Iface, -1, axis=0) - Iface


def structure_1d(N=32, alpha=1.2):
    """Single phase, periodic: spurious null spaces of B (velocity) and B^T (pressure), the projection identities, and
    the eigenvalues of L against the symbol."""
    from cases.viscous_minenergy import periodic_pccd_gram

    G = periodic_pccd_gram(N) * N                              # h = 1
    B = divergence(face_matrix(N, alpha))
    return _structure(G, B)


def _structure(G, B, tol=1e-10):
    N = G.shape[0]
    Gi = np.linalg.inv(G)
    L = B @ Gi @ B.T
    sv = np.linalg.svd(B, compute_uv=False)
    Pi = np.eye(N) - Gi @ B.T @ np.linalg.pinv(L, rcond=1e-12) @ B
    one = np.ones(N)
    u = np.random.default_rng(1).normal(size=N)
    return {"null_B": int((sv < tol * sv.max()).sum()), "null_L": int((np.linalg.eigvalsh(0.5 * (L + L.T)) < tol * np.abs(L).max()).sum()),
            "idempotent": float(np.abs(Pi @ Pi - Pi).max()), "B_Pi": float(np.abs(B @ Pi).max()),
            "energy_max": float(np.linalg.eigvalsh(Pi.T @ G @ Pi - G)[-1] / np.linalg.eigvalsh(G)[-1]),
            "momentum": float(abs(one @ G @ (Pi @ u) - one @ G @ u) / abs(one @ G @ np.abs(u))),
            "Pi_u_constant": float(np.ptp(Pi @ u) / np.abs(u).max())}


# --------------------------------------------------------------------------------------------- two phase, y-layers
def _y_operators(N, ratio, a, b):
    """G_A,y (rho-weighted Gram of the partial-cell split PCCD) and the face values of its reconstruction (N x N),
    with the largest jump between the one-sided face values."""
    from cases.pccd import WG, XG
    from cases.pccd_local import SplitBlend
    from cases.viscous_minenergy import MinEnergy, advection_basis
    from cases.viscous_stress import Setup

    S = Setup(N, ratio, a, b, 1.0, 1.0)
    ME = MinEnergy(S, nodes=(XG, WG))
    X = np.concatenate([c[0] for c in ME.cells])
    W = np.concatenate([c[1] for c in ME.cells])
    BA = advection_basis(N, ratio, a, b, X)
    G = BA.T @ (W[:, None] * BA)
    h = 1.0 / N
    xf = np.arange(N) * h
    eps = 1e-11
    I = np.zeros((N, N))
    jump = 0.0
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        R = SplitBlend(N, ratio, a, b, e, mixed_mode="partial")
        lo, hi = R((xf - eps) % 1.0), R(xf + eps)
        I[:, j] = 0.5 * (lo + hi)
        jump = max(jump, float(np.abs(hi - lo).max()))
    return G, I, S.m, jump


def _volume_average(N, ratio, a, b):
    """V (N x N): the plain volume average over each cell of the split-PCCD reconstruction of the (mass-weighted)
    averages q. In a mixed cell int rho u / int rho != int u / |K|; the x-face flux of a y-layered cell needs the
    volume average (docs/notes/104 supplement, the user's review of PR #123)."""
    from cases.pccd import WG, XG
    from cases.viscous_minenergy import MinEnergy, advection_basis
    from cases.viscous_stress import Setup

    S = Setup(N, ratio, a, b, 1.0, 1.0)
    ME = MinEnergy(S, nodes=(XG, WG))
    V = np.zeros((N, N))
    h = 1.0 / N
    for i, (X, Wr, _, _) in enumerate(ME.cells):
        heavy = ((X - a) % 1.0) < ((b - a) % 1.0)
        w = Wr / np.where(heavy, 1.0, 1.0 / ratio)              # geometric weights
        V[i] = w @ advection_basis(N, ratio, a, b, X) / h
    return V


def _mobilities(N, ratio, a, b):
    """Interval mobilities for the 2nd-order reference: x (within a y-layered cell, parallel): mean of 1/rho over the
    cell; y (between the centres of cells f-1 and f, in series): h / int rho."""
    from cases.pccd import WG, XG

    h = 1.0 / N

    def rho(y):
        return np.where(((y - a) % 1.0) < ((b - a) % 1.0), 1.0, 1.0 / ratio)

    def integ(fun, lo, hi):
        cuts = sorted({lo, hi, *[c + k for c in (a, b) for k in (-1.0, 0.0, 1.0) if lo < c + k < hi]})
        return sum((q - p) * (WG @ fun((p + (q - p) * XG) % 1.0)) for p, q in zip(cuts[:-1], cuts[1:]))
    bx = np.array([integ(lambda y: 1.0 / rho(y), i * h, (i + 1) * h) / h for i in range(N)])
    by = np.array([h / integ(rho, (f - 0.5) * h, (f + 0.5) * h) for f in range(N)])
    return bx, by


def layered(N=32, ratio=1e3, a=0.2613, b=0.7137, alphas=(0.0, 1.2), ntheta=33, variant="cross", xavg="mass",
            ref="point"):
    """Two-phase, interface on y-lines (rho depends on y only), Fourier in x: per theta_x the N x N Schur complement
    L = |b_x|^2 / (h g_x) G_y^{-1} / h^2 + D_y I_y G_y^{-1} I_y^T D_y^T / (h^3 g_x) against the 2nd-order
    variable-density Laplacian L2 = 4 sin^2(theta_x / 2) / h^3 M^{-1} + D_y M_f^{-1} D_y^T / h^3 (M cell masses, M_f
    face masses from the density at the face point). variant "cross": the delta^4 correction everywhere (its stencil
    crosses the interface); "run": only on faces whose 6-cell stencil has no mixed cell: the generalised eigenvalues (min, max), the count of near-zero
    ones, and the distance (cells) from the smallest mode's peak to the nearest mixed cell."""
    G, I0, m, jump = _y_operators(N, ratio, a, b)
    h = 1.0 / N
    th, g, Itab = _tables(1025)
    xf = np.arange(N) * h
    heavy = ((xf - a) % 1.0) < ((b - a) % 1.0)
    mf = np.where(heavy, 1.0, 1.0 / ratio) * h           # the density at the face point (layers: the exact face flux)
    mixed = [i for i in range(N) if (min(a, b) < (i + 1) * h and max(a, b) > i * h) and
             any(i * h < x < (i + 1) * h for x in (a, b))]
    Dy = np.roll(np.eye(N), -1, axis=0) - np.eye(N)      # (B v)_i = v_{i+1} - v_i  (faces x -> cells)
    Gi = np.linalg.inv(G)
    Vx = _volume_average(N, ratio, a, b) if xavg == "volume" else np.eye(N)
    bxm, bym = _mobilities(N, ratio, a, b)
    out = {"N": N, "ratio": ratio, "face_jump": jump, "gram_max": float(np.linalg.eigvalsh(G * (1 / np.sqrt(m))[:, None] * (1 / np.sqrt(m))[None, :])[-1]), "alphas": {}}
    for al in alphas:
        Cm = correction_matrix(N)
        if variant == "run":                               # no stencil across a mixed cell: zero those faces
            for f in range(N):
                if any((f + o) % N in mixed for o in range(-3, 3)):
                    Cm[f] = 0.0
        Iy = I0 + al * Cm
        By = Dy @ Iy
        null_By = int((np.linalg.svd(By, compute_uv=False) < 1e-10).sum())
        rows = []
        for tx in np.linspace(0.0, math.pi, ntheta):
            gx = float(np.interp(tx, th, g))
            sx = math.sin(tx / 2)
            Ix = float(np.interp(tx, th, Itab))
            bx2 = (2 * sx) ** 2 * (Ix ** 2 + (al * sx ** 5) ** 2)
            L = (bx2 * Vx @ Gi @ Vx.T + By @ Gi @ By.T) / (gx * h ** 3)
            if ref == "interval":
                L2 = (4 * sx ** 2 * np.diag(bxm) + Dy @ np.diag(bym) @ Dy.T) / h ** 4
            else:
                L2 = (4 * sx ** 2 * np.diag(1 / m) + Dy @ np.diag(1 / mf) @ Dy.T) / h ** 3
            # generalised eigenvalues on the complement of the constants when theta_x = 0
            if tx == 0.0:
                Q = np.linalg.qr(np.column_stack([np.ones(N), np.eye(N)[:, :N - 1]]))[0][:, 1:]
                L, L2 = Q.T @ L @ Q, Q.T @ L2 @ Q
            C = np.linalg.cholesky(0.5 * (L2 + L2.T))
            Ci = np.linalg.inv(C)
            ev, V = np.linalg.eigh(Ci @ (0.5 * (L + L.T)) @ Ci.T)
            pmode = Ci.T @ V[:, 0]
            if tx == 0.0:
                pmode = Q @ pmode
            k = int(np.argmax(np.abs(pmode)))
            dist = min(min(abs(k - mm), N - abs(k - mm)) for mm in mixed)
            rows.append({"theta_x": float(tx / math.pi), "min": float(ev[0]), "max": float(ev[-1]),
                         "near_zero": int((ev < 1e-8 * ev[-1]).sum()), "min_mode_dist": int(dist)})
        out["alphas"][al] = {"null_By": null_By, "min": min(r["min"] for r in rows), "max": max(r["max"] for r in rows),
                             "near_zero": max(r["near_zero"] for r in rows), "rows": rows,
                             "dist": min(rows, key=lambda r: r["min"])["min_mode_dist"]}
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    res = {"schur": schur_ratios(), "accuracy": {al: projection_accuracy(al) for al in (0.0, 1.2)},
           "structure_1d": {al: structure_1d(32, al) for al in (0.0, 1.2)}}
    for r in res["schur"]["pccd"]:
        print("alpha %.1f  1d %.3f..%.3f  2d %.3f..%.3f  vs MAC %.3f..%.3f  at theta/pi %s" % (
            r["alpha"], r["1d_min"], r["1d_max"], r["2d_min"], r["2d_max"], r["2d_vs_mac_min"], r["2d_vs_mac_max"], r["theta_min_2d"]))
    print("MAC 1d/2d min", res["schur"]["mac_1d_min"], res["schur"]["mac_2d_min"])
    for al, rows in res["accuracy"].items():
        print("accuracy alpha", al, " ".join(f"{r['theta']}: {r['div_free_err']:.2e}/{r['grad_err']:.2e}" for r in rows))
    for al, r in res["structure_1d"].items():
        print("structure alpha", al, r)
    jobs = [(N, r, c, v) for N in (32, 64) for r in (1.0, 1e3, 1e6) for c in (0.5, 0.01, 0.99) for v in ("cross", "run")]
    with ProcessPoolExecutor(18) as ex:
        fs = [(j, ex.submit(layered, j[0], j[1], (j[0] // 4 + 1 - j[2]) / j[0], variant=j[3])) for j in jobs]
        res["layered"] = []
        for (N, r, c, v), f in fs:
            o = f.result()
            o["c"], o["variant"] = c, v
            res["layered"].append(o)
            print("layered", v, "N", N, "ratio", r, "c", c, "jump %.1e gram %.1f" % (o["face_jump"], o["gram_max"]),
                  " ".join(f"alpha {al}: null_By {v['null_By']} min {v['min']:.2e} max {v['max']:.2f} zero {v['near_zero']} dist {v['dist']}"
                           for al, v in o["alphas"].items()), flush=True)
    json.dump(res, open("docs/figures/projection_gate.json", "w"), indent=1, default=float)


def supplement() -> None:
    """docs/notes/104 supplement: volume averages for the x-face fluxes, and the interval-mobility reference."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(N, r, c, xa, ref) for N in (32, 64) for r in (1.0, 1e3, 1e6) for c in (0.5, 0.01)
            for xa in ("mass", "volume") for ref in ("point", "interval")]
    out = []
    with ProcessPoolExecutor(18) as ex:
        fs = [(j, ex.submit(layered, j[0], j[1], (j[0] // 4 + 1 - j[2]) / j[0], xavg=j[3], ref=j[4])) for j in jobs]
        for (N, r, c, xa, ref), f in fs:
            o = f.result()
            row = {"N": N, "ratio": r, "c": c, "xavg": xa, "ref": ref,
                   "alphas": {al: {k: v[k] for k in ("null_By", "min", "max", "near_zero")} for al, v in o["alphas"].items()}}
            out.append(row)
            print(N, r, c, xa, ref, " ".join(f"a{al}: null {v['null_By']} min {v['min']:.2e} max {v['max']:.2e}"
                                              for al, v in row["alphas"].items()), flush=True)
    vdev = {r: float(np.abs(_volume_average(32, r, (8 + 0.5) / 32, 0.7137) - np.eye(32)).max()) for r in (1.0, 1e3, 1e6)}
    print("max |V - I|", vdev)
    json.dump({"rows": out, "volume_minus_identity": vdev}, open("docs/figures/projection_gate_supplement.json", "w"),
              indent=1, default=float)


def figure() -> None:
    """docs/figures/projection_gate.png: the 1-D Schur ratio L(theta)/theta^2 for several alpha, and the smallest
    generalised eigenvalue of (L, L2) against the density ratio for the layered two-phase case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ("#257de0", "#f47230", "#00b894", "#f2a600")
    th, g, I0 = _tables(1025)
    s = np.sin(th / 2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    ax = axes[0]
    for c, al in zip(cols, (0.0, 1.0, 1.2, 1.5)):
        b2 = (2 * s) ** 2 * (I0 ** 2 + (al * s ** 5) ** 2)
        ax.plot(th[1:] / math.pi, b2[1:] / g[1:] / th[1:] ** 2, color=c, lw=2, label=f"alpha = {al}")
    ax.plot(th[1:] / math.pi, 4 * s[1:] ** 2 / th[1:] ** 2, "--", color="#666666", lw=1, label="MAC")
    ax.set_xlabel("theta / pi", fontsize=8)
    ax.set_title("single phase: Schur symbol / theta^2", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    res = json.load(open("docs/figures/projection_gate.json"))
    ax = axes[1]
    for c, (v, al, lab) in zip(cols, (("cross", "1.2", "alpha 1.2, across the interface"),
                                       ("cross", "0.0", "alpha 0 (exact null; drawn at 1e-16)"))):
        rows = [r for r in res["layered"] if r["variant"] == v and r["N"] == 64 and r["c"] == 0.5]
        rows.sort(key=lambda r: r["ratio"])
        ax.loglog([r["ratio"] for r in rows], [max(r["alphas"][al]["min"], 1e-16) for r in rows], "-o", color=c,
                  lw=2, ms=5, label=lab)
    ax.set_xlabel("density ratio", fontsize=8)
    ax.set_title("two phase (y layers, N = 64): min eig(L, L2)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
    fig.tight_layout()
    fig.savefig("docs/figures/projection_gate.png")


if __name__ == "__main__":
    import sys

    if "--figure" in sys.argv:
        figure()
    elif "--supplement" in sys.argv:
        supplement()
    else:
        main()

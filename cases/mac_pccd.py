"""docs/notes/105 (Gate 1-A2, the user's proposal): divergence-consistent MAC-PCCD, single phase, periodic.

State: the face-normal (face-averaged) velocities only. In 1-D, with face values f_i (face i left of cell i):

    d_i = (f_{i+1} - f_i) / h                     (exact cell mean of the derivative)
    (R_F f)(x) = f_i + int_{x_i}^x (R_C d)        (R_C the cell-average PCCD reconstruction)

R_F reproduces f at both faces of every cell (R_C keeps the means), and d/dx R_F = R_C D. In 2-D,
u_h = (R_F,x (x) R_C,y) u_f, v_h = (R_C,x (x) R_F,y) v_f gives div u_h = (R_C,x (x) R_C,y)(D_x u_f + D_y v_f): a discretely
divergence-free state reconstructs to a pointwise divergence-free field. Energy G = R^* R, projection
Pi = I - G^{-1} B^T (B G^{-1} B^T)^+ B, pressure force G^{-1} B^T H_p p with H_p the PCCD Gram of the pressure averages.

    python -m cases.mac_pccd
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, cell0, reconstruction_operator

M5 = np.array([[1.0 / (p + q + 1) for q in range(5)] for p in range(5)])
M6 = np.array([[1.0 / (p + q + 1) for q in range(6)] for p in range(6)])


# --------------------------------------------------------------------------------------------- symbols (h = 1)
def face_coeffs(theta):
    """s-coefficients (6) of R_F on cell 0 for the faces f_j = e^{i j theta}."""
    c = cell0(theta) * (np.exp(1j * theta) - 1.0)            # R_C of d_j = f_{j+1} - f_j
    return np.concatenate([[1.0], c / np.arange(1, 6)])


def g_C(theta):
    c = cell0(theta)
    return float((np.conj(c) @ M5 @ c).real)


def g_F(theta):
    c = face_coeffs(theta)
    return float((np.conj(c) @ M6 @ c).real)


def _grid(n):
    th = np.linspace(0.0, math.pi, n)
    gc = np.array([1.0] + [g_C(t) for t in th[1:]])
    gf = np.array([g_F(t) for t in th])
    return th, gc, gf


def symbols_table(n=4097):
    """Gram ranges and the Schur ratios L(theta) / |theta|^2: MAC-PCCD, the classical MAC (G = I) and the cell-centred
    PCCD with the plain face values (docs/notes/104)."""
    th, gc, gf = _grid(n)
    s2 = 4 * np.sin(th / 2) ** 2
    k = th > 0
    out = {"g_C": [float(gc.min()), float(gc.max())], "g_F": [float(gf.min()), float(gf.max())],
           "theta_gF_min": float(th[np.argmin(gf)] / math.pi)}
    out["mac_pccd_1d"] = float((s2[k] / gf[k] / th[k] ** 2).min())
    L = s2[:, None] / (gf[:, None] * gc[None, :]) + s2[None, :] / (gc[:, None] * gf[None, :])
    den = th[:, None] ** 2 + th[None, :] ** 2
    m = den > 0
    out["mac_pccd_2d"] = float((L[m] / den[m]).min())
    out["mac_pccd_2d_max"] = float((L[m] / den[m]).max())
    out["mac_1d"] = float((s2[k] / th[k] ** 2).min())
    out["mac_2d"] = float(((s2[:, None] + s2[None, :])[m] / den[m]).min())
    i, j = np.unravel_index(np.argmin(np.where(m, L / np.where(m, den, 1.0), np.inf)), L.shape)
    out["theta_min_2d"] = [float(th[i] / math.pi), float(th[j] / math.pi)]
    return out


def projection_accuracy(thetas=(1.6, 0.8, 0.4, 0.2, 0.1), angles=24):
    """2-D Fourier: exact face averages of a divergence-free plane wave (stay) and of a gradient wave (should vanish).
    u on x-faces (x = i, y-cell j), v on y-faces; amplitudes relative to e^{i(theta_x i + theta_y j)}:
    u = i k_x sigma_y e^{i theta_y / 2}, v = i k_y sigma_x e^{i theta_x / 2} (gradient wave of e^{i k.x}).
    Errors in the G norm, max over directions."""
    out = []
    for r in thetas:
        edf = ecf = 0.0
        for t in np.linspace(0.0, math.pi / 2, angles):
            tx, ty = r * math.cos(t), r * math.sin(t)
            sig = lambda a: 1.0 if a == 0 else math.sin(a / 2) / (a / 2)  # noqa: E731
            G = np.diag([g_F(tx) * g_C(ty), g_C(tx) * g_F(ty)])
            b = np.array([np.exp(1j * tx) - 1.0, np.exp(1j * ty) - 1.0])[None, :]
            Gi = np.linalg.inv(G)
            P = np.eye(2) - Gi @ b.conj().T @ np.linalg.inv(b @ Gi @ b.conj().T) @ b
            ucf = np.array([1j * tx * sig(ty) * np.exp(0.5j * ty), 1j * ty * sig(tx) * np.exp(0.5j * tx)])
            udf = np.array([-1j * ty * sig(ty) * np.exp(0.5j * ty), 1j * tx * sig(tx) * np.exp(0.5j * tx)])
            nrm = lambda v: math.sqrt((np.conj(v) @ G @ v).real)  # noqa: E731
            edf = max(edf, nrm(P @ udf - udf) / nrm(udf))
            ecf = max(ecf, nrm(P @ ucf) / nrm(ucf))
        out.append({"theta": r, "div_free_err": edf, "grad_err": ecf})
    return out


# --------------------------------------------------------------------------------------------- matrices (periodic)
def operators(N):
    """(RF (N, 6, N), RC (N, 5, N), D (N cells x N faces), G_F, H_p) with h = 1 integrals (multiply by h)."""
    RC = reconstruction_operator(N)
    D = np.roll(np.eye(N), -1, axis=0) - np.eye(N)              # (D f)_i = f_{i+1} - f_i
    RF = np.zeros((N, 6, N))
    RF[:, 0, :] = np.eye(N)
    for p in range(5):
        RF[:, p + 1, :] = (RC[:, p, :] @ D) / (p + 1)
    GF = np.einsum("ipj,pq,iqk->jk", RF, M6, RF)
    Hp = np.einsum("ipj,pq,iqk->jk", RC, M5, RC)
    return RF, RC, D, GF, Hp


def identities(N=32):
    RF, RC, D, GF, Hp = operators(N)
    f = np.random.default_rng(0).normal(size=N)
    at0 = np.einsum("ipj,j->ip", RF, f)
    end = at0.sum(axis=1)                                        # value at s = 1
    deriv = at0[:, 1:] * np.arange(1, 6)                         # d/ds coefficients
    return {"face_left": float(np.abs(at0[:, 0] - f).max()), "face_right": float(np.abs(end - np.roll(f, -1)).max()),
            "dRF_eq_RC_D": float(np.abs(deriv - np.einsum("ipj,j->ip", RC, D @ f)).max()),
            "GF_one": float(np.abs(GF @ np.ones(N) - 1.0).max()),
            "GF_eigs": [float(x) for x in np.linalg.eigvalsh(GF)[[0, -1]]]}


def pressure_gradient(N, k=1):
    """1-D: a = G_F^{-1} B^T H_p p (B = D / h, integrals with h) from the exact cell averages of
    p = sin(2 pi k x) + 0.3 cos(6 pi k x); compared with -p' at the faces (relative max error), and the classical MAC
    difference of the averages. In 2-D the y factor is G_C^{-1} H_p = I (both PCCD), so the x-gradient on the x-faces
    is this 1-D operator applied row by row to the y-cell averages: the 1-D test covers it."""
    RF, RC, D, GF, Hp = operators(N)
    h = 1.0 / N
    xf = np.arange(N) * h
    w = 2 * math.pi * k
    P = lambda x: -np.cos(w * x) / w + 0.3 * np.sin(3 * w * x) / (3 * w)  # noqa: E731  (primitive)
    p = (P(xf + h) - P(xf)) / h
    dp = w * np.cos(w * xf) - 0.9 * w * np.sin(3 * w * xf)
    a = np.linalg.solve(GF * h, (D / h).T @ (Hp * h) @ p)
    mac = (p - np.roll(p, 1)) / h                                # classical MAC: (p_i - p_{i-1}) / h at face i
    return {"mac_pccd": float(np.abs(a + dp).max() / np.abs(dp).max()),
            "mac": float(np.abs(mac - dp).max() / np.abs(dp).max())}


def structure_2d(N=12, seed=0):
    """Full 2-D matrices: projection identities and the pointwise divergence of the reconstructed projected field."""
    RF, RC, D, GF, Hp = operators(N)
    I = np.eye(N)
    Gu, Gv = np.kron(GF, Hp), np.kron(Hp, GF)                  # u index (x face i, y cell j) -> i * N + j
    G = np.block([[Gu, np.zeros_like(Gu)], [np.zeros_like(Gu), Gv]])
    B = np.hstack([np.kron(D, I), np.kron(I, D)])
    Gi = np.linalg.inv(G)
    L = B @ Gi @ B.T
    Pi = np.eye(2 * N * N) - Gi @ B.T @ np.linalg.pinv(L, rcond=1e-12) @ B
    rng = np.random.default_rng(seed)
    w = rng.normal(size=2 * N * N)
    x = Pi @ w
    u, v = x[:N * N].reshape(N, N), x[N * N:].reshape(N, N)
    # pointwise divergence at random points: du/dx from RF (x) RC, dv/dy from RC (x) RF
    div = 0.0
    for _ in range(50):
        i, j = rng.integers(N), rng.integers(N)
        sx, sy = rng.random(), rng.random()
        mx5, my5 = sx ** np.arange(5), sy ** np.arange(5)
        mx6, my6 = sx ** np.arange(6), sy ** np.arange(6)
        dmx6 = np.concatenate([[0.0], np.arange(1, 6) * sx ** np.arange(5)])
        dmy6 = np.concatenate([[0.0], np.arange(1, 6) * sy ** np.arange(5)])
        ux = (dmx6 @ RF[i]) @ u @ (my5 @ RC[j])
        vy = (mx5 @ RC[i]) @ v @ (dmy6 @ RF[j])
        div = max(div, abs(ux + vy))
    ev = np.linalg.eigvalsh(0.5 * (L + L.T))
    one = np.ones(N * N)
    return {"null_L": int((ev < 1e-10 * ev[-1]).sum()), "idempotent": float(np.abs(Pi @ Pi - Pi).max()),
            "B_Pi": float(np.abs(B @ Pi).max()), "energy_max": float(np.linalg.eigvalsh(Pi.T @ G @ Pi - G)[-1]),
            "momentum": float(max(abs(one @ Gu @ (x[:N * N] - w[:N * N])), abs(one @ Gv @ (x[N * N:] - w[N * N:])))),
            "pointwise_div": float(div), "scale": float(np.abs(x).max())}


def main() -> None:
    res = {"symbols": symbols_table(), "identities": identities(32), "structure_2d": structure_2d(12),
           "projection_accuracy": projection_accuracy(),
           "pressure_gradient": {N: pressure_gradient(N) for N in (8, 12, 16, 24, 32, 48, 64, 96, 128)}}
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open("docs/figures/mac_pccd.json", "w"), indent=1, default=float)


def figure() -> None:
    """docs/figures/mac_pccd.png: the 1-D Schur symbols / theta^2 and the 2-D minimum over directions, and the 1-D
    pressure-gradient error against N."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.pccd import face_symbol

    def face_interp(t, al):          # docs/notes/104: face-centred PCCD face value + i alpha sin^5(theta / 2)
        return face_symbol(t)[0] * np.exp(0.5j * t) + 1j * al * math.sin(t / 2) ** 5

    BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#666666"
    th, gc, gf = _grid(1025)
    s2 = 4 * np.sin(th / 2) ** 2
    k = th > 0
    I12 = np.array([abs(face_interp(t, 1.2)) ** 2 for t in th])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    ax = axes[0]
    ax.plot(th[k] / math.pi, s2[k] / gf[k] / th[k] ** 2, color=BLUE, lw=2, label="MAC-PCCD")
    ax.plot(th[k] / math.pi, s2[k] * I12[k] / gc[k] / th[k] ** 2, color=ORANGE, lw=2, label="cell PCCD + delta^4, alpha 1.2")
    ax.plot(th[k] / math.pi, s2[k] / th[k] ** 2, "--", color=GRAY, lw=1, label="classical MAC")
    ax.set_xlabel("theta / pi", fontsize=8)
    ax.set_title("1-D Schur symbol / theta^2 (single phase)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    res = json.load(open("docs/figures/mac_pccd.json"))
    Ns = sorted(int(n) for n in res["pressure_gradient"])
    ax = axes[1]
    ax.loglog(Ns, [res["pressure_gradient"][str(n)]["mac_pccd"] for n in Ns], "-o", color=BLUE, lw=2, ms=4, label="MAC-PCCD (G_F^-1 B^T H_p p)")
    ax.loglog(Ns, [res["pressure_gradient"][str(n)]["mac"] for n in Ns], "-o", color=GRAY, lw=2, ms=4, label="classical MAC")
    ax.set_xlabel("N", fontsize=8)
    ax.set_title("1-D pressure gradient at the faces (rel. max error)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(True, color="#e6e6e6", lw=0.6)
    fig.tight_layout()
    fig.savefig("docs/figures/mac_pccd.png")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()

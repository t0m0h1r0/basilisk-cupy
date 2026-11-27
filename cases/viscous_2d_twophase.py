"""docs/notes/106 (Gate 1-D, audit issue #120 (d), the user's V6-3a/b): the 2-D full viscous stress tensor with a
two-phase y reconstruction (interfaces on y-lines) and Fourier in x: per theta_x a 2N x 2N matrix.

With the y basis Y_j of the minimum-viscous-energy reconstruction (docs/notes/100) and the 1-D x symbols a_x, b_x, c_x
(docs/notes/102), the energy int mu (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2) gives

    K_uu = 2 a_x B_y + b_x A_y,   K_vv = a_x B_y + 2 b_x A_y,   K_uv = c_x W_y,
    A_y = int mu Y' Y'^T,  B_y = int mu Y Y^T,  W_y = int mu Y' Y^T,

with the mass matrices G_C = b_x blockdiag(G_nu, G_nu) (method C) and G_D = g_A,x blockdiag(G_A,y, G_A,y) (method D,
g_A,x = h times the periodic PCCD Gram symbol). Note B_y != G_nu,y = int rho Y Y^T unless nu is constant.
Since (u_y + v_x)^2 <= 2 u_y^2 + 2 v_x^2, K <= 2 blockdiag(a_x B_y + b_x A_y, same) and

    lambda_max(K, G_D) <= (2 / g_A,x) (a_x lambda_max(B_y, G_A,y) + b_x lambda_max(A_y, G_A,y))   (the user's bound).

    python -m cases.viscous_2d_twophase
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, symbols
from cases.viscous_2d import symbols_1d
from cases.viscous_minenergy import MinEnergy, advection_basis
from cases.viscous_stress import Setup, mus


def y_matrices(N, ratio, a, b, mu_H, mu_L):
    """(A_y, B_y, W_y, G_nu, G_A, M) for the two-phase y reconstruction (8-point Gauss per phase piece)."""
    S = Setup(N, ratio, a, b, mu_H, mu_L)
    ME = MinEnergy(S, nodes=(XG, WG))
    h = S.h
    T = ME.T                                                    # face stresses per unit average
    Ys, dYs, wmu, wrho, Xall = [], [], [], [], []
    for i, (X, Wr, wl, wr) in enumerate(ME.cells):
        heavy = ((X - a) % 1.0) < ((b - a) % 1.0)
        rho = np.where(heavy, 1.0, 1.0 / ratio)
        mu = np.where(heavy, mu_H, mu_L)
        w = Wr / rho
        # the mass coordinate s at the points: mass from the cell's left face / m_i
        sl = X / h - i
        mass = np.zeros_like(sl)
        for s0, s1, hv in S.P[i]:
            r = 1.0 if hv else 1.0 / ratio
            mass += r * np.clip(sl - s0, 0.0, s1 - s0) * h
        s = mass / S.m[i]
        fl, fr = (i - 1) % N, i
        E = np.zeros((X.size, N))
        E[:, i] = 1.0
        Y = E + np.outer(wl, T[fl]) + np.outer(wr, T[fr])
        dY = (np.outer(1 - s, T[fl]) + np.outer(s, T[fr])) / mu[:, None]
        Ys.append(Y), dYs.append(dY), wmu.append(w * mu), wrho.append(Wr), Xall.append(X)
    Y, dY = np.concatenate(Ys), np.concatenate(dYs)
    wmu, wrho, X = np.concatenate(wmu), np.concatenate(wrho), np.concatenate(Xall)
    A = dY.T @ (wmu[:, None] * dY)
    B = Y.T @ (wmu[:, None] * Y)
    W = dY.T @ (wmu[:, None] * Y)
    Gnu = Y.T @ (wrho[:, None] * Y)
    BA = advection_basis(N, ratio, a, b, X)
    GA = BA.T @ (wrho[:, None] * BA)
    return {"A": A, "B": B, "W": W, "Gnu": Gnu, "GA": GA, "M": np.diag(S.m), "K1d": ME.K, "Gnu1d": ME.G}


def _lmax(K, G):
    L = np.linalg.cholesky(0.5 * (G + G.T))
    Li = np.linalg.inv(L)
    return float(np.linalg.eigvalsh(Li @ (0.5 * (K + K.T)) @ Li.conj().T)[-1])


def _eigs(K, G):
    L = np.linalg.cholesky(0.5 * (G + G.T))
    Li = np.linalg.inv(L)
    return np.linalg.eigvalsh(Li @ (0.5 * (K + K.conj().T)) @ Li.conj().T)


def matrices(tx, Ym, h):
    ax, bx, cx = symbols_1d(tx, h)
    A, B, W = Ym["A"], Ym["B"], Ym["W"]
    K = np.block([[2 * ax * B + bx * A, cx * W], [np.conj(cx) * W.T, ax * B + 2 * bx * A]])
    Z = np.zeros_like(A)
    GC = bx * np.block([[Ym["Gnu"], Z], [Z, Ym["Gnu"]]])
    gA = h * float(symbols(tx, 0.37)["gram"].real) if tx != 0 else h
    GD = gA * np.block([[Ym["GA"], Z], [Z, Ym["GA"]]])
    return K, GC, GD, ax, bx, gA


def check_single_phase(N=16, tx=0.7):
    """ratio 1, constant mu: the 2N rates of (K, G_C) equal the 2 x 2 Fourier rates over theta_y = 2 pi m / N."""
    from cases.viscous_2d import rates

    h = 1.0 / N
    Ym = y_matrices(N, 1.0, 0.2613, 0.7137, 1.0, 1.0)
    K, GC, _, _, _, _ = matrices(tx, Ym, h)
    ev = np.sort(_eigs(K, GC))
    ref = np.sort(np.concatenate([rates(tx, 2 * math.pi * m / N, h, 1.0, "C") for m in range(N)]))
    return {"rate_diff": float(np.abs(ev - ref).max() / ref.max()),
            "A_vs_K1d": float(np.abs(Ym["A"] - Ym["K1d"]).max() / np.abs(Ym["K1d"]).max()),
            "Gnu_vs_compact": float(np.abs(Ym["Gnu"] - Ym["Gnu1d"]).max() / np.abs(Ym["Gnu1d"]).max())}


def stiffness(N, ratio, kind, c, ntheta=17):
    """max over theta_x of lambda_max h^2 / nu_max for C and D, the user's bound for D, and the 1-D pieces."""
    h = 1.0 / N
    a = (N // 4 + 1 - c) * h
    mu_H, mu_L = mus(ratio, kind)
    nu = max(mu_H, mu_L * ratio)
    Ym = y_matrices(N, ratio, a, 0.7137, mu_H, mu_L)
    lB_A, lA_A = _lmax(Ym["B"], Ym["GA"]), _lmax(Ym["A"], Ym["GA"])
    lB_nu = _lmax(Ym["B"], Ym["Gnu"])
    out = {"N": N, "ratio": ratio, "kind": kind, "c": c, "C": 0.0, "D": 0.0, "bound_D": 0.0,
           "lambda_B_GA": lB_A * h * h / nu, "lambda_A_GA": lA_A * h * h / nu, "lambda_B_Gnu": lB_nu * h * h / nu,
           "gram_nu_M": _lmax(Ym["Gnu"], Ym["M"]), "B_over_nu_M": _lmax(Ym["B"], Ym["M"]) / nu,
           "D_1d": _lmax(Ym["K1d"], Ym["GA"]) * h * h / nu}
    for tx in np.linspace(0.0, math.pi, ntheta):
        K, GC, GD, ax, bx, gA = matrices(tx, Ym, h)
        lc, ld = _eigs(K, GC)[-1], _eigs(K, GD)[-1]
        bd = (2 / gA) * (ax * lB_A + bx * lA_A)
        out["C"] = max(out["C"], float(lc) * h * h / nu)
        if ld * h * h / nu > out["D"]:
            out["D"], out["theta_D"] = float(ld) * h * h / nu, float(tx / math.pi)
        out["bound_D"] = max(out["bound_D"], float(bd) * h * h / nu)
        out["bound_ok"] = bool(out.get("bound_ok", True) and ld <= bd * (1 + 1e-10))
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    print("single-phase check", check_single_phase())
    jobs = [(32, r, k, c) for r in (1.0, 1e3, 1e6) for k in ("mu", "water-air", "nu") for c in (0.5, 1e-2, 1e-4, 0.99)]
    res = {"check": check_single_phase(), "rows": []}
    with ProcessPoolExecutor(12) as ex:
        for j, f in [(j, ex.submit(stiffness, *j)) for j in jobs]:
            o = f.result()
            res["rows"].append(o)
            print(o["kind"], o["ratio"], o["c"], "C %.1f D %.1f (theta %.2f pi) bound %.1f ok %s | 1-D D %.1f | B/G_A %.1f A/G_A %.1f B/G_nu %.2f G_nu/M %.1f"
                  % (o["C"], o["D"], o["theta_D"], o["bound_D"], o["bound_ok"], o["D_1d"], o["lambda_B_GA"],
                     o["lambda_A_GA"], o["lambda_B_Gnu"], o["gram_nu_M"]), flush=True)
    json.dump(res, open("docs/figures/viscous_2d_twophase.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""監修（独立の検証）：ノート101 の「体積力（圧力勾配の型）では密度の跳びで 2〜3 次。応力の再構築を質量座標で
線形とすることの限界」を調べる。

開発側の実装（cases/viscous_walls.py の forcing("volume")）は、体積力を F_i = f h と集中化し、応力の関係
Hτ = D_b q + b には力を入れていない。ここでは、力を含めた最小化

    R_f q = argmin { ½∫μ|v'|² − ∫f v : A_ρ v = q }      （E-L：−(μv')' = f + λ_i ρ）

を使う。応力はセルごとに τ = (1 − s)τ_左 + sτ_右 + τ_p、τ_p(x) = −∫_{x_i}^x f + s(x)∫_{K_i} f（両方の面で 0、
純相のセルで f が一定なら恒等的に 0）。平均の差の厳密な関係は

    Hτ + p = D_b q + b,    p_g = ∫ψ_g τ_p/μ      （p は混合セルの両側の面だけ非零）

Galerkin（試験関数は R_ν e_i）を部分積分すると  G q̇ = −D_bᵀτ + F（F_i = ∫_{K_i} f）で、変わるのは p だけ。

 (1) 定常の Poiseuille（体積力）：集中化（開発側）／力を含めた形。
 (2) 静止からの始動（体積力、非定常）：参照解は層ごとの Chebyshev 選点法。

    python -m audit.body_force
"""
from __future__ import annotations

import json

import numpy as np

from cases.pccd import WG, XG
from cases.viscous_stress import mus
from cases.viscous_walls import Wall, exact_steady


def p_vector(W: Wall, f=1.0):
    """p_g = ∫ψ_g τ_p/μ（一定の体積力 f）。純相のセルでは τ_p = 0。"""
    N, h = W.N, W.h
    p = np.zeros(N + 1)
    for i, ps in enumerate(W.P):
        if len(ps) == 1:
            continue
        mi = W.m[i]
        R = 0.0
        for s0, s1, hv in ps:
            rho = 1.0 if hv else 1.0 / W.ratio
            mu = W.mu_H if hv else W.mu_L
            L = (s1 - s0) * h
            d = L * XG
            x_rel = s0 * h + d                     # セルの左の面からの距離
            s = (R + rho * d) / mi                 # 質量座標
            tp = -f * x_rel + s * f * h
            w = L * WG
            p[i + 1] += w @ (s * tp) / mu          # 右の面（ψ は上昇 = s）
            p[i] += w @ ((1 - s) * tp) / mu        # 左の面（ψ は下降 = 1 − s）
            R += rho * L
    return p


def steady(N, ratio, kind, a=0.3613):
    mu_H, mu_L = mus(ratio, kind)
    W = Wall(N, ratio, a, mu_H, mu_L)
    qe = W.averages(exact_steady(mu_H, mu_L, ratio, a, 0.0, 0.0, "volume", 1.0))
    F = W.forcing("volume", 1.0)
    Hi_D = np.linalg.solve(W.H, W.Db)
    K = W.Db.T @ Hi_D
    q_lumped = np.linalg.solve(K, F)
    p = p_vector(W)
    q_aware = np.linalg.solve(K, F + W.Db.T @ np.linalg.solve(W.H, p))
    n = np.abs(qe).max()
    return {"lumped": float(np.abs(q_lumped - qe).max() / n), "force_aware": float(np.abs(q_aware - qe).max() / n)}


# ------------------------------------------------------------------ 参照解（Chebyshev、2 層、壁 u = 0）
def _cheb(n):
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    c = np.where((k == 0) | (k == n), 2.0, 1.0) * (-1.0) ** k
    X = np.tile(x, (n + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return -D[::-1, ::-1], x[::-1]


def reference(mu_H, mu_L, ratio, a, t, f=1.0, n=64):
    """静止から体積力 f で始動した u(x, t) を返す関数（層ごとの Chebyshev 補間）。"""
    D0, xi = _cheb(n)
    m = n + 1
    lay = [(0.0, a, 1.0, mu_H), (a, 1.0, 1.0 / ratio, mu_L)]
    Ds = [D0 * 2 / (x1 - x0) for x0, x1, _, _ in lay]
    xs = [x0 + (xi + 1) * (x1 - x0) / 2 for x0, x1, _, _ in lay]
    # 未知数：全節点 2m 個。拘束 4 本：u(0) = 0、u(1) = 0、[u] = 0、[μu'] = 0 → 内部の節点で表す
    Cn = np.zeros((4, 2 * m))
    Cn[0, 0] = 1
    Cn[1, 2 * m - 1] = 1
    Cn[2, n] = 1; Cn[2, m] = -1
    Cn[3, :m] = mu_H * Ds[0][n]; Cn[3, m:] = -mu_L * Ds[1][0]
    bd = [0, n, m, 2 * m - 1]
    it = [k for k in range(2 * m) if k not in bd]
    E = np.zeros((2 * m, len(it)))
    E[it, np.arange(len(it))] = 1
    E[bd] = -np.linalg.solve(Cn[:, bd], Cn[:, it])
    Lf = np.zeros((2 * m, 2 * m))
    Lf[:m, :m] = mu_H * Ds[0] @ Ds[0] / 1.0
    Lf[m:, m:] = mu_L * Ds[1] @ Ds[1] * ratio
    A = (Lf @ E)[it]                                  # u̇_int = A u_int + g
    g = np.concatenate([np.full(m, f), np.full(m, f * ratio)])[it]
    ev, V = np.linalg.eig(A)
    us = -np.linalg.solve(A, g)                       # 定常
    c = np.linalg.solve(V, -us)
    ui = (us + (V @ (np.exp(ev * t) * c))).real
    uf = E @ ui

    def bary(xq, xn, fn):
        w = np.ones(m)
        w[1::2] = -1
        w[0] *= 0.5; w[-1] *= 0.5
        d = xq[:, None] - xn[None, :]
        hit = np.abs(d) < 1e-14
        d[hit] = 1.0
        num = (w / d) @ fn
        den = (w / d).sum(1)
        out = num / den
        r, cidx = np.where(hit)
        out[r] = fn[cidx]
        return out

    def u(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        k = x <= a
        if k.any():
            out[k] = bary(x[k], xs[0], uf[:m])
        if (~k).any():
            out[~k] = bary(x[~k], xs[1], uf[m:])
        return out
    return u


def startup(N, ratio, kind, t, a=0.3613, n_ref=64):
    mu_H, mu_L = mus(ratio, kind)
    W = Wall(N, ratio, a, mu_H, mu_L)
    qe = W.averages(reference(mu_H, mu_L, ratio, a, t, n=n_ref))
    F = W.forcing("volume", 1.0)
    K = W.Knu
    p = p_vector(W)
    out = {}
    L = np.linalg.cholesky(W.Gnu)
    Li = np.linalg.inv(L)
    ev, V = np.linalg.eigh(Li @ K @ Li.T)
    V = Li.T @ V
    for name, Feff in (("lumped", F), ("force_aware", F + W.Db.T @ np.linalg.solve(W.H, p))):
        qs = np.linalg.solve(K, Feff)
        qt = qs + V @ (np.exp(-ev * t) * (V.T @ (W.Gnu @ (0 - qs))))
        out[name] = float(np.abs(qt - qe).max() / np.abs(qe).max())
    return out


def main():
    res = {"steady": [], "startup": [], "ref_check": []}
    for kind, r in (("mu", 1e3), ("water-air", 1e3), ("nu", 1e3), ("water-air", 1.0)):
        for N in (32, 64, 128, 256):
            s = steady(N, r, kind)
            res["steady"].append({"kind": kind, "ratio": r, "N": N, **s})
            print("定常", kind, r, N, "集中化 %.2e  力を含む %.2e" % (s["lumped"], s["force_aware"]))
    for kind, r in (("mu", 1e3), ("water-air", 1e3)):
        mu_H, mu_L = mus(r, kind)
        for t in (1e-3, 1e-2, 1e-1):
            xx = np.linspace(0.003, 0.997, 400)
            d = np.abs(reference(mu_H, mu_L, r, 0.3613, t, n=64)(xx) - reference(mu_H, mu_L, r, 0.3613, t, n=96)(xx)).max()
            sc = np.abs(reference(mu_H, mu_L, r, 0.3613, t, n=96)(xx)).max()
            res["ref_check"].append({"kind": kind, "t": t, "ref_n64_vs_n96_rel": float(d / sc)})
            print("参照解の自己確認", kind, t, "n=64 と 96 の差（相対）%.1e" % (d / sc))
            for N in (32, 64, 128, 256):
                s = startup(N, r, kind, t, n_ref=96)
                res["startup"].append({"kind": kind, "ratio": r, "t": t, "N": N, **s})
                print("  始動", kind, r, "t=%g" % t, N, "集中化 %.2e  力を含む %.2e" % (s["lumped"], s["force_aware"]))
    json.dump(res, open("audit/body_force.json", "w"), indent=1)


if __name__ == "__main__":
    main()

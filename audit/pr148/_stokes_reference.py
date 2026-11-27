"""監修（独立の検証）：二層 Stokes の厳密な減衰率（ノート99 の 5 節、ノート101）を、開発側の
伝達行列の走査とは別の方法で確かめる。

方法：各層を Chebyshev の選点法（Lobatto 点）で離散化し、界面で u と μu' を連続にした一般化固有値問題を解く
（スペクトル精度）。周期（2 界面）と壁（u = 0、1 界面）。固有値は最小から数えるので、低いモードの取りこぼしは
構造上起きない。

    python -m audit.stokes_reference
"""
from __future__ import annotations

import numpy as np


def cheb(n):
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    c = np.where((k == 0) | (k == n), 2.0, 1.0) * (-1.0) ** k
    X = np.tile(x, (n + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return D, x          # x: 1 → −1


def layer(n, x0, x1):
    D, x = cheb(n)
    D = -D[::-1, ::-1]                 # 並びを −1 → 1 に
    x = x[::-1]
    L = x1 - x0
    return x0 + (x + 1) * L / 2, D * 2 / L


def rates(layers, bc, n=48, k=6):
    """layers: [(長さ, ρ, μ), ...] を左から。bc: "periodic" または "wall"（両端 u = 0）。
    ρ λ u = −(μ u')' の λ を小さい順に k 個。"""
    nl = len(layers)
    Ds, sizes = [], []
    x0 = 0.0
    for L, rho, mu in layers:
        _, D = layer(n, x0, x0 + L)
        Ds.append(D)
        x0 += L
    m = n + 1
    Nt = nl * m
    A = np.zeros((Nt, Nt))
    Bm = np.zeros((Nt, Nt))
    for j, (L, rho, mu) in enumerate(layers):
        s = slice(j * m, (j + 1) * m)
        A[s, s] = -mu * Ds[j] @ Ds[j]
        Bm[s, s] = rho * np.eye(m)
    def cont(jl, jr, row_u, row_t):
        # 層 jl の右端と層 jr の左端：u の連続、μu' の連続
        A[row_u] = 0; Bm[row_u] = 0
        A[row_u, jl * m + n] = 1; A[row_u, jr * m] = -1
        A[row_t] = 0; Bm[row_t] = 0
        A[row_t, jl * m:(jl + 1) * m] = layers[jl][2] * Ds[jl][n]
        A[row_t, jr * m:(jr + 1) * m] -= layers[jr][2] * Ds[jr][0]
    for j in range(nl - 1):
        cont(j, j + 1, j * m + n, (j + 1) * m)
    if bc == "periodic":
        cont(nl - 1, 0, (nl - 1) * m + n, 0)
    else:
        A[0] = 0; Bm[0] = 0; A[0, 0] = 1
        r = Nt - 1
        A[r] = 0; Bm[r] = 0; A[r, r] = 1
    # B は特異（拘束の行）：A^{-1}B の固有値 1/λ を取る（周期は零モードがあるので A を少しずらす）
    sh = 1.0
    ev = np.linalg.eigvals(np.linalg.solve(A + sh * Bm, Bm))
    ev = ev[np.abs(ev) > 1e-12]
    lam = 1 / ev - sh
    lam = lam[np.abs(lam.imag) < 1e-6 * (1 + np.abs(lam.real))].real
    return np.sort(lam)[:k]


def main():
    a, b, r = 0.2613, 0.7137, 1e3
    LH = b - a
    kinds = {"nu（動粘度一定）": (1.0, 1.0 / r), "mu（粘度一定）": (1.0, 1.0), "water-air": (1.0, 1.0 / 55)}
    print("周期（重い相 (a, b)、密度比 1e3）：最小の 5 個（零モードを含む）")
    for name, (muH, muL) in kinds.items():
        # 周期：重い層（長さ LH）+ 軽い層（長さ 1 − LH）
        for n in (40, 64):
            lam = rates([(LH, 1.0, muH), (1 - LH, 1 / r, muL)], "periodic", n=n, k=5)
            print(f"  {name:18s} n={n}: " + " ".join(f"{x:.8f}" for x in lam))
    print("単相・周期： (2π)² =", (2 * np.pi) ** 2, rates([(0.4, 1, 1), (0.6, 1, 1)], "periodic", k=4))
    print("壁（重い層 (0, 0.3613)、u = 0）：最小の 3 個")
    aw = 0.3613
    for name, (muH, muL) in kinds.items():
        lam = rates([(aw, 1.0, muH), (1 - aw, 1 / r, muL)], "wall", n=64, k=3)
        print(f"  {name:18s}: " + " ".join(f"{x:.8f}" for x in lam))
    print("開発側の関数の値：")
    from cases.viscous_stress import stokes_rate
    from cases.viscous_walls import wall_rates
    for name, (muH, muL) in kinds.items():
        print("  周期", name, stokes_rate(muH, muL, r, a, b, which=4))
        print("  壁  ", name, wall_rates(muH, muL, r, aw, which=3))


if __name__ == "__main__":
    main()

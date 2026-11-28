"""監修 C-3：必須の性能試験「斜めの二層せん断流 + Galilean boost」（ノート111 の 6 節）の参照解。
A のコード（cases/）とライブラリを使わない（numpy だけ）。

問題（周期の [0, 1]^2、n = (a, b) は整数の組）
    z = a (x - Ux t) + b (y - Uy t)  (mod 1),     t_hat = (-b, a) / |n|
    u(x, y, t) = U + t_hat w(z, t),   p = 定数,   rho = rho(z),  mu = mu(z)
    rho dw/dt = |n|^2 d/dz (mu dw/dz),   界面で w と mu dw/dz が連続
    相 1（密度 rho1、粘性 mu1）は alpha <= z < beta の帯、残りが相 2。

導出の要点：grad u = (t_hat ⊗ n) w'、t_hat·n = 0 なので (u·grad)u = U·grad u、
div[mu (grad u + grad u^T)] = |n|^2 (mu w')' t_hat。界面は一様な U で並進するだけで動かない（共動系）。

参照解：層ごとに複数の要素を持つ Legendre のスペクトル要素法（弱形式は対称）。固有対 K phi = lambda M phi
（M は密度の重み、phi は M で正規直交）を作り、時間は厳密：w(z, t) = Σ c_k e^{-lambda_k t} phi_k(z)。
定数モード（lambda = 0）が総運動量 ∫rho w dz を保つ。

使い方
    ref = Shear(n=(1, 1), U=(1.0, 0.37), alpha=0.25, beta=0.75, rho=(1.0, 1e-3), mu=(1e-2, 1e-2 / 55))
    c = ref.coefficients(modes={1: 1.0, 2: 0.3}, mean=0.0)    # 低次の固有モードの重ね合わせ（界面の条件を満たす）
    u, v = ref.velocity_cells(N, c, t)                        # N x N のセル平均（[i, j]、x が i）
    f = ref.fraction_cells(N, t)                              # 相 1 の体積分率（厳密）
初期条件を任意の関数にする場合は ref.project(w0)。界面で mu w' が連続でない初期値は t = 0 で特異になるので、
収束の次数を測る試験には固有モードの重ね合わせを使う。
"""
from __future__ import annotations

import numpy as np


def lgl(p):
    """Legendre-Gauss-Lobatto の節点・重み・微分行列（次数 p）。"""
    x = -np.cos(np.pi * np.arange(p + 1) / p)
    for _ in range(60):
        P = np.zeros((p + 1, p + 1))
        P[:, 0] = 1.0
        P[:, 1] = x
        for k in range(2, p + 1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        dx = (x * P[:, p] - P[:, p - 1]) / ((p + 1) * P[:, p])
        x = x - dx
        if np.abs(dx).max() < 1e-15:
            break
    w = 2.0 / (p * (p + 1) * P[:, p] ** 2)
    L = P[:, p]
    D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = L[i] / (L[j] * (x[i] - x[j]))
    D[0, 0], D[p, p] = -p * (p + 1) / 4.0, p * (p + 1) / 4.0
    return x, w, D


class Shear:
    def __init__(self, n=(1, 1), U=(1.0, 0.37), alpha=0.25, beta=0.75, rho=(1.0, 1e-3), mu=(1e-2, 1e-2 / 55.0),
                 elements=8, order=16):
        if int(n[0]) != n[0] or int(n[1]) != n[1] or n[0] ** 2 + n[1] ** 2 == 0:
            raise ValueError("n = (a, b) must be integers (periodicity), not both zero")
        if not 0.0 <= alpha < beta <= alpha + 1.0:
            raise ValueError("0 <= alpha < beta <= alpha + 1")
        self.n, self.U = (int(n[0]), int(n[1])), (float(U[0]), float(U[1]))
        self.alpha, self.beta, self.rho, self.mu = alpha, beta, rho, mu
        self.k2 = float(n[0] ** 2 + n[1] ** 2)
        self.xi, self.w, self.D = lgl(order)
        p = order
        edges, phase = [], []
        for lo, hi, ph in ((alpha, beta, 0), (beta, alpha + 1.0, 1)):
            e = np.linspace(lo, hi, elements + 1)
            for j in range(elements):
                edges.append((e[j], e[j + 1]))
                phase.append(ph)
        ne = len(edges)
        ndof = ne * p                                            # 周期：最後の節点は最初の節点
        M = np.zeros(ndof)
        K = np.zeros((ndof, ndof))
        self.dofs = []
        for e, ((lo, hi), ph) in enumerate(zip(edges, phase)):
            idx = np.arange(e * p, e * p + p + 1) % ndof
            J = (hi - lo) / 2.0
            M[idx] += rho[ph] * self.w * J
            K[np.ix_(idx, idx)] += self.k2 * mu[ph] / J * (self.D.T * self.w) @ self.D
            self.dofs.append(idx)
        self.edges, self.phase, self.M, self.K = edges, phase, M, K
        s = 1.0 / np.sqrt(M)
        lam, V = np.linalg.eigh(s[:, None] * K * s[None, :])
        self.lam = np.where(np.abs(lam) < 1e-9 * max(1.0, np.abs(lam).max() * 1e-6), 0.0, lam)
        self.lam[0] = 0.0
        self.phi = s[:, None] * V                                # M-直交：phi.T diag(M) phi = I
        for k in range(ndof):                                    # 符号と大きさ：max|phi_k| = 1 に正規化する係数
            self.phi[:, k] *= np.sign(self.phi[np.argmax(np.abs(self.phi[:, k])), k])
        self.peak = np.abs(self.phi).max(axis=0)

    # ---- 係数 -------------------------------------------------------------------------------------------------
    def coefficients(self, modes=None, mean=0.0):
        """modes: {番号 k >= 1: max|w| で測った振幅}。mean: 定数の部分（w の定数モード）。"""
        c = np.zeros(len(self.lam))
        c[0] = mean / self.phi[0, 0]
        for k, a in (modes or {}).items():
            c[k] = a / self.peak[k]
        return c

    def nodes(self):
        z = np.zeros(len(self.M))
        for (lo, hi), idx in zip(self.edges, self.dofs):
            z[idx[:-1]] = lo + (hi - lo) * (self.xi[:-1] + 1) / 2
        return z

    def project(self, w0):
        """任意の周期関数 w0(z) の M-射影の係数。"""
        return self.phi.T @ (self.M * w0(self.nodes() % 1.0))

    # ---- 評価 -------------------------------------------------------------------------------------------------
    def nodal(self, c, t):
        return self.phi @ (c * np.exp(-self.lam * t))

    def w_at(self, z, c, t):
        """共動の座標 z（任意の配列）での w。"""
        v = self.nodal(c, t)
        z = (np.asarray(z) - self.alpha) % 1.0 + self.alpha
        out = np.zeros(z.shape)
        done = np.zeros(z.shape, bool)
        for (lo, hi), idx in zip(self.edges, self.dofs):
            m = (~done) & (z >= lo - 1e-13) & (z <= hi + 1e-13)
            if not m.any():
                continue
            s = 2 * (z[m] - lo) / (hi - lo) - 1
            # 重心形の Lagrange 補間（LGL）
            bw = np.array([1.0 / np.prod(self.xi[i] - np.delete(self.xi, i)) for i in range(len(self.xi))])
            d = s[:, None] - self.xi[None, :]
            hit = np.abs(d) < 1e-14
            d[hit] = 1.0
            q = bw / d
            val = (q @ v[idx]) / q.sum(1)
            row, col = np.nonzero(hit)
            val[row] = v[idx][col]
            out[m] = val
            done |= m
        return out

    def _cell_average(self, g, N, t, gauss=8):
        """g(z)（区分的に滑らか、z = alpha, beta で折れる）の N x N のセル平均。
        a b != 0 のとき z のセル内の分布は台形：x の方向を解析的な区間に分けて積分する。"""
        a, b = self.n
        h = 1.0 / N
        shift = (a * self.U[0] + b * self.U[1]) * t
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        # セル内の z の最小値と、各方向の広がり
        z0 = a * i * h + b * j * h + min(a, 0) * h + min(b, 0) * h - shift
        A, B = abs(a) * h, abs(b) * h
        if A < B:
            A, B = B, A
        xg, wg = np.polynomial.legendre.leggauss(gauss)
        out = np.zeros((N, N))
        # s = z - z0 の密度（A >= B）：[0, B] 上昇、[B, A] 平ら、[A, A + B] 下降
        pieces = [(0.0, A)] if B == 0 else [(0.0, B), (B, A), (A, A + B)]
        for ii in range(N):
            for jj in range(N):
                base = z0[ii, jj]
                total = 0.0
                for lo, hi in pieces:
                    if hi - lo <= 0:
                        continue
                    cuts = [lo, hi]
                    for zi in (self.alpha, self.beta):
                        k0 = np.ceil(base + lo - zi)
                        while zi + k0 < base + hi:
                            cuts.append(zi + k0 - base)
                            k0 += 1
                    cuts = np.unique(np.clip(cuts, lo, hi))
                    for c0, c1 in zip(cuts[:-1], cuts[1:]):
                        s = c0 + (c1 - c0) * (xg + 1) / 2
                        if B == 0:
                            dens = np.full_like(s, 1.0 / A)
                        else:
                            dens = np.minimum(np.minimum(s, B), A + B - s) / (A * B)
                        total += np.sum(wg * dens * g(base + s)) * (c1 - c0) / 2
                out[ii, jj] = total
        return out

    def w_cells(self, N, c, t):
        """格子の周期性を使い、異なる z の位相だけを計算する（z0 は h の整数倍だけずれる）。"""
        return self._by_phase(lambda z: self.w_at(z, c, t), N, t)

    def fraction_cells(self, N, t):
        ind = lambda z: (((z - self.alpha) % 1.0) < (self.beta - self.alpha)).astype(float)  # noqa: E731
        return self._by_phase(ind, N, t)

    def _by_phase(self, g, N, t):
        a, b = self.n
        h = 1.0 / N
        key = (a * np.arange(N)[:, None] + b * np.arange(N)[None, :]) % N       # z0 mod 1 = key h + 定数
        uniq, first = np.unique(key, return_index=True)
        full = np.zeros(N)
        shift = (a * self.U[0] + b * self.U[1]) * t
        A, B = abs(a) * h, abs(b) * h
        if A < B:
            A, B = B, A
        xg, wg = np.polynomial.legendre.leggauss(8)
        pieces = [(0.0, A)] if B == 0 else [(0.0, B), (B, A), (A, A + B)]
        for k, f0 in zip(uniq, first):
            i, j = divmod(int(f0), N)
            base = a * i * h + b * j * h + min(a, 0) * h + min(b, 0) * h - shift
            total = 0.0
            for lo, hi in pieces:
                if hi - lo <= 0:
                    continue
                cuts = [lo, hi]
                for zi in [self.alpha, self.beta] + [e[0] for e in self.edges]:   # 界面と要素の境界で分割
                    m = np.ceil(base + lo - zi)
                    while zi + m < base + hi:
                        cuts.append(zi + m - base)
                        m += 1
                cuts = np.unique(np.clip(cuts, lo, hi))
                for c0, c1 in zip(cuts[:-1], cuts[1:]):
                    s = c0 + (c1 - c0) * (xg + 1) / 2
                    dens = np.full_like(s, 1.0 / A) if B == 0 else np.minimum(np.minimum(s, B), A + B - s) / (A * B)
                    total += np.sum(wg * dens * g(base + s)) * (c1 - c0) / 2
            full[k] = total
        return full[key]

    def velocity_cells(self, N, c, t):
        w = self.w_cells(N, c, t)
        norm = np.sqrt(self.k2)
        return self.U[0] - self.n[1] / norm * w, self.U[1] + self.n[0] / norm * w

    def momentum(self, c, t):
        """∫ rho w dz（保存量）。"""
        return float(self.M @ self.nodal(c, t))

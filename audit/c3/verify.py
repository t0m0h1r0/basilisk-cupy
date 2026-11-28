"""参照解の検証（参照解自身と独立の方法で）：
(1) 固有値：層ごとの伝達行列の分散関係 tr(P2 P1) = 2 の符号の変化（別の定式化）。
(2) 解像度：要素数・次数を変えて固有値と w が変わらないこと。
(3) 時間発展：細かい有限体積（界面の格子に揃え、Crank–Nicolson）と比較。
(4) 密度・粘性の跳びが無い場合：厳密な Fourier 解と比較。(5) 運動量の保存、界面の条件、セル平均と体積分率。"""
import numpy as np
from oblique_shear_reference import Shear

def dispersion(ref, lam):
    P = np.eye(2)
    for ph, L in ((0, ref.beta - ref.alpha), (1, 1 - (ref.beta - ref.alpha))):
        rho, mu = ref.rho[ph], ref.mu[ph]
        k = np.sqrt(lam * rho / (ref.k2 * mu))
        P = np.array([[np.cos(k * L), np.sin(k * L) / (mu * k)], [-mu * k * np.sin(k * L), np.cos(k * L)]]) @ P
    return np.trace(P) - 2.0

for n in ((1, 1), (1, 2), (0, 1)):
    ref = Shear(n=n)
    fine = Shear(n=n, elements=12, order=20)
    print("n=%s  lambda_1..6:" % (n,), " ".join("%.8g" % v for v in ref.lam[1:7]))
    print("   解像度を上げた差（相対、1..12）: %.1e" % np.max(np.abs(ref.lam[1:13] - fine.lam[1:13]) / fine.lam[1:13]))
    # 分散関係：各固有値の前後で F の符号が変わる（単根）か、F が小さい（重根）こと
    for k in range(1, 7):
        l = ref.lam[k]
        a, b = dispersion(ref, l * (1 - 1e-6)), dispersion(ref, l * (1 + 1e-6))
        d = (b - a) / (2e-6 * l)
        print("   k=%d: tr-2 の零点までの相対距離（Newton の 1 歩） %.1e" % (k, abs(0.5 * (a + b) / d) / l))
# (3) 有限体積 + CN
ref = Shear(n=(1, 1))
c = ref.coefficients(modes={1: 1.0, 2: 0.4, 3: 0.2}, mean=0.1)
T = 1.0 / ref.lam[2]
for M in (400, 800, 1600):
    z = (np.arange(M) + 0.5) / M
    ph = np.where((z >= 0.25) & (z < 0.75), 0, 1)
    rho = np.array(ref.rho)[ph]; mu = np.array(ref.mu)[ph]
    muf = 2 * mu * np.roll(mu, -1) / (mu + np.roll(mu, -1))                    # 面 i+1/2（界面は面に揃う）
    h = 1.0 / M
    A = np.zeros((M, M))
    for i in range(M):
        for j, m in ((i + 1, muf[i]), (i - 1, muf[i - 1])):
            A[i, j % M] += ref.k2 * m / h ** 2; A[i, i] -= ref.k2 * m / h ** 2
    steps = 2000
    dt = T / steps
    S = np.linalg.solve(np.diag(rho) - 0.5 * dt * A, np.diag(rho) + 0.5 * dt * A)
    w = ref.w_at(z, c, 0.0)
    w = np.linalg.matrix_power(S, steps) @ w
    print("FV+CN M=%4d: max|w_fv - w_ref| = %.2e   運動量の変化 %.1e" % (M, np.abs(w - ref.w_at(z, c, T)).max(), abs(rho @ w - rho @ ref.w_at(z, c, 0)) * h))
# (4) 跳びなし
u = Shear(n=(1, 1), rho=(1.0, 1.0), mu=(1e-2, 1e-2))
cu = u.project(lambda z: np.sin(2 * np.pi * z) + 0.5 * np.cos(6 * np.pi * z))
z = np.linspace(0, 1, 401)
ex = np.sin(2 * np.pi * z) * np.exp(-2e-2 * 4 * np.pi ** 2 * 0.3) + 0.5 * np.cos(6 * np.pi * z) * np.exp(-2e-2 * 36 * np.pi ** 2 * 0.3)
print("一様な物性：Fourier の厳密解との差 %.1e" % np.abs(u.w_at(z, cu, 0.3) - ex).max())
# (5)
print("運動量 ∫rho w: t=0 %.12f, t=T %.12f" % (ref.momentum(c, 0), ref.momentum(c, T)))
for zi in (0.25, 0.75):
    e = 1e-6
    w = lambda z: ref.w_at(np.array([z]), c, T)[0]
    dl = (3 * w(zi) - 4 * w(zi - e) + w(zi - 2 * e)) / (2 * e); dr = (-3 * w(zi) + 4 * w(zi + e) - w(zi + 2 * e)) / (2 * e)
    ml, mr = (ref.mu[1], ref.mu[0]) if zi == 0.25 else (ref.mu[0], ref.mu[1])
    print("界面 z=%.2f: [w] %.1e   mu w' 左 %.8g 右 %.8g" % (zi, w(zi + 1e-12) - w(zi - 1e-12), ml * dl, mr * dr))
# セル平均：独立の直接積分（細分した中点則）と比較、体積分率の和
for n in ((1, 1), (1, 2), (0, 1)):
    r = Shear(n=n)
    cc = r.coefficients(modes={1: 1.0, 2: 0.4})
    N, t, m = 16, 0.13, 400
    x = (np.arange(N * m) + 0.5) / (N * m)
    brute = np.zeros((N, N))
    for i in range(N):
        xs = x[i * m:(i + 1) * m]
        for j in range(0, N):
            X, Y = np.meshgrid(xs, x[j * m:(j + 1) * m], indexing="ij")
            zz = r.n[0] * (X - r.U[0] * t) + r.n[1] * (Y - r.U[1] * t)
            brute[i, j] = r.w_at(zz, cc, t).mean()
        if i == 1:
            break
    wc = r.w_cells(N, cc, t); f = r.fraction_cells(N, t)
    print("n=%s セル平均：中点則（%d^2 点）との差 %.1e；体積分率の平均 %.12f（厳密 0.5）、[0,1] の中 %s" % (n, m, np.abs(wc[:2] - brute[:2]).max(), f.mean(), bool((f > -1e-12).all() and (f < 1 + 1e-12).all())))

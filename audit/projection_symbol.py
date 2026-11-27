"""監修（先回り、P2）：単相 PCCD の面の値から作る発散 B = Div∘FaceVal と、G 内積の厳密な射影の
Schur 補行列 L = B G⁻¹ Bᵀ の Fourier 記号（一次元、h = 1）。checkerboard の零空間を確かめる。

    python -m audit.projection_symbol
"""
import numpy as np

from cases.pccd import ccd_matrices, reconstruction_operator

N = 64
A, B = ccd_matrices(N)
Z = np.linalg.solve(A, B)             # (v, g) の面の値 ← セル平均
V = Z[0::2]                           # 面 i（セル i の左）の値
R = reconstruction_operator(N)
Mq = np.array([[1.0 / (p + q + 1) for q in range(5)] for p in range(5)])
G = np.einsum("ipj,pq,iqk->jk", R, Mq, R)        # h = 1
Dv = np.roll(np.eye(N), -1, axis=0) - np.eye(N)  # (Dv v)_i = v_{i+1} − v_i
Bm = Dv @ V
L = Bm @ np.linalg.solve(G, Bm.T)
F = np.fft.fft(np.eye(N)) / np.sqrt(N)
sym = lambda Mx: np.real(np.diag(F @ Mx @ F.conj().T))
th = 2 * np.pi * np.arange(N) / N
vs = np.abs(np.diag(F @ V @ F.conj().T))
Ls, Gs = sym(L), sym(G)
print("θ/π   |v̂(θ)|   Ĝ(θ)   L̂(θ)   L̂/θ²   L̂/(2−2cosθ)")
for k in (1, 4, 8, 16, 24, 28, 30, 31, 32):
    print(f"{th[k]/np.pi:.4f} {vs[k]:.3e} {Gs[k]:.4f} {Ls[k]:.3e} {Ls[k]/th[k]**2:.4f} {Ls[k]/(2-2*np.cos(th[k])):.4f}")
ev = np.sort(np.linalg.eigvalsh(0.5 * (L + L.T)))
print("L の固有値：最小 3 個", ev[:3], "最大", ev[-1], " → 零空間の次元（< 1e-10）:", int((ev < 1e-10).sum()))
# 二次元のテンソル積：L2(θx, θy) = Lx(θx)·(Gy⁻¹ の重み…) の代わりに、面の値を接線方向にセル平均のまま使う最も単純な形
# B2 = [Bx ⊗ I, I ⊗ By]、G2 = G ⊗ G。記号：L2 = |b(θx)|²/(Ĝ(θx)Ĝ(θy)) + |b(θy)|²/(Ĝ(θx)Ĝ(θy))
bs = np.abs(np.diag(F @ Bm @ F.conj().T)) ** 2
L2 = (bs[:, None] + bs[None, :]) / (Gs[:, None] * Gs[None, :])
lap = th[:, None] ** 2 + th[None, :] ** 2
k = N // 2
print("二次元（テンソル積）：L̂(π, π) = %.2e、L̂(π, 0) = %.3e、L̂(π, π/2)/(厳密 θ²) = %.3f" % (L2[k, k], L2[k, 0], L2[k, k // 2] / (np.pi ** 2 + (np.pi / 2) ** 2)))
print("二次元：零でない最小の L̂ / 最大の L̂ =", np.sort(L2.ravel())[2] / L2.max(), "（条件数の目安。5 点 Laplacian は", (2 - 2 * np.cos(th[1])) / 8, "）")

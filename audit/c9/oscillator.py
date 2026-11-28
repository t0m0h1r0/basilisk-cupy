"""監修 C-9 の 2：E1 が毛管波の減衰を増やす機構の予測。界面の変位 f と速度 u を 1 つの振動子（f' = u, u' = -w^2 f）で
表し、ソルバーの時間の組み方を写す（輸送は面の速度で f を動かし、そのあと f^{n+1} の表面張力で速度を直す）。
  base : f^{n+1} = f^n + dt u^n;                         u^{n+1} = u^n - dt w^2 f^{n+1}          （シンプレクティック Euler）
  E1   : f^{n+1} = f^n + dt (u^n + (u^n - u^{n-1})/2);   u^{n+1} = u^n - dt w^2 f^{n+1}
  E1m  : E1 で、力を (f^n + f^{n+1})/2 から作る
増幅率 lambda から、数値の減衰率 -ln|lambda|/dt を出す。"""
import numpy as np

def decay(kind, w, dt):
    z2 = (w * dt) ** 2
    # 状態 (f, v = dt u, vprev = dt u_prev)
    if kind == "base":
        F = np.array([[1, 1, 0]], float)
    else:
        F = np.array([[1, 1.5, -0.5]], float)
    src = (np.array([1.0, 0, 0]) + F[0]) / 2 if kind == "E1m" else F[0]
    M = np.array([F[0], np.array([0, 1, 0]) - z2 * src, [0, 1, 0]])
    lam = np.linalg.eigvals(M)
    osc = lam[np.argmax(np.abs(np.angle(lam)) * (np.abs(lam) > 0.5))]
    return -np.log(abs(osc)) / dt

if __name__ == "__main__":
    w = 15.712
    for n, h in ((32, 1 / 32), (64, 1 / 64), (128, 1 / 128)):
        dt = np.sqrt(0.5005 * h ** 3 / np.pi)
        print("N=%-3d dt=%.3e  numerical decay: base %+.5f  E1 %+.5f  E1m %+.5f   (w^2 dt/4 = %.5f)" % (n, dt, *[decay(k, w, dt) for k in ("base", "E1", "E1m")], w * w * dt / 4))
    print("measured extra damping (PR #177, capillary_wave): N=32 %.4f, N=64 %.4f" % (0.8600 - 0.7391, 0.7696 - 0.7291))

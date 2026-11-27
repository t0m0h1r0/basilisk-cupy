"""監修（C）：ノート113 の B 節の予測（E1・E2・E3、素朴な CN）を、A のコードもライブラリも使わない
擬スペクトルの理想化モデル（空間は厳密、分割なし）で確かめる。
  輸送：u* = u^n − dt ∇·(a ⊗ û)、û = u^n + (dt/2)(−(u^n·∇)u^n + s)、a = 運ぶ速度（n / ab 外挿 / 厳密な n+1/2）
  s = g^n（前のステップの射影の圧力の加速度）[+ νΔu^n]
  u1 = u* + dt g^n → 粘性（BE / CN（陽の半分は u^n）/ 素朴な CN（陽の半分は u1））→ −dt g^n → 射影、g^{n+1} を更新
誤差を s1 = w~、s2 = (U·∇)w~ に射影し、Σdt² で割る。予測：λ²/2 = 0.3117、±λ/2 = ±0.3948。"""
import math, sys
import numpy as np

N = 32
k1 = np.fft.fftfreq(N, 1.0 / N) * 2 * math.pi
KX, KY = np.meshgrid(k1, k1, indexing="ij")
K2 = KX ** 2 + KY ** 2
K2i = np.where(K2 == 0, 1.0, K2)
x1 = np.arange(N) / N
X, Y = np.meshgrid(x1, x1, indexing="ij")
F, I = np.fft.fft2, lambda a: np.real(np.fft.ifft2(a))
dx = lambda a: I(1j * KX * F(a))
dy = lambda a: I(1j * KY * F(a))
lap = lambda a: I(-K2 * F(a))

def psi(flow, x, y):
    tp = 2 * math.pi
    if flow == "tg":
        return np.sin(tp * x) * np.sin(tp * y) / tp
    return 0.02 * (np.sin(tp * (x + 2 * y)) + 0.6 * np.cos(tp * (2 * x - y)) + 0.4 * np.sin(tp * (2 * x + y) + 1.0))

def exact(flow, t, nu, U):
    kk = (8 if flow == "tg" else 20) * math.pi ** 2
    p = math.exp(-nu * kk * t) * psi(flow, X - U[0] * t, Y - U[1] * t)
    return U[0] + dy(p), U[1] - dx(p)

def project(u, v):
    d = 1j * KX * F(u) + 1j * KY * F(v)
    phi = d / (-K2i); phi[0, 0] = 0
    return u - I(1j * KX * phi), v - I(1j * KY * phi)

def run(flow, nu, U, steps, level, visc_src, visc, T=0.5):
    kk = (8 if flow == "tg" else 20) * math.pi ** 2
    lam = nu * kk
    dt = T / steps
    u, v = exact(flow, 0.0, nu, U)
    # g^0：厳密な圧力の加速度 = u_t + (u·∇)u − νΔu の勾配の部分（射影で作る）
    adv = lambda a, b, p, q: (a * dx(p) + b * dy(p), a * dx(q) + b * dy(q))
    cu, cv = adv(u, v, u, v)
    pu, pv = project(-cu, -cv)
    g = (pu + cu, pv + cv)                                  # −∇p = −(I − P)(u·∇u)
    prev = None
    for n in range(steps):
        t = n * dt
        if level == "n":
            a, b = u, v
        elif level == "exact":
            a, b = exact(flow, t + dt / 2, nu, U)
        else:
            a, b = (u, v) if prev is None else (1.5 * u - 0.5 * prev[0], 1.5 * v - 0.5 * prev[1])
        prev = (u, v)
        cu, cv = adv(u, v, u, v)
        su, sv = g
        Lu, Lv = nu * lap(u), nu * lap(v)
        if visc_src:
            su, sv = su + Lu, sv + Lv
        hu, hv = u + dt / 2 * (-cu + su), v + dt / 2 * (-cv + sv)
        us = u - dt * (dx(a * hu) + dy(b * hu))
        vs = v - dt * (dx(a * hv) + dy(b * hv))
        u1, v1 = us + dt * g[0], vs + dt * g[1]
        if visc == "be":
            u2, v2 = I(F(u1) / (1 + dt * nu * K2)), I(F(v1) / (1 + dt * nu * K2))
        else:
            eu, ev = (Lu, Lv) if visc == "cn" else (nu * lap(u1), nu * lap(v1))
            u2 = I(F(u1 + dt / 2 * eu) / (1 + dt / 2 * nu * K2))
            v2 = I(F(v1 + dt / 2 * ev) / (1 + dt / 2 * nu * K2))
        u3, v3 = u2 - dt * g[0], v2 - dt * g[1]
        un, vn = project(u3, v3)
        g = ((un - u3) / dt, (vn - v3) / dt)
        u, v = un, vn
    ue, ve = exact(flow, T, nu, U)
    ex, ey = u - ue, v - ve
    s1 = (ue - U[0], ve - U[1])
    s2 = (U[0] * dx(s1[0]) + U[1] * dy(s1[0]), U[0] * dx(s1[1]) + U[1] * dy(s1[1]))
    pr = lambda s: float(((ex * s[0]).sum() + (ey * s[1]).sum()) / ((s[0] ** 2).sum() + (s[1] ** 2).sum())) if np.abs(s[0]).max() > 0 else float("nan")
    sd = steps * dt * dt
    return float(np.hypot(ex, ey).max()), pr(s1) / sd, pr(s2) / sd, lam

if __name__ == "__main__":
    U = (1.0, 0.37)
    cases = [("n+g+BE  (E1+E2+E3)", "n", False, "be"), ("ab+g+BE (E2+E3)", "ab", False, "be"), ("ab+g+CN (E2)", "ab", False, "cn"),
             ("ab+visc+BE (E3)", "ab", True, "be"), ("ab+visc+naiveCN", "ab", True, "naive"), ("ab+visc+CN (none)", "ab", True, "cn"),
             ("exact+visc+CN", "exact", True, "cn"), ("n+visc+CN (E1)", "n", True, "cn")]
    for flow, nu in (("tg", 0.01), ("asym", 0.004)):
        for UU in (U, (0.0, 0.0), (-0.6, 1.1)):
            print("== %s  nu=%g  U=%s   predictions: lam^2/2 = %.4f, lam/2 = %.4f" % (flow, nu, UU, (nu * (8 if flow == "tg" else 20) * math.pi ** 2) ** 2 / 2, nu * (8 if flow == "tg" else 20) * math.pi ** 2 / 2))
            for name, level, vs, visc in cases:
                r = [run(flow, nu, UU, s, level, vs, visc) for s in (50, 100, 200)]
                print("  %-20s max %.2e %.2e %.2e (ratio %.2f)   c1 %+.4f %+.4f %+.4f   c2 %+.4f %+.4f %+.4f"
                      % (name, r[0][0], r[1][0], r[2][0], r[1][0] / r[2][0], r[0][1], r[1][1], r[2][1], r[0][2], r[1][2], r[2][2]), flush=True)

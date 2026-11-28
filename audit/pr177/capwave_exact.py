"""監修（C-2）：条件 9 の毛管波の厳密な線形の参照（初期値問題）。A の参照は線形の見積り 2k²Σμ/Σρ で、厳密でない。
ここでは、二層（有限の深さ h = 0.5、壁は滑り、密度比 1000、粘度比 55）の線形化した Navier–Stokes を、静止 + 変位 a0 の
初期値から Laplace 変換で解き（各層の解は sinh の閉じた形、界面の 4 条件）、数値で逆変換する。得た a(t) に、A と同じ
周期と減衰率の読み方（cases/r0_combined_tension.py）を適用する。固有モード（正規モード）の根も出す。"""
import math, json
import numpy as np
import mpmath as mp
mp.mp.dps = 50
k, h, sig, a0 = 2 * mp.pi, mp.mpf("0.5"), mp.mpf(1), mp.mpf("0.01")
rl, ru, ml, mu_ = mp.mpf(1), mp.mpf("1e-3"), mp.mpf("1e-2"), mp.mpf("1e-2") / 55

def system(s):
    m1 = mp.sqrt(k * k + s * rl / ml); m2 = mp.sqrt(k * k + s * ru / mu_)
    # 下層 phi = A sinh k(y+h) + B sinh m1(y+h)、上層 phi = C sinh k(h-y) + D sinh m2(h-y)（壁で phi = phi'' = 0）
    def lower(q, d): return q ** d * (mp.tanh(q * h) if d % 2 == 0 else 1)
    def upper(q, d): return (-q) ** d * (mp.tanh(q * h) if d % 2 == 0 else 1)
    L = lambda d: [lower(k, d), lower(m1, d), 0, 0]
    U = lambda d: [0, 0, upper(k, d), upper(m2, d)]
    comb = lambda *terms: [sum(c * v[j] for c, v in terms) for j in range(4)]
    rows = [comb((1, L(0)), (-1, U(0))),
            comb((1, L(1)), (-1, U(1))),
            comb((ml, L(2)), (ml * k * k, L(0)), (-mu_, U(2)), (-mu_ * k * k, U(0))),
            # (P_u - P_l)/k - 2k(mu_u - mu_l) phi' + sigma k^3 phi / s = -(i と eta0 の項)
            comb((mu_ / k, U(3)), (-mu_ * k - ru * s / k, U(1)), (-ml / k, L(3)), (ml * k + rl * s / k, L(1)),
                 (-2 * k * (mu_ - ml), L(1)), (sig * k ** 3 / s, L(0)))]
    return mp.matrix(rows), L(0)

def eta_hat(s):
    M, phi0 = system(s)
    # 法線の条件：... + sigma k^3 phi/s は eta = (eta0 - i k phi)/s の phi の部分。eta0 の部分を右辺へ。
    rhs = mp.matrix([0, 0, 0, -1j * sig * k ** 2 * a0 / s])
    x = mp.lu_solve(M, rhs)
    phi = sum(phi0[j] * x[j] for j in range(4))
    return (a0 - 1j * k * phi) / s

def measure(t, a):
    zc = [t[i] - a[i] * (t[i + 1] - t[i]) / (a[i + 1] - a[i]) for i in range(len(a) - 1) if a[i] * a[i + 1] < 0]
    period = float(2 * np.mean(np.diff(zc))) if len(zc) > 2 else None
    ext = [i for i in range(1, len(a) - 1) if abs(a[i]) >= abs(a[i - 1]) and abs(a[i]) >= abs(a[i + 1])]
    damping = float(-np.polyfit(t[ext], np.log(np.abs(a[ext])), 1)[0]) if len(ext) > 2 else None
    return period, damping, ext

if __name__ == "__main__":
    # 検算：非粘性の周波数 omega^2 = sigma k^3 / (rho_l coth kh + rho_u coth kh)
    w0 = mp.sqrt(sig * k ** 3 / ((rl + ru) * mp.coth(k * h)))
    print("inviscid frequency", mp.nstr(w0, 8), "period", mp.nstr(2 * mp.pi / w0, 8))
    det = lambda s: mp.det(system(s)[0])
    root = mp.findroot(det, mp.mpc(-0.8, w0), solver="muller") if False else mp.findroot(det, (mp.mpc(-0.5, w0), mp.mpc(-0.8, w0 * 0.99), mp.mpc(-1.0, w0 * 1.01)), solver="muller", maxsteps=100)
    print("viscous normal mode s =", mp.nstr(root, 8), "-> damping %.4f, period %.5f" % (-float(root.real), 2 * math.pi / abs(float(root.imag))))
    t = np.arange(1, int(1.2 / 0.004) + 1) * 0.004
    a = np.array([float(mp.invertlaplace(eta_hat, mp.mpf(float(ti)), method="dehoog", degree=24).real) for ti in t])
    print("check a(t -> 0) =", a[0], " (a0 = 0.01)")
    p, d, ext = measure(t, a)
    print("exact linear IVP, read as A reads it (t <= 1.2): period %.5f damping %.4f  (extrema at t = %s)" % (p, d, ", ".join("%.3f" % t[i] for i in ext)))
    json.dump({"t": t.tolist(), "a": a.tolist(), "period": p, "damping": d, "mode_damping": -float(root.real)}, open("audit/pr177/capwave_exact.json", "w"))
    A = json.load(open("docs/figures/r0_combined_tension.json"))
    for r in A:
        if r["case"] == "capillary_wave":
            print("A  %-10s N=%d period %.5f damping %.4f  (error vs exact: period %+.2f%%, damping %+.1f%%)" % (r["conf"], r["n"], r["period"], r["damping"], 100 * (r["period"] / p - 1), 100 * (r["damping"] / d - 1)))

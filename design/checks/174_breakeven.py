"""design/174 の確認と、R1 の結果から損益分岐点を計算する式（3・5 節）。

1. 時間刻み 0 の極限への外挿の重み（3.2 節）：u(c), u(c/2), u(c/4) から、c の 2 次式の定数項。
2. 要因計画の恒等式（4.2 節）：e00 = e11 + (e10 - e11) + (e01 - e11) + (e00 - e10 - e01 + e11)。
3. 固定の CFL での格子の粗さの上限 G_max = (1 + rho_s)^(1/m) と、目標に要る rho_s（3.3 節）。
4. 損益の計算（5 節）：測った誤差の曲線（べき乗則の当てはめ）と 1 ステップの時間から、
   同じ誤差に達する総計算時間、加速率 S、損益分岐の費用の比 r_c*。合成のデータで自己点検する。

実行：python design/checks/174_breakeven.py     （sympy・numpy。数秒）
"""

import numpy as np
import sympy as sp


# -- 1. zero-time-step extrapolation ------------------------------------------------------
def zero_dt_weights():
    a, b, q, c = sp.symbols("a b q c")
    u = lambda cc: a + b * cc + q * cc**2
    w = sp.symbols("w0 w1 w2")
    expr = sp.expand(w[0] * u(c) + w[1] * u(c / 2) + w[2] * u(c / 4) - a)
    sol = sp.solve([expr.coeff(a, 1), expr.coeff(b, 1), expr.coeff(q, 1)], w)
    return [sol[x] for x in w]


# -- 3. fixed CFL ----------------------------------------------------------------------------
def required_rho(r_c, m, d=2, target=1.5, dt_power=1.0):
    """Smallest rho_s = E_s / E_tau (baseline, operating grid) for a speed-up `target` when the interior spatial
    error is removed entirely; the remaining error is of order m; work ~ h^-(d + dt_power)."""
    return (target * r_c) ** (m / (d + dt_power)) - 1.0


# -- 4. break-even from measured curves ----------------------------------------------------------
def fit_power(h, e):
    """log e = log A + p log h (least squares). Returns (A, p)."""
    p, la = np.polyfit(np.log(h), np.log(e), 1)
    return float(np.exp(la)), float(p)


def h_for_error(eps, parts):
    """Largest h with sqrt(sum_k (A_k h^p_k)^2) <= eps (parts: list of (A, p), combined in squares)."""
    lo, hi = 1e-8, 1.0
    f = lambda h: np.sqrt(sum((A * h**p) ** 2 for A, p in parts))
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) <= eps else (lo, mid)
    return lo


def breakeven(eps, base_parts, new_parts, cost, d=2, dt_power=1.0, target=1.5):
    """base_parts / new_parts: error parts (A, p) of the baseline and of the scenario at the SAME CFL number.
    cost(h) -> seconds per step of the baseline at grid h (measured, (T)). Work = steps * cost, steps ~ h^-dt_power.
    Returns G = h'/h, the speed-up at equal step cost S1, and the break-even step-cost ratio r_c* (S = target)."""
    h0, h1 = h_for_error(eps, base_parts), h_for_error(eps, new_parts)
    w0 = cost(h0) / h0**dt_power
    w1 = cost(h1) / h1**dt_power
    s1 = w0 / w1
    return {"h_base": h0, "h_new": h1, "G": h1 / h0, "S_equal_cost": s1, "r_c_breakeven": s1 / target}


if __name__ == "__main__":
    w = zero_dt_weights()
    assert w == [sp.Rational(1, 3), -2, sp.Rational(8, 3)], w
    print("1. 時間刻み 0 への外挿：u0 = (u(c) - 6 u(c/2) + 8 u(c/4)) / 3   重み", w)

    e = sp.symbols("e00 e10 e01 e11")
    assert sp.expand(e[3] + (e[1] - e[3]) + (e[2] - e[3]) + (e[0] - e[1] - e[2] + e[3]) - e[0]) == 0
    print("2. 要因計画の恒等式：ok")

    print("3. 内部の空間誤差を 0 にしても目標（1.5 倍）に届くために要る rho_s = E_s/E_tau（d = 2）")
    for m, name in ((1, "残る誤差が 1 次"), (2, "残る誤差が 2 次")):
        for dtp, dname in ((1.0, "dt ∝ h"), (1.5, "dt ∝ h^1.5")):
            print("   %s、%s： " % (name, dname) + "  ".join("r_c = %g → %.2f" % (r, required_rho(r, m, dt_power=dtp)) for r in (1, 1.5, 2, 3)))
    assert abs(required_rho(1.0, 1) - (1.5 ** (1 / 3) - 1)) < 1e-12

    # 4. 合成のデータで自己点検：E = sqrt((As h^2)^2 + (At h^2)^2 + (Ab h^1.5)^2)、cost ∝ h^-2
    hs = np.array([1 / 32, 1 / 64, 1 / 128, 1 / 256])
    As, At, Ab = 4.0, 1.0, 0.05
    fitted = [fit_power(hs, A * hs**p) for A, p in ((As, 2), (At, 2), (Ab, 1.5))]
    assert all(abs(a - A) < 1e-9 * A and abs(p - P) < 1e-9 for (a, p), (A, P) in zip(fitted, ((As, 2), (At, 2), (Ab, 1.5))))
    cost = lambda h: 1e-7 / h**2
    r = breakeven(1e-4, fitted, [(As * 0.0 + 1e-300, 2), fitted[1], fitted[2]], cost)
    g_closed = ((As**2 + At**2) ** 0.5 / At) ** 0.5            # without the band: G = (E_tot/E_tau)^(1/2)
    print("4. 合成の例（内部の空間誤差 4 : 時間 1、帯つき、eps = 1e-4）：G = %.3f（帯なしの閉じた形 %.3f）、"
          "同じ費用での加速率 %.2f、損益分岐の r_c* = %.2f" % (r["G"], g_closed, r["S_equal_cost"], r["r_c_breakeven"]))
    assert r["G"] < g_closed and abs(r["S_equal_cost"] - r["G"] ** 3) < 1e-6 * r["G"] ** 3
    print("ok")

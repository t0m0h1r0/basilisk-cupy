"""design/179 の 2・4 節の確認：1 ステップの局所誤差の主要項（Δt² の係数）と、粘性の段の安定関数。

模型（空間は厳密、線形）：  u' = c(t) A u + V u
    A : 移流（-u.grad にあたる。運ぶ速度の大きさ c(t) だけが時間に依存）、V : 粘性。A と V は可換でない。
    c0 = c(t^n)、c1 = c'(t^n)。c1 のうち、粘性による分を c1v と書く（予測に粘性を入れない段が落とす分）。
    厳密な 1 ステップ： 1 + dt (c0 A + V) + dt^2/2 [(c0 A + V)^2 + c1 A] + O(dt^3)

段の模型
    輸送   T = 1 + dt (c0 + chi dt c1_used) A + dt^2/2 c0 A (c0 A + s V)
           chi = 0（運ぶ速度が t^n）／1/2（t^{n+1/2}）。s = 1：面の値の源の項に粘性の加速度あり（E2）、0：なし（帯の中）
    粘性   BE：(1 - dt V)^{-1} y ／ CN：(1 - dt V/2)^{-1} (y + dt/2 V u^n) ／ L2：輸送の後の y に 2 次の V の流れ

実行：python design/checks/179_local_error.py     （sympy だけ。数秒）
"""

import sympy as sp

dt, c0, c1, c1v = sp.symbols("dt c0 c1 c1v")
A, V = sp.symbols("A V", commutative=False)


def trunc(e):
    e = sp.expand(e)
    return sp.expand(sum(e.coeff(dt, k) * dt**k for k in range(3)))


EXACT = trunc(1 + dt * (c0 * A + V) + dt**2 / 2 * ((c0 * A + V) ** 2 + c1 * A))


def transport(chi, s, c1_used):
    return 1 + dt * (c0 + chi * dt * c1_used) * A + dt**2 / 2 * c0 * A * (c0 * A + s * V)


def step(chi, s, visc, c1_used=c1):
    T = transport(chi, s, c1_used)
    if visc == "BE":
        S = (1 + dt * V + dt**2 * V**2) * T
    elif visc == "CN":
        S = (1 + dt / 2 * V + dt**2 / 4 * V**2) * (T + dt / 2 * V)
    elif visc == "L2":
        S = (1 + dt * V + dt**2 / 2 * V**2) * T
    return sp.expand(sp.expand(trunc(S) - EXACT).coeff(dt, 2))


E1 = -c1 * A / 2                       # 運ぶ速度の時刻：ノート113 の +(dt^2/2)(u_t.grad)u（A = -u.grad）
E2 = -c0 * A * V / 2                   # 面の値の源の項に粘性が無い：+(dt^2/2) u.grad(nu Lap u)
E3 = V * (c0 * A + V) / 2              # 後退 Euler：+(dt^2/2) nu Lap u_t
Ec = -c1v * A / 2                      # 予測した運ぶ速度に粘性の加速度が無い

ROWS = [
    ("c  現行（uf^n、E2 あり、BE）", step(0, 1, "BE"), E1 + E3),
    ("c  同、帯の中（E2 なし）", step(0, 0, "BE"), E1 + E2 + E3),
    ("   ノート113 の全部（外挿 + E2 + CN）", step(sp.Rational(1, 2), 1, "CN"), 0),
    ("a  帯の中（外挿、E2 なし、BE）", step(sp.Rational(1, 2), 0, "BE"), E2 + E3),
    ("d1/b + BE（純相の内部）", step(sp.Rational(1, 2), 1, "BE"), E3),
    ("d1 + BE、予測に粘性なし", step(sp.Rational(1, 2), 1, "BE", c1 - c1v), E3 + Ec),
    ("d1/b + BE、帯の中", step(sp.Rational(1, 2), 0, "BE"), E2 + E3),
    ("d1/b + CN（単相：ノート113 の cons/pred+visc+cn）", step(sp.Rational(1, 2), 1, "CN"), 0),
    ("輸送の後に 2 次の粘性の流れ（L2、Lie の分割）", step(sp.Rational(1, 2), 1, "L2"), V * c0 * A / 2),
]

if __name__ == "__main__":
    print("局所誤差（方式 - 厳密）の dt^2 の係数")
    for name, got, want in ROWS:
        assert sp.expand(got - want) == 0, (name, got, want)
        print("  %-52s %s" % (name, got))

    # 粘性の段の安定関数 R(z)（z = dt * 固有値 <= 0）と、硬い極限
    zz = sp.symbols("z")
    g = 1 - 1 / sp.sqrt(2)
    R = {
        "BE": 1 / (1 - zz),
        "CN（陽的な半分 + 陰的な半分）": (1 + zz / 2) / (1 - zz / 2),
        "BE(dt/2) の予測 + その状態への陽的な適用": 1 + zz / (1 - zz / 2),
        "SDIRK2（gamma = 1 - 1/sqrt 2、BE 型の求解 2 回）": (1 + (1 - 2 * g) * zz) / (1 - g * zz) ** 2,
        "BE + 遅延補正（BE の求解 2 回）": (1 - zz - zz**2 / 2) / (1 - zz) ** 2,
    }
    print("\n粘性の段の安定関数：2 次か（exp z との差が O(z^3)）、R(-∞)")
    for k, r in R.items():
        order2 = sp.series(r - sp.exp(zz), zz, 0, 3).removeO() == 0
        print("  %-48s 2 次：%-5s R(-∞) = %s" % (k, order2, sp.limit(r, zz, -sp.oo)))
    assert sp.simplify(R["CN（陽的な半分 + 陰的な半分）"] - R["BE(dt/2) の予測 + その状態への陽的な適用"]) == 0

    # 7 節の仕事量の模型：誤差 eps = Cs h^p + Ct (c h)^q、仕事量 W ∝ 1/(c h^{d+1}) を、空間の誤差の割合 x で最小化
    x, d, p, q = sp.symbols("x d p q", positive=True)
    logW = -(1 / q) * sp.log(1 - x) - (d / p) * sp.log(x)          # W ∝ eps^{-(1/q + d/p)} / ((1-x)^{1/q} x^{d/p})
    xs = sp.solve(sp.diff(logW, x), x)[0]
    assert sp.simplify(xs - (d / p) / (1 / q + d / p)) == 0
    print("\n仕事量の模型：最適な空間の誤差の割合 x* =", sp.simplify(xs), "、W ∝ eps^-(1/q + d/p)")
    for qq in (1, 2):
        print("  q = %d：" % qq + "  ".join("p = %d → 指数 %s" % (pp, sp.Rational(1, qq) + sp.Rational(2, pp)) for pp in (2, 4, 6)) + "   (d = 2)")
    print("ok")

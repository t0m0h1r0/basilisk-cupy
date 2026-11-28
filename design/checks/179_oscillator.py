"""design/179 の 3 節（模型 M1）の確認：界面の変位 f と速度 v の線形の振動子  f' = v,  v' = -w^2 f  に、
各選択肢の「運ぶ速度の時刻」と「力を作る界面の時刻」の組を写したときの、1 ステップの増幅率。

    z = (w dt)^2,  v は dt を掛けた量で書く（f' = f + v_carrier,  v' = v - z * f_force）。

運ぶ速度（carrier）
    n   : v^n                                        （現行。conserving.h）
    ab2 : v^n + (v^n - v^{n-1})/2                    （E1。ノート113 の外挿、等間隔の時間刻み）
    fe  : v^n - (z/2) f^n                            （力 F(f^n) の半ステップの予測：選択肢 d）
    se  : 現行の 1 ステップを dt/2 で行った後の速度  （ft = f + v/2,  vt = v - (z/2) ft：選択肢 b）
力を作る界面（force）
    new : f^{n+1}                                     （現行）
    avg : (f^n + f^{n+1})/2                           （ポテンシャルの平均。監修 C-9 の反証の試験の形）
    mid : 半ステップの段の界面 ft                     （se のときだけ意味がある）
    lag : 予測の力に、前のステップの平均の力 (f^{n-1} + f^n)/2 を使う（保存してある g をそのまま予測に使う形）

実行：python design/checks/179_oscillator.py     （sympy・numpy だけ。数秒）
"""

import numpy as np
import sympy as sp

z = sp.symbols("z", positive=True)
f, v, vp, fp = sp.symbols("f v vp fp")


def scheme(carrier, force):
    """(f, v[, history]) -> the next state, as sympy expressions; returns (state symbols, next state)."""
    ft = None
    if carrier == "n":
        vc = v
    elif carrier == "ab2":
        vc = v + (v - vp) / 2
    elif carrier == "fe":
        vc = v - z / 2 * f
    elif carrier == "fe-lag":
        vc = v - z / 2 * (f + fp) / 2
    elif carrier == "se":
        ft = f + v / 2
        vc = v - z / 2 * ft
    f1 = f + vc
    if force == "new":
        ff = f1
    elif force == "avg":
        ff = (f + f1) / 2
    elif force == "mid":
        ff = ft
    v1 = v - z * ff
    if carrier == "ab2":
        return (f, v, vp), (f1, v1, v)
    if carrier == "fe-lag":
        return (f, v, fp), (f1, v1, f)
    return (f, v), (f1, v1)


def matrix(carrier, force):
    s, nxt = scheme(carrier, force)
    return sp.Matrix([[sp.diff(sp.expand(e), x) for x in s] for e in nxt])


def radius(M, zz):
    return float(max(abs(np.linalg.eigvals(np.array(M.subs(z, zz), dtype=float)))))


def physical_log(M, zz):
    """log |lambda| of the oscillatory pair (the roots that tend to exp(+-i w dt))."""
    ev = np.linalg.eigvals(np.array(M.subs(z, zz), dtype=float))
    if abs(ev.imag).max() < 1e-12:                    # no oscillatory pair left: the spectral radius
        return float(np.log(max(abs(ev).max(), 1e-300)))
    return float(np.log(abs(ev[np.argmax(abs(ev.imag))])))


def first_unstable(M):
    for zz in np.linspace(1e-3, 6.0, 6000):
        if radius(M, zz) > 1.0 + 1e-9:
            return zz
    return None


CASES = [
    ("c  現行（シンプレクティック Euler）", "n", "new"),
    ("   現行 + 力だけ平均（監修 C-9 の対照）", "n", "avg"),
    ("a  E1（外挿）、力は f^{n+1}", "ab2", "new"),
    ("a  E1（外挿）+ 力の平均", "ab2", "avg"),
    ("d0 conserving_predict そのまま（力は f^{n+1}）", "fe", "new"),
    ("d1 予測の力 F(f^n) + 力の平均（速度 Verlet）", "fe", "avg"),
    ("   d1 で予測に保存した g（前のステップの平均の力）", "fe-lag", "avg"),
    ("b0 半ステップの段 + 力は f^{n+1}", "se", "new"),
    ("b1 半ステップの段 + 力は中間の界面（位置 Verlet）", "se", "mid"),
    ("b2 半ステップの段 + 力の平均", "se", "avg"),
]

if __name__ == "__main__":
    print("log|lambda| は 1 ステップあたり（負 = 数値の減衰、正 = 増幅）。数値の減衰率は -log|lambda|/dt。")
    print("%-52s %-26s %-10s %s" % ("組", "log|lambda| の主要項", "不安定の始まり z", "log|lambda|（z = 0.01, 0.1, 0.5, 1, 2）"))
    for name, c, fo in CASES:
        M = matrix(c, fo)
        if M.shape[0] == 2:
            det, tr = sp.factor(M.det()), sp.expand(M.trace())
            lead = sp.series(sp.log(det) / 2, z, 0, 3).removeO()
            # 2 x 2 real: complex pair iff tr^2 < 4 det, then |lambda|^2 = det
            lead = "det = %s → %s" % (det, lead)
        else:
            c1 = physical_log(M, 1e-3) / 1e-3
            c2 = physical_log(M, 1e-2) / 1e-4
            lead = "%+.3f z" % c1 if abs(c1) > 1e-2 else "%+.4f z^2" % c2
        lim = first_unstable(M)
        vals = " ".join("%+.4f" % physical_log(M, zz) for zz in (0.01, 0.1, 0.5, 1.0, 2.0))
        print("%-52s %-26s %-10s %s" % (name, lead, "なし(<6)" if lim is None else "%.2f" % lim, vals))

    # 記号での確認（2 x 2 の 5 つ）
    assert sp.simplify(matrix("n", "new").det() - 1) == 0
    assert sp.simplify(matrix("fe", "avg").det() - 1) == 0 and sp.expand(matrix("fe", "avg").trace() - (2 - z)) == 0
    assert sp.simplify(matrix("se", "mid").det() - 1) == 0 and sp.expand(matrix("se", "mid").trace() - (2 - z)) == 0
    assert sp.simplify(matrix("fe", "new").det() - (1 - z / 2)) == 0
    assert sp.simplify(matrix("se", "new").det() - (1 - z / 2)) == 0
    assert sp.simplify(matrix("se", "avg").det() - (1 - z**2 / 8)) == 0
    assert sp.simplify(matrix("n", "avg").det() - (1 + z / 2)) == 0
    # 3 準位の 2 つ：振動の根の log|lambda| -> +z^2/8（z -> 0）
    for c in ("ab2", "fe-lag"):
        assert abs(physical_log(matrix(c, "avg"), 1e-2) / 1e-4 - 0.125) < 2e-3
    assert abs(physical_log(matrix("ab2", "new"), 1e-3) / 1e-3 + 0.25) < 1e-3

    # 監修 C-9 の毛管波（w = 15.712、dt = sqrt(0.5005 h^3 / pi)）での、数値の減衰率の予測
    print("\n監修 C-9 の毛管波の配置での余分の減衰率（-log|lambda|/dt。厳密な減衰率は 0.7147）")
    w = 15.712
    for n in (32, 64, 128):
        dt = np.sqrt(0.5005 / n**3 / np.pi)
        zz = (w * dt) ** 2
        row = "  ".join("%s %+.2e" % (k, -physical_log(matrix(c, fo), zz) / dt) for k, c, fo in (
            ("c", "n", "new"), ("a/E1", "ab2", "new"), ("a/E1+avg", "ab2", "avg"), ("d0", "fe", "new"),
            ("d1", "fe", "avg"), ("b1", "se", "mid"), ("b2", "se", "avg")))
        print("N = %-3d z = %.2e  %s   (w^2 dt/4 = %.3f)" % (n, zz, row, w * w * dt / 4))

    # 物理の減衰（v' = -2 gamma v を、力の後に後退 Euler で掛ける）を入れたとき、E1 + 力の平均が安定になる下限
    def damped(c, fo, gd):
        s, nxt = scheme(c, fo)
        nxt = list(nxt)
        nxt[1] = nxt[1] / (1 + 2 * gd)
        return sp.Matrix([[sp.diff(sp.expand(e), x) for x in s] for e in nxt])

    print("\nE1 + 力の平均が安定になる gamma*dt の下限（予測：z^2/8）")
    for zz in (0.1, 0.5, 1.0, 2.0):
        lo, hi = 0.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if radius(damped("ab2", "avg", mid), zz) > 1 else (lo, mid)
        print("  z = %-4g gamma*dt >= %.4f   (z^2/8 = %.4f)" % (zz, hi, zz * zz / 8))
        assert abs(hi - zz * zz / 8) < 0.25 * zz * zz / 8 + 1e-4
    for name, c, fo in (("c", "n", "new"), ("d1", "fe", "avg"), ("b1", "se", "mid"), ("b2", "se", "avg")):
        worst = max(radius(damped(c, fo, gd), zz) for gd in (0.0, 0.01, 0.1, 1.0, 10.0) for zz in np.linspace(0.01, 3.99, 200))
        assert worst <= 1 + 1e-9, (name, worst)
    print("  c・d1・b1・b2 は、gamma*dt = 0〜10、z < 4 の全てでスペクトル半径 <= 1")
    print("ok")

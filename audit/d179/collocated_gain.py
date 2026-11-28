"""監修 #179：模型 M1 に、近似射影（セルと面の 2 つの速度）の利得 G を入れる。

現行のソルバー（時間の段を線形化）：面の速度 uf^{n+1} = q^n + dt a^{n+1}（q = セルの速度の面への補間）、
セルの速度は q^{n+1} = q^n + dt G a^{n+1}（a は面→セル→面の平均を通る。長い波で G -> 1、格子の規模で G -> 0）。
a = -w^2 f_F。z = (w dt)^2。G = 1 で設計の M1 に一致する。"""
import numpy as np

def radius(name, z, G):
    # 状態から次の状態への線形写像を単位ベクトルで組み立てる
    def step(s):
        if name == "c":            # f, q(=q^{n-1}) : uf^n = q + a(f^n)
            f, q = s
            a = -z * f
            return [f + q + a, q + G * a]
        if name in ("a:E1", "a:E1+avg"):   # f^n, uf^n, uf^{n-1}, a^n
            f, uf, um, a = s
            f1 = f + 1.5 * uf - 0.5 * um
            a1 = -z * (f1 if name == "a:E1" else (f + f1) / 2)
            q = uf - a + G * a              # q^n = q^{n-1} + G a^n, uf^n = q^{n-1} + a^n
            return [f1, q + a1, uf, a1]
        if name == "d1":           # f, q^n : 運ぶ速度 = q + a(f^n)/2
            f, q = s
            f1 = f - z * f / 2 + q
            return [f1, q - G * z * (f + f1) / 2]
        if name == "b1":           # f, q^n : 半ステップ（現行を dt/2）：ft = f + uf/2。uf を q で代用できないので uf も持つ
            f, q, uf = s
            ft = f + uf / 2
            vt = q - z * ft / 2    # 半ステップの段の面の速度 = q(transport) + (dt/2) a(ft)
            f1 = f + vt
            a = -z * ft
            return [f1, q + G * a, q + a]
    n = {"c": 2, "a:E1": 4, "a:E1+avg": 4, "d1": 2, "b1": 3}[name]
    M = np.array([step(list(e)) for e in np.eye(n)]).T
    return max(abs(np.linalg.eigvals(M)))

if __name__ == "__main__":
    for name in ("c", "a:E1", "a:E1+avg", "d1", "b1"):
        print(name)
        for z in (0.5, 1.0, 2.0):
            print("   z=%-4g " % z + "  ".join("G=%-4g log|lam| %+.4f" % (G, np.log(radius(name, z, G))) for G in (1.0, 0.9, 0.75, 0.5, 0.25)))
    print("\na:E1+avg が減衰に変わる G の上限（G < この値で安定）")
    for z in (0.001, 0.1, 0.5, 1.0, 2.0):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if radius("a:E1+avg", z, m) <= 1 + 1e-12 else (lo, m)
        print("  z=%-6g G* = %.4f   1-G* = %.4f" % (z, lo, 1 - lo))

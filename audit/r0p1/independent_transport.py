"""監修（C）：vof.h の濃度の輸送（c = 1、中心の勾配、膨張の項は開始時の値で凍結）の、ライブラリを使わない再実装。
周期は np.roll（ゴーストセルを持たない）。ノート112 の 2・3 節の数値と突き合わせ、A が試していない
非対称な流れ（kx != ky）・非対称な場でも確かめる。"""
import math, sys
import numpy as np
import sympy as sp

GX, GW = np.polynomial.legendre.leggauss(8)
GX, GW = 0.5 * (GX + 1), 0.5 * GW

def cavg(fun, n):
    i = np.arange(n) / n
    out = np.zeros((n, n))
    for a, wa in zip(GX, GW):
        for b, wb in zip(GX, GW):
            X, Y = np.meshgrid(i + a / n, i + b / n, indexing="ij")
            out += wa * wb * fun(X, Y)
    return out

X_, Y_ = sp.symbols("x y")

class Case:
    def __init__(self, psi, q, name):
        self.name = name
        self.psi = sp.lambdify((X_, Y_), psi, "numpy")
        u, v = sp.diff(psi, Y_), -sp.diff(psi, X_)
        S = u * sp.diff(q, X_) + v * sp.diff(q, Y_)       # 定常解 q を持つ源の項
        A = lambda f: -u * sp.diff(f, X_)
        B = lambda f: -v * sp.diff(f, Y_)
        d = sp.diff(u, X_)
        lam = lambda e: sp.lambdify((X_, Y_), e, "numpy")
        self.q, self.S = lam(q), lam(S)
        self.odd = lam(sp.Rational(1, 2) * (B(A(q)) - A(B(q)) + d * (A(q) + B(q))))
        self.even = lam(sp.Rational(1, 2) * (u * sp.diff(S, X_) + v * sp.diff(S, Y_)))

    def faces(self, n):                                    # 流れ関数の頂点の差：面の平均として厳密
        xv = np.arange(n + 1) / n
        P = self.psi(*np.meshgrid(xv, xv, indexing="ij"))
        return ((P[:-1, 1:] - P[:-1, :-1]) * n, -(P[1:, :-1] - P[:-1, :-1]) * n)   # uf[i,j]: セル i の左の面、vf[i,j]: 下の面

def sweep(q, tc, uf, dt, n, ax, S=None):
    h = 1.0 / n
    un = uf * dt / h
    s = np.where(un > 0, 1.0, -1.0)
    g = (np.roll(q, -1, ax) - np.roll(q, 1, ax)) / 2
    qm, gm = np.roll(q, 1, ax), np.roll(g, 1, ax)          # 面の左のセル
    ff = np.where(s > 0, qm + (1 - un) * gm / 2, q - (1 + un) * g / 2)
    if S is not None:
        ff = ff + 0.25 * dt * (S + np.roll(S, 1, ax))
    F = ff * uf
    return q + dt / h * (F - np.roll(F, -1, ax) + tc * (np.roll(uf, -1, ax) - uf))

def step(q, case, n, dt, first, src_face, S):
    uf, vf = case.faces(n)
    tc = q.copy()
    for ax in ((0, 1) if first == 0 else (1, 0)):
        q = sweep(q, tc, uf if ax == 0 else vf, dt, n, ax, S if src_face else None)
    return q + dt * S

rms = lambda a: float(np.sqrt((a * a).mean()))

def split(case, n, cfl, src_face):
    dt = cfl / n
    q0, S = cavg(case.q, n), cavg(case.S, n)
    exy = step(q0, case, n, dt, 0, src_face, S) - q0
    eyx = step(q0, case, n, dt, 1, src_face, S) - q0
    ev, od = (exy + eyx) / 2, (exy - eyx) / 2
    po, pe = dt ** 2 * cavg(case.odd, n), dt ** 2 * cavg(case.even, n)
    return rms(od) / dt ** 2, rms(od - po) / rms(od), rms(ev) / dt ** 2, rms(ev - pe) / max(rms(pe), 1e-300)

def glob(case, n, cfl, T, src_face, alternate=True):
    steps = int(math.ceil(T / (cfl / n) - 1e-9)); dt = T / steps
    q0, S = cavg(case.q, n), cavg(case.S, n)
    q = q0.copy()
    for i in range(steps):
        q = step(q, case, n, dt, (i % 2) if alternate else 0, src_face, S)
    return float(np.abs(q - q0).max())

p = sp.pi
TG = Case(sp.sin(2 * p * X_) * sp.sin(2 * p * Y_) / (2 * p), sp.sin(2 * p * X_) * sp.cos(2 * p * Y_), "TG (note 112, x-momentum)")
ASY = Case(sp.sin(2 * p * X_) * sp.sin(4 * p * Y_) / (4 * p) + sp.cos(2 * p * (X_ - Y_)) / (8 * p),
           2 + sp.sin(2 * p * (X_ + 2 * Y_)) + sp.cos(4 * p * X_) * sp.sin(2 * p * Y_) / 2, "asymmetric (kx != ky)")

if __name__ == "__main__":
    for case in (TG, ASY):
        print("==", case.name)
        for src in (False, True):
            for n in (64, 128, 256):
                o = split(case, n, 0.4, src)
                print("split src_face=%-5s n=%3d cfl=0.4: odd/dt2 %.4f (odd-pred)/odd %.4f  even/dt2 %.4f  (even-pred)/pred %.4f" % ((src, n) + o))
        for src, alt in ((False, True), (True, True), (True, False)):
            for n in (64, 128, 256):
                print("global src_face=%-5s alt=%-5s n=%3d T=0.25:" % (src, alt, n),
                      " ".join("%.3e" % glob(case, n, c, 0.25, src, alt) for c in (0.4, 0.2, 0.1, 0.05)))

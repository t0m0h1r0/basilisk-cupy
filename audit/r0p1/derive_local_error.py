"""監修（C）：ノート112 B 節の 1 ステップの局所誤差の主要項を、A のコードを使わずに記号計算で導く。
モデル：空間は厳密、面の値は特性線を半ステップ戻した値 q - (dt/2) u q_x (+ sigma (dt/2) S)、
膨張の項は開始時の値 q0 で凍結（vof.h）。輸送の後に dt S を足す。"""
import sympy as sp

x, y, dt, sig = sp.symbols("x y dt sigma")
u = sp.Function("u")(x, y)
psi = sp.Function("psi")(x, y)
u, v = sp.diff(psi, y), -sp.diff(psi, x)          # 発散 0
q0 = sp.Function("q")(x, y)
S = sp.Function("S")(x, y)

def X(q):
    return q - dt * sp.diff(u * (q - dt / 2 * u * sp.diff(q, x) + sig * dt / 2 * S), x) + dt * q0 * sp.diff(u, x)

def Y(q):
    return q - dt * sp.diff(v * (q - dt / 2 * v * sp.diff(q, y) + sig * dt / 2 * S), y) + dt * q0 * sp.diff(v, y)

L = lambda f: -u * sp.diff(f, x) - v * sp.diff(f, y)
exact = q0 + dt * (L(q0) + S) + dt ** 2 / 2 * L(L(q0) + S)
c2 = lambda e: sp.expand(sp.series(sp.expand(e), dt, 0, 3).removeO()).coeff(dt, 2)
c1 = lambda e: sp.expand(sp.series(sp.expand(e), dt, 0, 3).removeO()).coeff(dt, 1)
exy = Y(X(q0)) + dt * S - exact
eyx = X(Y(q0)) + dt * S - exact
assert sp.simplify(c1(exy)) == 0 and sp.simplify(c1(eyx)) == 0
A = lambda f: -u * sp.diff(f, x)
B = lambda f: -v * sp.diff(f, y)
d = sp.diff(u, x)
odd = sp.simplify(c2((exy - eyx) / 2))
even = sp.simplify(c2((exy + eyx) / 2))
user = sp.Rational(1, 2) * (B(A(q0)) - A(B(q0)) + d * (A(q0) + B(q0)))
print("odd - user formula      :", sp.simplify(odd - user))
src = sp.Rational(1, 2) * (u * sp.diff(S, x) + v * sp.diff(S, y))
print("even - (1-sigma) u.grad S/2:", sp.simplify(even - (1 - sig) * src))

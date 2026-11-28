"""design/198 の確認：二層 Poiseuille（x は周期、y は粘着壁、体積力 F が x 向き）の、x に一様な状態のまわりの
1 ステップの線形の模型。x の Fourier モード exp(ikx) ごとに、y は N 行をそのまま持つ小さい行列の固有値を見る。
ソルバーは動かさない（centered.h の 1 ステップを、擾乱について書き下した模型）。

入れたもの（centered.h + two-phase.h の順）
  1. VOF：界面の面（重い相の最上の行 j0-1 と、その上の行 j0 の間）の面の速度 v_I で、 df_{j0} += dt v_I / h、
     df_{j0-1} -= dt v_I / h（線形化：界面の変位。上流の切り替えは入れない）
  2. 物性：セルの密度 rho(f)、x の面は両隣のセルの f の平均、y の面は上下のセルの f の平均
  3. 粘性：後退 Euler。 rho0 (du** - du - dt dg)/dt - drho g0 = L du**   （g0 = F / rho0。基底は準定常と置く）
     その後 du <- du** - dt dg
  4. 力：x の面の加速度 a = F / rho_face。 da = -F drho_face / rho0^2
  5. 面の速度：uf = I_f(du) + dt da、 射影（係数は基底の alpha）、 g = C(a - alpha grad p)、 du += dt g
入れていないもの：基底の流れによる擾乱の移流、粘性の係数の擾乱、上流の切り替え（f を 0・1 に切る非線形）。

出力：1 ステップの写像のスペクトル半径（1 を超えれば、その模型では増幅する）と、切り分けのスイッチ
  frozen_force : 力の加速度を、擾乱した密度ではなく基底の密度で作る（da = 0、粘性の段の drho g0 も 0）
  sharp_face   : 界面の面の速度の補間を、質量加重 (rho_H v_H + rho_L v_L)/(rho_H + rho_L) にする（診断用）
  cell_force   : セルの力の加速度を、面の平均 C(F/rho_face) ではなくセルの密度 F/rho_cell で作る（診断用）
  no_mass_term : 粘性の段の drho g0（新しい密度 × 古い加速度の不整合）を落とす（模型の中だけ）

実行：python design/checks/198_linear_mode.py        （numpy。空いた計算機で約 2 分）
"""

import numpy as np


def step_matrix(n, m, dtn, ratio=1000.0, mu=(1.0, 1.0 / 55.0), F=1.0, f_int=0.0, frozen_force=False,
                sharp_face=False, viscous=True, cell_force=False, no_mass_term=False):
    """The one-step map of the perturbation for the x-mode m on an n x n grid (dt = dtn / n).
    f_int: the base volume fraction of the row just above the heavy rows (0: the interface on a face)."""
    h = 1.0 / n
    dt = dtn / n
    k = 2.0 * np.pi * m
    cx = np.cos(k * h / 2.0)                       # cell <-> x-face average
    dx = 2j * np.sin(k * h / 2.0) / h              # cell <-> x-face difference
    rho1, rho2 = 1.0, 1.0 / ratio
    drho = rho1 - rho2
    j0 = n // 2                                    # rows 0 .. j0-1 heavy, row j0 has f_int, above light
    f0 = np.where(np.arange(n) < j0, 1.0, 0.0)
    f0[j0] = f_int
    rho0 = rho2 + drho * f0
    mu0 = mu[1] + (mu[0] - mu[1]) * f0
    fy = np.empty(n + 1)                           # y-faces 0 .. n (0 and n are the walls)
    fy[1:n] = 0.5 * (f0[:-1] + f0[1:])
    fy[0], fy[n] = f0[0], f0[-1]
    ay = 1.0 / (rho2 + drho * fy)                  # alpha on the y-faces
    ax = 1.0 / rho0                                # alpha on the x-faces of each row
    muy = mu[1] + (mu[0] - mu[1]) * fy
    g0 = F / rho0
    jf = j0 if f_int == 0.0 else j0                # the face whose velocity moves the interface: below row j0

    # the pressure operator div(alpha grad p) for the mode
    A = np.zeros((n, n), dtype=complex)
    for j in range(n):
        A[j, j] += ax[j] * (dx * dx).real
        if j > 0:
            A[j, j] -= ay[j] / h**2
            A[j, j - 1] += ay[j] / h**2
        if j < n - 1:
            A[j, j] -= ay[j + 1] / h**2
            A[j, j + 1] += ay[j + 1] / h**2
    # backward Euler: rho0 (u' - u)/dt = mu0 (dxx) u' + d/dy(muy du'/dy), no-slip walls (ghost = -u)
    V = np.zeros((n, n), dtype=complex)
    for j in range(n):
        V[j, j] += rho0[j] / dt - mu0[j] * (dx * dx).real
        lo = muy[j] / h**2
        hi = muy[j + 1] / h**2
        V[j, j] += lo + hi
        if j > 0:
            V[j, j - 1] -= lo
        else:
            V[j, j] += lo                          # wall: u_ghost = -u
        if j < n - 1:
            V[j, j + 1] -= hi
        else:
            V[j, j] += hi

    Vinv, Ainv = np.linalg.inv(V), np.linalg.inv(A)

    # state: [df (n), u (n), v (n), gx (n), gy (n), vI (1)]
    size = 5 * n + 1

    def step(s):
        df, u, v, gx, gy, vI = s[:n].copy(), s[n:2 * n].copy(), s[2 * n:3 * n].copy(), s[3 * n:4 * n], s[4 * n:5 * n], s[5 * n]
        # 1 VOF
        df[j0] += dt * vI / h
        df[j0 - 1] -= dt * vI / h
        # 2 properties
        drc = drho * df                             # cells
        drx = drho * cx * df                        # x-faces (the mode's amplitude on the faces)
        # 3 viscosity
        if viscous:
            rhs_u = rho0 * (u + dt * gx) / dt + (0.0 if (frozen_force or no_mass_term) else drc * g0)
            rhs_v = rho0 * (v + dt * gy) / dt
            u = Vinv @ rhs_u - dt * gx
            v = Vinv @ rhs_v - dt * gy
        # 4 force
        da = np.zeros(n, dtype=complex) if frozen_force else -F * drx / rho0**2
        # 5 face velocities, projection, cells
        ufx = cx * u + dt * da
        ufy = np.zeros(n + 1, dtype=complex)
        ufy[1:n] = 0.5 * (v[:-1] + v[1:])
        if sharp_face:
            ufy[j0] = (rho0[j0 - 1] * v[j0 - 1] + rho0[j0] * v[j0]) / (rho0[j0 - 1] + rho0[j0])
        div = dx * ufx + (ufy[1:] - ufy[:-1]) / h
        p = Ainv @ (div / dt)
        gxf = da - ax * dx * p                      # x-faces
        gyf = np.zeros(n + 1, dtype=complex)
        gyf[1:n] = -ay[1:n] * (p[1:] - p[:-1]) / h
        ufy = ufy + dt * gyf
        gx1 = cx * gxf
        if cell_force and not frozen_force:         # the cells' force acceleration from the cells' own density
            gx1 = gx1 - cx * da + (-F * drc / rho0**2)
        gy1 = 0.5 * (gyf[:-1] + gyf[1:])
        u = u + dt * gx1
        v = v + dt * gy1
        return np.concatenate([df, u, v, gx1, gy1, [ufy[jf]]])

    M = np.zeros((size, size), dtype=complex)
    for i in range(size):
        e = np.zeros(size, dtype=complex)
        e[i] = 1.0
        M[:, i] = step(e)
    return M


def radius(*a, **k):
    return float(max(abs(np.linalg.eigvals(step_matrix(*a, **k)))))


def worst(n, dtn, **k):
    """The largest spectral radius over a set of x-modes (m = 1 .. n/2 sampled), and its mode."""
    r = [(radius(n, m, dtn, **k), m) for m in sorted({1, 2, 3, 4, 6, 8, n // 4 + n // 8, n // 2})]
    return max(r)


if __name__ == "__main__":
    n = 32
    print("1 ステップの写像のスペクトル半径の最大（x のモード m = 1〜N/2、N = %d）。括弧はそのモード" % n)
    print("%-44s" % "配置" + "".join("dt N = %-8g" % d for d in (0.5, 0.29, 0.1, 0.02)))
    rows = [("面の上、密度比 1000、粘度比 55", dict()),
            ("同、粘度比 1", dict(mu=(1.0, 1.0))),
            ("同、密度比 100", dict(ratio=100.0)),
            ("同、密度比 10", dict(ratio=10.0)),
            ("同、力なし（Couette に当たる）", dict(F=0.0)),
            ("界面を 0.3 セルずらす", dict(f_int=0.3)),
            ("面の上、力を基底の密度で作る（診断）", dict(frozen_force=True)),
            ("面の上、界面の面の補間を質量加重に（診断）", dict(sharp_face=True)),
            ("面の上、粘性なし（模型の中だけ）", dict(viscous=False)),
            ("面の上、セルの力をセルの密度で（診断）", dict(cell_force=True)),
            ("面の上、粘性の段の密度の項なし（模型の中だけ）", dict(no_mass_term=True)),
            ("面の上、質量加重の補間 + セルの力（診断）", dict(sharp_face=True, cell_force=True))]
    table = {}
    for name, kw in rows:
        vals = [worst(n, d, **kw) for d in (0.5, 0.29, 0.1, 0.02)]
        table[name] = vals
        print("%-44s" % name + "".join("%-13s" % ("%.3f (%d)" % v) for v in vals))
    print("\nN への依存（面の上、密度比 1000、dt N = 0.1）：" + "  ".join("N = %d: %.3f (%d)" % ((nn,) + worst(nn, 0.1)) for nn in (16, 32, 64)))
    print("ok")

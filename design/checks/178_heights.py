"""design/178 の確認（静的。ソルバーは動かさない）：厳密な体積分率の配列に対して、reduced のポテンシャル
phi = G.(x_I - Z) の界面の位置（ライブラリの ``heights.position``）を、候補の形で評価する。

候補（どれも f と輸送に触れない。高さ関数に渡す「写し」と、高さの後処理だけ）
  B-1  判定用の写し：c <= eps を 0、c >= 1 - eps を 1 に丸めた写しから、高さ・界面のセル・ポテンシャルを作る
  A-1  壁で閉じた列：壁のゴーストセルを、鏡像ではなく「壁に接するセルの、壁の側の相」（0 か 1）で埋めて高さを作る。
       界面が壁と交わるセル（接触のセル：壁に沿った両隣で壁の側の相が違う部分的なセル、箱の角の部分的なセル）の
       壁に垂直な向きの高さは捨て、壁に沿った隣の列の高さの線形外挿（h = 2 h1 - h2、埋めた値を次に使う）で埋める
  A-2  高さの線の規則：標準の規則（|h| <= 1 の高さ）で位置が決まらない界面のセルで、PLIC の重心に落ちる前に、
       |h| <= 2.5 の高さとその傾きで作る直線を、セルの中へ辿った点を使う
  （第 1 版の A-1・A-2〔定数延長 + 線の規則、隣の 2 列の外挿〕は、監修の走査で 56 配置が残った：設計の 3 節）

確かめること（合成の配列だけ。走査はしない：1850 配置の走査は、設計の 5.2 節の E-A1 として A の実験）
  1. 周期の斜めの帯（壁なし）に、界面の隣のセルへ薄片（1e-22、1 - 1e-16）を入れたときの、面の力 |phi_f (f_r - f_l)|
     （薄片のセルが phi を持つ経路の確認と、B-1 でそれが消えること）
  2. 閉じた箱の平面の 15 配置（ノート120 と監修の不成立の配置、監修の反例の型を含む）で、t = 0 の max|phi|：候補が平面で厳密か

実行：PYTHONPATH=. python design/checks/178_heights.py        （numpy。数十秒）
"""

import importlib
import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from basilisk_cupy import BoundaryConditions, CartesianGrid  # noqa: E402
from basilisk_cupy.backend import to_numpy  # noqa: E402
from basilisk_cupy.interface.fractions import fraction  # noqa: E402

importlib.import_module("basilisk_cupy.interface.heights")
H = sys.modules["basilisk_cupy.interface.heights"]
EPS = 1e-10


def snap(c, eps=EPS):
    return np.where(c <= eps, 0.0, np.where(c >= 1.0 - eps, 1.0, c))


def _pad_constant(arr, bc, width):
    """pad_bc with the walls' ghosts replaced by the constant extension of the wall cell (periodic sides unchanged)."""
    out = arr
    for ax in range(arr.ndim):
        if bc.get(ax, 0).periodic:
            out = np.concatenate([out.take(range(-width, 0), ax), out, out.take(range(0, width), ax)], axis=ax)
        else:
            pad = [(0, 0)] * arr.ndim
            pad[ax] = (width, width)
            out = np.pad(out, pad, mode="edge")
    return out


def _wall_phase(c0, c1):
    """The phase on the wall side of the interface in a wall cell (c0: the wall cell, c1: the next cell inwards):
    1 if phase 1 fills the wall side, 0 otherwise. Pure cells keep their value."""
    return np.where(c0 >= 1.0, 1.0, np.where(c0 <= 0.0, 0.0,
                    np.where(c1 < c0, 1.0, np.where(c1 > c0, 0.0, np.round(c0)))))


def _pad_closed(arr, bc, width):
    """pad_bc with the walls' ghosts filled with the wall-side phase (0 or 1) of each wall cell: a column that
    crosses the wall is closed by the wall instead of reading mirrored fluid."""
    out = arr
    for ax in range(arr.ndim):
        if bc.get(ax, 0).periodic:
            out = np.concatenate([out.take(range(-width, 0), ax), out, out.take(range(0, width), ax)], axis=ax)
            continue
        lo = _wall_phase(out.take([0], ax), out.take([1], ax))
        hi = _wall_phase(out.take([-1], ax), out.take([-2], ax))
        out = np.concatenate([np.repeat(lo, width, ax), out, np.repeat(hi, width, ax)], axis=ax)
    return out


def _contact_cells(c, bc):
    """Wall cells where the interface meets the wall inside the cell: partial cells in a wall row whose two
    neighbours along the wall have different wall-side phases. Returned per axis (the axis normal to the wall):
    the closed column of such a cell is truncated by the wall and its height is not the interface's."""
    out = []
    for ax in range(2):
        m = np.zeros(c.shape, dtype=bool)
        if not bc.get(ax, 0).periodic:
            t = 1 - ax
            for side, (i0, i1) in enumerate(((0, 1), (-1, -2))):
                c0, c1 = c.take(i0, ax), c.take(i1, ax)
                ph = _wall_phase(c0, c1)
                left = np.concatenate([ph[:1], ph[:-1]])
                right = np.concatenate([ph[1:], ph[-1:]])
                if bc.get(t, 0).periodic:
                    left, right = np.roll(ph, 1), np.roll(ph, -1)
                contact = (c0 > 0.0) & (c0 < 1.0) & (left != right)
                if not bc.get(t, 0).periodic:               # a partial cell in a corner of the box: both walls
                    contact[0] = (c0[0] > 0.0) & (c0[0] < 1.0)
                    contact[-1] = (c0[-1] > 0.0) & (c0[-1] < 1.0)
                idx = [slice(None)] * 2
                idx[ax] = i0
                m[tuple(idx)] = contact
        out.append(m)
    return out


def _extrapolate_all(hs, c, bc, invalid=None, passes=64):
    """A-2 iterated: missing (or invalidated) heights of interfacial cells from the two nearest filled columns of
    the same axis on one side (h = 2 h1 - h2), repeated so that filled values propagate along the wall."""
    out = []
    real = (c > 0.0) & (c < 1.0)
    for ax in range(2):
        h = H.height(np.asarray(hs[ax]), np)
        h = np.where(np.abs(h) > 5.6, np.nan, h)
        if invalid is not None:
            h = np.where(invalid[ax], np.nan, h)
        t = 1 - ax
        per = bc.get(t, 0).periodic
        for _ in range(passes):
            fill = np.full(h.shape, np.nan)
            for s in (1, -1):
                h1, h2 = np.roll(h, -s, t), np.roll(h, -2 * s, t)
                if not per:
                    idx = [slice(None)] * 2
                    idx[t] = slice(-2, None) if s == 1 else slice(0, 2)
                    h1 = h1.copy(); h2 = h2.copy()
                    h2[tuple(idx)] = np.nan
                    idx[t] = slice(-1, None) if s == 1 else slice(0, 1)
                    h1[tuple(idx)] = np.nan
                cand = 2.0 * h1 - h2
                fill = np.where(np.isnan(fill) & np.isfinite(cand), cand, fill)
            todo = real & np.isnan(h) & np.isfinite(fill)
            if not todo.any():
                break
            h = np.where(todo, fill, h)
        out.append(np.where(np.isnan(h), H.NODATA, h))
    return out


def _line_points(hs, c, bc):
    """A-1: for each cell and axis, the point (in cell units, relative to the centre) where the line through the
    column's height with the heights' slope crosses the cell; NaN where there is none."""
    pts = []
    for ax in range(2):
        hall = H.height(np.asarray(hs[ax]), np)
        hall = np.where(np.abs(hall) > 5.6, np.nan, hall)       # every complete column (for the slope)
        h = np.where(np.abs(hall) > 2.5, np.nan, hall)          # the cell's own height: at most 2.5 cells away
        t = 1 - ax
        per = bc.get(t, 0).periodic
        hp, hm = np.roll(hall, -1, t), np.roll(hall, 1, t)
        if not per:
            idx = [slice(None)] * 2
            idx[t] = slice(-1, None)
            hp = hp.copy(); hp[tuple(idx)] = np.nan
            idx[t] = slice(0, 1)
            hm = hm.copy(); hm[tuple(idx)] = np.nan
        slope = np.where(np.isfinite(hp) & np.isfinite(hm), (hp - hm) / 2.0,
                         np.where(np.isfinite(hp), hp - h, np.where(np.isfinite(hm), h - hm, np.nan)))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(slope != 0.0, (-0.5 - h) / slope, np.where(np.abs(h) <= 0.5, -0.5, np.nan))
            t2 = np.where(slope != 0.0, (0.5 - h) / slope, np.where(np.abs(h) <= 0.5, 0.5, np.nan))
        lo = np.maximum(np.minimum(t1, t2), -0.5)
        hi = np.minimum(np.maximum(t1, t2), 0.5)
        tau = np.where(lo <= hi, (lo + hi) / 2.0, np.nan)
        xi = h + slope * tau
        pts.append((xi, tau))
    return pts


def position(c, bc, grid, G, Z, variant="library", eps=None):
    """variant: "library" (mirror ghosts); "ghost" (constant ghosts, nothing else: a control);
    "A-1" (wall-closed columns + contact cells extrapolated); "A-1+A-2" (and the line rule)."""
    ce = snap(c, eps) if eps else c
    saved = H.pad_bc
    try:
        if variant == "ghost":
            H.pad_bc = _pad_constant
        elif variant != "library":
            H.pad_bc = _pad_closed
        hs = H.heights(ce, bc)
        if variant.startswith("A-1"):
            hs = _extrapolate_all(hs, ce, bc, invalid=_contact_cells(ce, bc))
        p = np.asarray(to_numpy(H.position(ce, bc, grid, G, Z, hs=hs)))
        if "A-2" not in variant:
            return p
        # cells where the standard rule found no height (the library falls back to the PLIC centroid there)
        hh = [H.height(np.asarray(h), np) for h in hs]
        nohf = (np.abs(hh[0]) > 1.0) & (np.abs(hh[1]) > 1.0) & (p < H.NODATA / 2)
        pts = _line_points(hs, ce, bc)
        x, y = (np.asarray(to_numpy(a)) for a in grid.meshgrid())
        d = float(grid.h[0])
        cp1 = H.pad_bc(ce, bc, 1)
        nx_, ny_ = np.abs(cp1[2:, 1:-1] - cp1[:-2, 1:-1]), np.abs(cp1[1:-1, 2:] - cp1[1:-1, :-2])
        order = np.where(nx_ >= ny_, 0, 1)                   # the axis of the larger normal component first
        out = p.copy()
        done = np.zeros(p.shape, dtype=bool)
        for rank in (0, 1):
            for ax in range(2):
                use = nohf & ~done & (np.where(rank == 0, order, 1 - order) == ax) & np.isfinite(pts[ax][0])
                xi, tau = pts[ax]
                px = x + d * (xi if ax == 0 else tau)
                py = y + d * (tau if ax == 0 else xi)
                val = (px - Z[0]) * G[0] + (py - Z[1]) * G[1]
                out = np.where(use, val, out)
                done |= use
        return out
    finally:
        H.pad_bc = saved


def face_force(p, c, periodic):
    """max |phi_f (f_r - f_l)| over the faces (phi_f as iforce.h: the mean where both sides have phi)."""
    ok = p < H.NODATA / 2
    v = np.where(ok, p, 0.0)
    m = 0.0
    for ax in range(2):
        n = c.shape[ax]
        sl = (lambda a: a) if periodic else (lambda a: a.take(range(0, n - 1), ax))
        fl, fr = sl(c), sl(np.roll(c, -1, ax))
        pl, pr = sl(v), sl(np.roll(v, -1, ax))
        ol, orr = sl(ok), sl(np.roll(ok, -1, ax))
        pf = np.where(ol & orr, (pl + pr) / 2, np.where(orr, pr, np.where(ol, pl, 0.0)))
        m = max(m, float(np.abs(pf * (fr - fl)).max()))
    return m


def band(n, a, b, off):
    r = math.hypot(a, b)
    grid = CartesianGrid((n, n))

    def phi(x, y):
        z = (a * x + b * y - off) % 1.0
        return np.where(z < 0.5, np.minimum(z, 0.5 - z), -np.minimum(z - 0.5, 1 - z)) / r

    return grid, np.asarray(to_numpy(fraction(grid, phi, refine=8))), r


def neighbours(mask, periodic=True):
    nb = np.zeros_like(mask)
    for ax in range(2):
        for s in (-1, 1):
            m = np.roll(mask, s, ax)
            if not periodic:
                idx = [slice(None)] * 2
                idx[ax] = slice(0, 1) if s == 1 else slice(-1, None)
                m = m.copy()
                m[tuple(idx)] = False
            nb |= m
    return nb


def with_slivers(c, rng, periodic=True):
    real = (c > 0) & (c < 1)
    nb = neighbours(real, periodic)
    c2 = c.copy()
    c2[nb & (c == 0.0) & (rng.random(c.shape) < 0.5)] = 1e-22
    c2[nb & (c == 1.0) & (rng.random(c.shape) < 0.5)] = 1.0 - 1e-16
    return c2


def plane(n, th, off):
    t = math.radians(th)
    nx, ny = math.sin(t), math.cos(t)
    grid = CartesianGrid((n, n))
    c = np.asarray(to_numpy(fraction(grid, lambda x, y: -(nx * (x - 0.5) + ny * (y - 0.5 - off)))))
    return grid, c, (-nx, -ny), (0.5, 0.5 + off)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print("1. 周期の斜めの帯 + 界面の隣の薄片（壁なし）：面の力 max|phi_f df|（セルの単位。G は単位法線）")
    bcp = BoundaryConditions.periodic(2)
    worst = {"library": 0.0, "B-1": 0.0}
    for a, b in ((1, 1), (1, 2), (2, 3)):
        for n in (32, 64):
            grid, c, r = band(n, a, b, 0.0137)
            G, Z = (a / r, b / r), (0.0, 0.0)

            def fold(p):                                    # the two interfaces: distance to the nearest line, cells
                ok = p < H.NODATA / 2
                return np.where(ok, (((p * r - 0.0137 + 0.25) % 0.5) - 0.25) / r * n, H.NODATA)

            c2 = with_slivers(c, rng)
            f0 = face_force(fold(position(c, bcp, grid, G, Z)), c, True)
            f1 = face_force(fold(position(c2, bcp, grid, G, Z)), c2, True)
            f2 = face_force(fold(position(c2, bcp, grid, G, Z, eps=EPS)), c2, True)
            worst["library"], worst["B-1"] = max(worst["library"], f1), max(worst["B-1"], f2)
            print("   n = (%d, %d) N = %-3d 薄片なし %.1e   薄片あり：ライブラリ %.1e   B-1 %.1e" % (a, b, n, f0, f1, f2))
    assert worst["library"] > 1e-2 and worst["B-1"] < 1e-12

    print("2. 閉じた箱の平面（theta, delta, N）：t = 0 の max|phi|（最後の列は、界面の隣に薄片を入れて B-1 も掛けた場合）")
    bcw = BoundaryConditions()
    SAMPLES = ((45.0, 0.0137, 32), (45.0, 0.0137, 64), (26.565, 0.2313, 32),            # note 120
               (37.5, 0.45, 64), (30.0, 0.40, 64), (60.0, 0.40, 64),                    # the audit on PR #165
               (5.0, -0.45, 32), (12.5, -0.45, 32), (7.5, 0.47, 64), (22.5, 0.45, 64),  # the audit's counter-example type
               (37.5, 0.0, 64), (63.435, -0.3, 32), (10.0, 0.3, 32), (85.0, 0.2, 32), (0.0, 0.0137, 32))
    for th, off, n in SAMPLES:
        grid, c, G, Z = plane(n, th, off)
        row = []
        for var in ("library", "ghost", "A-1", "A-1+A-2"):
            p = position(c, bcw, grid, G, Z, variant=var)
            row.append(float(np.abs(np.where(p < H.NODATA / 2, p, 0.0)).max()))
        c2 = with_slivers(c, rng, periodic=False)
        p = position(c2, bcw, grid, G, Z, variant="A-1+A-2", eps=EPS)
        last = float(np.abs(np.where(p < H.NODATA / 2, p, 0.0)).max())
        print("   (%6.3f, %+.4f, %2d)  library %.1e  ghost %.1e  A-1 %.1e  A-1+A-2 %.1e   薄片 + B-1 %.1e" % (th, off, n, *row, last))
        assert row[3] < 1e-12 and last < 1e-12, (th, off, n)
        assert abs(row[0] - row[1]) <= 1e-12 + 1e-6 * row[0]     # constant ghosts alone change nothing
    print("ok")

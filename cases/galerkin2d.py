"""docs/notes/87 (P1-1): the geometry-consistent Galerkin transport in two dimensions, exact geometry.

Periodic [0, 1)^2, N x N cells. A heavy band {frac(a x + b y - d) in (zA, zB)} (integers a, b: the band is periodic,
no spurious interface at the domain boundary), translated by a constant velocity U (d = (a Ux + b Uy) t).

Reconstruction of the mass-weighted cell averages (the note-82/85 design in 2-D):
* near an interface (normal distance <= D1 cells from the cell centre): the piecewise polynomial
  P = P0(xi, eta) + max(xi, 0) P1(xi, eta) (P0 of total degree 4, P1 of total degree 3: 25 coefficients; xi, eta the
  normal and tangential coordinates of the nearest interface line in cells, the kink on the far side of the cell's
  centre): continuous across the interface, its normal gradient may jump;
* far from it (>= D2): the single polynomial P0 (15 coefficients);
* between: the two mixed by a smooth step of the distance (both keep the cell's average and the constants).
Each fit: weighted least squares over a window (7 x 7 cells for the piecewise, 5 x 5 for the single), weights
(m_j / max m)^alpha, then a constant shift that restores the cell's own mass-weighted average ("shift", note 85).

Transports, all from one common partition (new cell i intersected with the moved old cell j and the phases, Gauss on
the triangles of the convex pieces, exact for degree 9): G0 = B0^T W B0, G1 = B1^T W B1, C = B1^T W B0 (so the block
matrix [[G0, C^T], [C, G1]] is positive semi-definite and C^T G1^-1 C <= G0: the energy cannot grow), and the direct
remap Cg_i = sum_{q in K_i} w B0. T_G = G1^-1 C, T_geo = M1^-1 Cg, T_corr = T_G [I + (I - G0^-1 C^T T_G)/2].

    python -m cases.galerkin2d
"""

from __future__ import annotations

import math

import numpy as np

_xg, _wg = np.polynomial.legendre.leggauss(6)
XG, WG = 0.5 * (_xg + 1.0), 0.5 * _wg
P0_EXP = [(p, q) for d in range(5) for p in range(d + 1) for q in [d - p]]          # 15
P1_EXP = [(p, q) for d in range(4) for p in range(d + 1) for q in [d - p]]          # 10


class Geometry:
    def __init__(self, a=1, b=1, zA=0.213, zB=0.647, ratio=1e3, d=0.0):
        self.a, self.b, self.zA, self.zB, self.ratio, self.d = a, b, zA, zB, ratio, d
        self.g = math.hypot(a, b)
        self.n = np.array([a, b]) / self.g

    def moved(self, U, t):
        return Geometry(self.a, self.b, self.zA, self.zB, self.ratio, self.d + (self.a * U[0] + self.b * U[1]) * t)

    def L(self, x, y):
        return self.a * x + self.b * y

    def heavy(self, x, y):
        z = np.mod(self.L(x, y) - self.d, 1.0)
        return (z > self.zA) & (z < self.zB)

    def rho(self, x, y):
        return np.where(self.heavy(x, y), 1.0, 1.0 / self.ratio)

    def breaks(self, lo, hi):
        out = []
        for z in (self.zA, self.zB):
            k0 = math.floor(lo - self.d - z)
            for k in range(k0, k0 + int(hi - lo) + 3):
                c = z + self.d + k
                if lo < c < hi:
                    out.append(c)
        return sorted(out)

    def nearest(self, x, y):
        """Signed normal distance (physical) from (x, y) to the nearest interface line, and that line's value."""
        Lv = self.L(x, y)
        best = None
        for z in (self.zA, self.zB):
            k = round(Lv - self.d - z)
            for kk in (k - 1, k, k + 1):
                c = z + self.d + kk
                s = (Lv - c) / self.g
                if best is None or abs(s) < abs(best[0]):
                    best = (s, c)
        return best


def _clip(poly, a, b, c, keep_ge):
    """Sutherland-Hodgman: keep a x + b y >= c (keep_ge) or <= c."""
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        inp = fp >= 0 if keep_ge else fp <= 0
        inq = fq >= 0 if keep_ge else fq <= 0
        if inp:
            out.append(p)
        if inp != inq:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _quad(poly):
    """Points and weights on a convex polygon (fan triangles, collapsed Gauss, exact for degree 9)."""
    pts, wts = [], []
    p0 = np.array(poly[0])
    for k in range(1, len(poly) - 1):
        p1, p2 = np.array(poly[k]), np.array(poly[k + 1])
        det = abs((p1 - p0)[0] * (p2 - p1)[1] - (p1 - p0)[1] * (p2 - p1)[0])
        if det < 1e-300:
            continue
        u, v = np.meshgrid(XG, XG, indexing="ij")
        wu, wv = np.meshgrid(WG, WG, indexing="ij")
        x = p0[None, None, :] + u[..., None] * (p1 - p0) + (u * v)[..., None] * (p2 - p1)
        pts.append(x.reshape(-1, 2))
        wts.append((wu * wv * u * det).reshape(-1))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(pts), np.concatenate(wts)


def pieces(geom, x0, y0, x1, y1):
    """The rectangle cut by the interface lines: [(points, weights, heavy)]."""
    if hasattr(geom, "pieces"):
        return geom.pieces(x0, y0, x1, y1)
    rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    Ls = [geom.L(x, y) for x, y in rect]
    cuts = [min(Ls)] + geom.breaks(min(Ls), max(Ls)) + [max(Ls)]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 1e-15:
            continue
        poly = _clip(_clip(rect, geom.a, geom.b, lo, True), geom.a, geom.b, hi, False)
        if len(poly) < 3:
            continue
        x, w = _quad(poly)
        if w.size == 0 or w.sum() <= 0:
            continue
        m = 0.5 * (lo + hi)
        hv = bool(np.mod(m - geom.d, 1.0) > geom.zA and np.mod(m - geom.d, 1.0) < geom.zB)
        out.append((x, w, hv))
    return out


_SHARED = {}


def _call(args):
    key, method, item = args
    return getattr(_SHARED[key], method)(item)


def _parallel(obj, method, items, workers):
    """[obj.method(i) for i in items], over ``workers`` forked processes (obj is inherited, not pickled)."""
    items = list(items)
    if workers <= 1:
        return [getattr(obj, method)(i) for i in items]
    import multiprocessing as mp

    key = id(obj)
    _SHARED[key] = obj
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_call, [(key, method, i) for i in items], chunksize=max(1, len(items) // (4 * workers)))
    finally:
        _SHARED.pop(key, None)


def _smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Reconstruction2D:
    def __init__(self, N, geom, alpha=0.75, D1=3.5, D2=4.5, m_pw=3, m_s=2, kink=True, window="cell", R=4.5,
                 switch="fixed", delta=1.0, tau=None, diff=False, workers=1, hinge="exact", centre_single=True):
        """docs/notes/88 options: ``window`` "cell" (the (2 m_pw + 1)^2 cells around the cell, note 87) or "interface"
        (the cells within R cells of the cell's foot on the interface, weights (1 - (r/R)^2)^2: both sides of the
        interface always hold data); ``switch`` "fixed" (D1, D2) or "angle" (D1 = d_safe = (m_s + 1/2)(|nx| + |ny|)
        cells, the reach of the single window, D2 = D1 + delta); ``tau``: the continuous SVD filter of the
        interface component's pseudo-inverse (relative to its largest singular value), None = plain lstsq;
        ``diff``: the shift written on velocity differences (u_i + sum (m_k - mbar_k) a_k, a = W (u - u_i 1))."""
        self.N, self.geom, self.alpha = N, geom, alpha
        self.h = h = 1.0 / N
        self.kink = kink
        self.window, self.R, self.tau, self.diff = window, R, tau, diff
        self.hinge = hinge  # curved interfaces (docs/notes/89): "exact" signed distance or "tangent" line
        self.centre_single = centre_single
        self._switch, self._delta = switch, delta
        if switch == "angle" and not hasattr(geom, "frame"):
            D1 = (m_s + 0.5) * (abs(geom.n[0]) + abs(geom.n[1]))
            D2 = D1 + delta
        self.rank_log = []  # (distance in cells, numerical rank of the interface component, its size)
        self._D, self._m = (D1, D2), (m_pw, m_s)
        self.sigma_min = np.inf
        # every cell's quadrature (canonical position), rho-weights, mass; then every cell's fits: both per row of
        # cells, over ``workers`` processes (docs/notes/89)
        quad = _parallel(self, "_quad_row", range(N), workers)
        self.qx = [x for row in quad for x, _ in row]
        self.qw = [w for row in quad for _, w in row]
        self.mass = np.array([w.sum() for w in self.qw])
        out = _parallel(self, "_fit_row", range(N), workers)
        self.comps = [c for row, _, _ in out for c in row]
        self.sigma_min = min([self.sigma_min] + [sm for _, sm, _ in out])
        self.rank_log = [r for _, _, rl in out for r in rl]

    def _quad_row(self, iy):
        N, h, geom = self.N, self.h, self.geom
        row = []
        for ix in range(N):
            xs, ws = [], []
            for x, w, hv in pieces(geom, ix * h, iy * h, (ix + 1) * h, (iy + 1) * h):
                xs.append(x)
                ws.append(w * (1.0 if hv else 1.0 / geom.ratio))
            row.append((np.vstack(xs), np.concatenate(ws)))
        return row

    def _fit_row(self, iy):
        N, h, geom = self.N, self.h, self.geom
        (D1, D2), (m_pw, m_s) = self._D, self._m
        self.sigma_min, self.rank_log = np.inf, []
        row = []
        for ix in range(N):
            c = np.array([(ix + 0.5) * h, (iy + 0.5) * h])
            s, _ = geom.nearest(*c)
            dist = abs(s) / h
            if self._switch == "angle" and hasattr(geom, "frame"):  # the local normal of a curved interface
                nl = geom.frame(c, s, h, self.hinge)[1]
                D1 = (m_s + 0.5) * (abs(nl[0]) + abs(nl[1]))
                D2 = D1 + self._delta
            lam = 1.0 - _smooth((dist - D1) / (D2 - D1))
            comps = []
            if lam > 0 and self.kink:
                comps.append((lam, self._fit(ix, iy, m_pw, True, c, s)))
            if lam < 1 or not self.kink:
                comps.append((1.0 - lam if self.kink else 1.0, self._fit(ix, iy, m_s, False, c, s)))
            row.append(comps)
        return row, self.sigma_min, self.rank_log

    # -- frames and monomials ----------------------------------------------------------------------------
    def _frame(self, c, s):
        if hasattr(self.geom, "frame"):
            return self.geom.frame(c, s, self.h, self.hinge)
        n = self.geom.n
        tau = np.array([-n[1], n[0]])
        xg = c - s * n                        # the foot on the nearest interface line
        sgn = -1.0 if s > 0 else 1.0          # the kink acts on the side away from the cell centre
        return xg, n * sgn, tau

    def _mono(self, X, frame, pw):
        xg, n, tau = frame[:3]
        d = X - xg
        xi = d @ n / self.h
        eta = d @ tau / self.h
        cols = [xi ** p * eta ** q for p, q in P0_EXP]
        if pw:
            # the hinge variable: xi (a line) or a geometry's own (e.g. the signed distance to a circle, in cells)
            k = np.maximum(frame[3](X) if len(frame) > 3 and frame[3] is not None else xi, 0.0)
            cols += [k * xi ** p * eta ** q for p, q in P1_EXP]
        return np.stack(cols, axis=1)

    def _cell_points(self, jx, jy):
        """Quadrature of the (unwrapped) cell (jx, jy): points shifted to the unwrapped position, weights."""
        N, h = self.N, self.h
        cx, cy = jx % N, jy % N
        X = self.qx[cy * N + cx] + np.array([(jx - cx) * h, (jy - cy) * h])
        return X, self.qw[cy * N + cx], cy * N + cx

    def _fit(self, ix, iy, m, pw, c, s):
        frame = self._frame(c, s)
        if not pw and self.centre_single:
            # the single polynomial does not depend on the frame's origin (a full total-degree space): centre it at
            # the cell (docs/notes/90: about a foot ~20 cells away, xi^4 ~ 2e5 and the fit map rounds to 1e-9)
            frame = (np.asarray(c, dtype=float),) + tuple(frame[1:3])
        rows, cells, mass, geo = [], [], [], []
        h = self.h
        if pw and self.window == "interface":
            xw = np.asarray(frame[0])            # the foot of the cell centre on the interface (docs/notes/89: the
                                                 # frame's own foot; curved hinges do not flip the normal)
            R = self.R
            r0 = int(math.ceil(R + abs(s) / h + 1))
            offs = []
            for dy in range(-r0, r0 + 1):
                for dx in range(-r0, r0 + 1):
                    cj = c + np.array([dx, dy]) * h
                    r = float(np.linalg.norm(cj - xw)) / h
                    if r < R or (dx == 0 and dy == 0):
                        offs.append((dx, dy, max(1.0 - (r / R) ** 2, 0.0) ** 2))
        else:
            offs = [(dx, dy, 1.0) for dy in range(-m, m + 1) for dx in range(-m, m + 1)]
        k = None
        for t, (dx, dy, wg) in enumerate(offs):
            X, W, j = self._cell_points(ix + dx, iy + dy)
            rows.append(W @ self._mono(X, frame, pw) / W.sum())
            cells.append(j)
            mass.append(W.sum())
            geo.append(wg)
            if dx == 0 and dy == 0:
                k = t
        A = np.array(rows)
        mass = np.array(mass)
        sw = np.sqrt(np.array(geo) * (mass / mass.max()) ** self.alpha)
        if self.diff or self.tau is not None or self.window == "interface":
            return self._fit_split(A, sw, k, np.array(cells), frame, pw, s)
        Bw = sw[:, None] * A
        sv = np.linalg.svd(Bw, compute_uv=False)
        self.sigma_min = min(self.sigma_min, float(sv[-1] / sv[0]))
        Astar = np.linalg.lstsq(Bw, np.diag(sw), rcond=None)[0]
        k = (2 * m + 1) * m + m  # the cell itself
        e0 = np.zeros(A.shape[1])
        e0[0] = 1.0
        E = np.eye(len(cells))
        Wc = Astar + np.outer(e0, E[k] - A[k] @ Astar)
        return (np.array(cells), Wc, frame, pw)

    def _fit_split(self, A, sw, k, cells, frame, pw, s):
        """The shift on velocity differences, with the interface component (the hinge columns Y, after removing the
        ordinary polynomial X) inverted by the continuous filter f_tau (or plain lstsq when tau is None)."""
        n0 = len(P0_EXP)
        E = np.eye(len(cells))
        D = E - np.outer(np.ones(len(cells)), E[k])        # u - u_k 1
        Ad = A - A[k][None, :]                               # rows minus the cell's own row (constant drops out)
        Xw = sw[:, None] * Ad[:, 1:n0]
        Sw = np.diag(sw) @ D
        if pw:
            Yw = sw[:, None] * Ad[:, n0:]
            Xp = np.linalg.pinv(Xw)
            Z = Yw - Xw @ (Xp @ Yw)
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            rank = int(np.sum(S > 1e-10 * max(S[0], 1e-300)))
            self.rank_log.append((abs(s) / self.h, rank, Z.shape[1]))
            if self.tau is None:
                f = np.where(S > 1e-12 * max(S[0], 1e-300), 1.0 / np.where(S > 0, S, 1.0), 0.0)
            else:
                t = self.tau * max(S[0], 1e-300)
                f = np.where(S >= t, 1.0 / np.where(S > 0, S, 1.0), 2 * S / t ** 2 - S ** 3 / t ** 4)
            Zp = Vt.T @ (f[:, None] * U.T)
            Bmap = Zp @ (Sw - Xw @ (Xp @ Sw))                # hinge coefficients
            Amap = Xp @ (Sw - Yw @ Bmap)                     # ordinary coefficients (k >= 1)
            coef = np.vstack([Amap, Bmap])
        else:
            coef = np.linalg.lstsq(Xw, Sw, rcond=None)[0]
            sv = np.linalg.svd(Xw, compute_uv=False)
            self.sigma_min = min(self.sigma_min, float(sv[-1] / sv[0]))
        # P = u_k + sum_{m >= 1} (mono_m - A[k, m]) coef_m: constant coefficient = e_k - A[k, 1:] coef
        Wc = np.vstack([E[k] - A[k, 1:] @ coef, coef])
        return (cells, Wc, frame, pw)

    def basis(self, i, X):
        """Values at points X (near cell i's canonical position) of the basis functions on the cells: (cells, V)."""
        vals = {}
        for w, (cells, Wc, frame, pw) in self.comps[i]:
            V = self._mono(X, frame, pw) @ Wc
            for k, j in enumerate(cells):
                vals[j] = vals.get(j, 0.0) + w * V[:, k]
        idx = np.array(sorted(vals))
        return idx, np.stack([vals[j] for j in idx], axis=1)


class _Assembler:
    def __init__(self, N, rec0, rec1, U, dt):
        self.N, self.rec0, self.rec1, self.U, self.dt = N, rec0, rec1, U, dt

    def row(self, iy):
        """Triplets of G0, G1, C, Cg for the new cells of row iy."""
        N, rec0, rec1 = self.N, self.rec0, self.rec1
        h = 1.0 / N
        D = np.array(self.U) * self.dt
        geom = rec1.geom
        trip = {k: ([], [], []) for k in ("G0", "G1", "C", "Cg")}

        def add(k, r, c, v):
            R, Cc = np.meshgrid(r, c, indexing="ij")
            trip[k][0].append(R.ravel())
            trip[k][1].append(Cc.ravel())
            trip[k][2].append(np.asarray(v).ravel())

        for ix in range(N):
            i = iy * N + ix
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    jx, jy = ix + ox, iy + oy
                    x0, y0 = max(ix * h, jx * h + D[0]), max(iy * h, jy * h + D[1])
                    x1, y1 = min((ix + 1) * h, (jx + 1) * h + D[0]), min((iy + 1) * h, (jy + 1) * h + D[1])
                    if x1 - x0 < 1e-15 or y1 - y0 < 1e-15:
                        continue
                    j = (jy % N) * N + (jx % N)
                    shift_j = np.array([(jx - jx % N) * h, (jy - jy % N) * h])
                    for X, w, hv in pieces(geom, x0, y0, x1, y1):
                        wr = w * (1.0 if hv else 1.0 / geom.ratio)
                        i1, V1 = rec1.basis(i, X)
                        i0, V0 = rec0.basis(j, X - D - shift_j)
                        add("G1", i1, i1, (V1 * wr[:, None]).T @ V1)
                        add("G0", i0, i0, (V0 * wr[:, None]).T @ V0)
                        add("C", i1, i0, (V1 * wr[:, None]).T @ V0)
                        add("Cg", [i], i0, (wr @ V0)[None, :])
        n2 = N * N
        return {k: (np.concatenate(v[0]) * n2 + np.concatenate(v[1]), np.concatenate(v[2])) for k, v in trip.items()}

    def rows(self, iys):
        """Several rows: duplicates summed (unique linear indices and values)."""
        acc = {k: ([], []) for k in ("G0", "G1", "C", "Cg")}
        for iy in iys:
            for k, (ix_, v) in self.row(iy).items():
                acc[k][0].append(ix_)
                acc[k][1].append(v)
        out = {}
        for k, (ix_, v) in acc.items():
            ix_, v = np.concatenate(ix_), np.concatenate(v)
            u, inv = np.unique(ix_, return_inverse=True)
            out[k] = (u, np.bincount(inv, weights=v, minlength=u.size))
        return out


def assemble(N, rec0, rec1, U, dt, workers=1):
    """G0, G1, C and the direct remap Cg on the common partition of new cell x moved old cell x phase (rows of new
    cells over ``workers`` processes)."""
    n2 = N * N
    chunks = [list(range(N))[k::max(workers, 1)] for k in range(max(workers, 1))]
    out = _parallel(_Assembler(N, rec0, rec1, U, dt), "rows", [c for c in chunks if c], workers)
    mats = []
    for k in ("G0", "G1", "C", "Cg"):
        M = np.zeros(n2 * n2)
        for t in out:
            idx, v = t[k]
            M[idx] += v          # unique within each chunk
        mats.append(M.reshape(n2, n2))
    return tuple(mats)


def _assemble_serial(N, rec0, rec1, U, dt):
    """The original loop (kept for checking the parallel one)."""
    h = 1.0 / N
    D = np.array(U) * dt
    n2 = N * N
    G0, G1, C, Cg = (np.zeros((n2, n2)) for _ in range(4))
    geom = rec1.geom
    for iy in range(N):
        for ix in range(N):
            i = iy * N + ix
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    jx, jy = ix + ox, iy + oy
                    x0, y0 = max(ix * h, jx * h + D[0]), max(iy * h, jy * h + D[1])
                    x1, y1 = min((ix + 1) * h, (jx + 1) * h + D[0]), min((iy + 1) * h, (jy + 1) * h + D[1])
                    if x1 - x0 < 1e-15 or y1 - y0 < 1e-15:
                        continue
                    j = (jy % N) * N + (jx % N)
                    shift_j = np.array([(jx - jx % N) * h, (jy - jy % N) * h])
                    for X, w, hv in pieces(geom, x0, y0, x1, y1):
                        wr = w * (1.0 if hv else 1.0 / geom.ratio)
                        i1, V1 = rec1.basis(i, X)
                        i0, V0 = rec0.basis(j, X - D - shift_j)
                        G1[np.ix_(i1, i1)] += (V1 * wr[:, None]).T @ V1
                        G0[np.ix_(i0, i0)] += (V0 * wr[:, None]).T @ V0
                        C[np.ix_(i1, i0)] += (V1 * wr[:, None]).T @ V0
                        Cg[i, i0] += wr @ V0
    return G0, G1, C, Cg


def averages(N, geom, f):
    """Mass-weighted cell averages of f(x, y) (exact geometry, Gauss on the pieces)."""
    h = 1.0 / N
    out = np.zeros(N * N)
    for iy in range(N):
        for ix in range(N):
            num = den = 0.0
            for X, w, hv in pieces(geom, ix * h, iy * h, (ix + 1) * h, (iy + 1) * h):
                r = 1.0 if hv else 1.0 / geom.ratio
                num += r * float(w @ f(X[:, 0], X[:, 1]))
                den += r * float(w.sum())
            out[iy * N + ix] = num / den
    return out


def field(geom0, kind="phases"):
    """q(x, y) at t = 0: smooth, continuous across the interfaces, its normal gradient jumps there."""
    def q(x, y):
        z = np.mod(geom0.L(x, y) - geom0.d, 1.0)
        base = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        if kind == "smooth":
            return base
        if kind == "shear":  # the user's exact shear: a velocity component tau f(z), f continuous, kinked
            bump = np.where((z > geom0.zA) & (z < geom0.zB), (z - geom0.zA) * (geom0.zB - z), 0.0)
            return np.sin(2 * np.pi * z) + 3.0 * bump
        bump = np.where((z > geom0.zA) & (z < geom0.zB), (z - geom0.zA) * (geom0.zB - z), 0.0)
        return base + 3.0 * bump * np.cos(2 * np.pi * (x - y))
    return q


def run(N, transport="galerkin", ratio=1e3, a=1, b=1, U=(1.0, 0.5), t_end=0.1, cfl=0.37, kind="phases",
        alpha=0.75, samples=(), rk=None, workers=1):
    h = 1.0 / N
    umax = max(abs(U[0]), abs(U[1]))
    steps = int(math.ceil(t_end / (cfl * h / umax)))
    dt = t_end / steps
    g0 = Geometry(a, b, ratio=ratio)
    q0 = field(g0, kind)
    exact = lambda t: averages(N, g0.moved(U, t), lambda x, y: q0(x - U[0] * t, y - U[1] * t))  # noqa: E731
    rk = {**(rk or {}), "workers": workers}
    rec0 = Reconstruction2D(N, g0, alpha, **rk)
    u = exact(0.0)
    cond_max, umax = 0.0, 0.0
    u_init_max = float(np.abs(u).max())
    E0 = None
    errs, infs, smin = {}, 0.0, rec0.sigma_min
    at = {int(round(t / dt)): t for t in samples}
    for k in range(steps):
        rec1 = Reconstruction2D(N, g0.moved(U, (k + 1) * dt), alpha, **rk)
        smin = min(smin, rec1.sigma_min)
        G0, G1, C, Cg = assemble(N, rec0, rec1, U, dt, workers)
        if E0 is None:
            E0 = float(u @ G0 @ u)
        if transport == "geo":
            T = Cg / rec1.mass[:, None]
        else:
            T = np.linalg.solve(G1, C)
            if transport == "corr":
                T = T + 0.5 * (T - T @ np.linalg.solve(G0, C.T @ T))
        infs = max(infs, float(np.abs(T).sum(1).max()))
        Mh = np.diag(1 / np.sqrt(rec1.mass))
        ev = np.linalg.eigvalsh(Mh @ G1 @ Mh)
        cond_max = max(cond_max, float(ev.max() / ev.min()))
        u = T @ u
        umax = max(umax, float(np.abs(u).max()))
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(u - exact((k + 1) * dt)).max())
        rec0 = rec1
        Glast = G1
    errs[t_end] = float(np.abs(u - exact(t_end)).max())
    Mh = np.diag(1 / np.sqrt(rec0.mass))
    ev = np.linalg.eigvalsh(Mh @ Glast @ Mh)
    return {"errors": errs, "energy": float(u @ Glast @ u) / E0, "inf_step": infs, "steps": steps,
            "cond": float(ev.max() / ev.min()), "cond_max": cond_max, "sigma_min": smin,
            "umax_ratio": float(np.abs(u).max()) / u_init_max, "umax_over_run": umax / u_init_max}


def free_stream(N, ratio=1e3, a=1, b=1, U=(1.0, 0.5), cfl=0.37, rk=None):
    h = 1.0 / N
    g0 = Geometry(a, b, ratio=ratio)
    dt = cfl * h / max(abs(U[0]), abs(U[1]))
    rk = rk or {}
    rec0, rec1 = Reconstruction2D(N, g0, **rk), Reconstruction2D(N, g0.moved(U, dt), **rk)
    G0, G1, C, Cg = assemble(N, rec0, rec1, U, dt)
    T = np.linalg.solve(G1, C)
    one = np.ones(N * N)
    v = np.sin(np.arange(N * N))
    # block matrix positive semi-definite (energy cannot grow)
    blk = np.block([[G0, C.T], [C, G1]])
    return {"free_stream": float(np.abs(T @ one - 1).max()),
            "momentum": float(abs(rec1.mass @ (T @ v) - rec0.mass @ v) / rec0.mass.sum()),
            "mass_G1": float(np.abs(G1 @ one - rec1.mass).max() / rec1.mass.max()),
            "block_min_eig": float(np.linalg.eigvalsh(blk).min() / np.abs(blk).max()),
            "remap_free_stream": float(np.abs(Cg @ one / rec1.mass - 1).max())}


def exactness(N, ratio=1e3, a=1, b=1, seed=0, near=2.5, **rk):
    """Each of the 25 local basis functions (xi^p eta^q, and max(xi, 0) xi^p eta^q, xi and eta in cells, about the
    line a x + b y = zA), one at a time, against the reconstruction in the cells within ``near`` cells of the line;
    the largest error relative to the function's largest value on the checked cells (docs/notes/88). ``rk``:
    options of Reconstruction2D."""
    g = Geometry(a, b, ratio=ratio)
    rec = Reconstruction2D(N, g, **rk)
    h = 1.0 / N
    n = g.n
    tau = np.array([-n[1], n[0]])
    xg = n * g.zA / g.g
    funcs = [(False, e) for e in P0_EXP] + [(True, e) for e in P1_EXP]
    worst = 0.0
    targets = []
    for iy in range(N):
        for ix in range(N):
            cxy = np.array([(ix + 0.5) * h, (iy + 0.5) * h])
            if a:
                sx, sy = -round((g.L(*cxy) - g.zA) / a), 0
            else:
                sx, sy = 0, -round((g.L(*cxy) - g.zA) / b)
            if abs(g.L(cxy[0] + sx, cxy[1] + sy) - g.zA) / g.g / h <= near:
                targets.append((ix, iy, np.array([sx, sy], dtype=float)))
    for hinge, (p, q_) in funcs:
        def q(X):
            d = X - xg
            xi, eta = d @ n / h, d @ tau / h
            v = xi ** p * eta ** q_
            return np.maximum(xi, 0.0) * v if hinge else v
        err, scale = 0.0, 0.0
        for ix, iy, sh in targets:
            i = iy * N + ix
            X = rec.qx[i]
            idx, V = rec.basis(i, X)
            u = {}
            for dy in range(-7, 8):
                for dx in range(-7, 8):
                    Xc, W, j = rec._cell_points(ix + dx, iy + dy)
                    u[j] = float(W @ q(Xc + sh)) / W.sum()
            ref = q(X + sh)
            err = max(err, float(np.abs(V @ np.array([u[j] for j in idx]) - ref).max()))
            scale = max(scale, float(np.abs(ref).max()))
        worst = max(worst, err / max(scale, 1e-300))
    return worst


def one_step(N, ratio=1e3, a=1, b=1, U=(1.0, 0.5), cfl=0.37, kind="phases", offsets=6, rk=None, transport="galerkin"):
    """docs/notes/88: the largest one-step error, ||T||_inf and cond over ``offsets`` interface positions across one
    cell (the same geometric statistics for every N)."""
    h = 1.0 / N
    rk = rk or {}
    dt = cfl * h / max(abs(U[0]), abs(U[1]))
    err = inf = cond = 0.0
    for o in np.linspace(0.0, 1.0, offsets, endpoint=False):
        g0 = Geometry(a, b, ratio=ratio, d=o * h * math.hypot(a, b))
        q0 = field(g0, kind)
        rec0, rec1 = Reconstruction2D(N, g0, **rk), Reconstruction2D(N, g0.moved(U, dt), **rk)
        G0, G1, C, Cg = assemble(N, rec0, rec1, U, dt)
        T = Cg / rec1.mass[:, None] if transport == "geo" else np.linalg.solve(G1, C)
        u0 = averages(N, g0, q0)
        ex = averages(N, g0.moved(U, dt), lambda x, y: q0(x - U[0] * dt, y - U[1] * dt))
        err = max(err, float(np.abs(T @ u0 - ex).max()))
        inf = max(inf, float(np.abs(T).sum(1).max()))
        Mh = np.diag(1 / np.sqrt(rec1.mass))
        ev = np.linalg.eigvalsh(Mh @ G1 @ Mh)
        cond = max(cond, float(ev.max() / ev.min()))
    return {"err": err, "inf": inf, "cond": cond}


def recon_jump(N, a=1, b=1, ratio=1e3, spacing=4e-3, rk=None, seed=0):
    """The largest change of the reconstructed field (random cell data, 5 x 5 points per cell) between interface
    positions ``spacing`` cells apart (normal), over one cell: a jump does not shrink with the spacing."""
    h = 1.0 / N
    rk = rk or {}
    u = np.random.default_rng(seed).normal(size=N * N)
    pts = [np.stack(np.meshgrid((ix + np.linspace(0.1, 0.9, 5)) * h, (iy + np.linspace(0.1, 0.9, 5)) * h), -1)
           .reshape(-1, 2) for iy in range(N) for ix in range(N)]
    worst, where, prev = 0.0, None, None
    for o in np.arange(0.0, 1.0, spacing):
        rec = Reconstruction2D(N, Geometry(a, b, ratio=ratio, d=o * h * math.hypot(a, b)), **rk)
        vals = []
        for i in range(N * N):
            idx, V = rec.basis(i, pts[i])
            vals.append(V @ u[idx])
        vals = np.concatenate(vals)
        if prev is not None:
            dv = float(np.abs(vals - prev).max())
            if dv > worst:
                worst, where = dv, float(o)
        prev = vals
    return {"max_change": worst, "at_offset": where}


def continuity(N, a=1, b=1, ratio=1e3, spacing=4e-3, rk=None):
    """The largest relative change of G between interface positions ``spacing`` cells apart (normal), over one cell:
    a jump of the reconstruction does not shrink with the spacing."""
    h = 1.0 / N
    rk = rk or {}
    worst, prev = 0.0, None
    U, dt = (1.0, 0.0), 1e-9
    for o in np.arange(0.0, 1.0, spacing):
        g = Geometry(a, b, ratio=ratio, d=o * h * math.hypot(a, b))
        G1 = assemble(N, Reconstruction2D(N, g, **rk), Reconstruction2D(N, g.moved(U, dt), **rk), U, dt)[1]
        if prev is not None:
            worst = max(worst, float(np.linalg.norm(G1 - prev, 2) / np.linalg.norm(G1, 2)))
        prev = G1
    return worst


def main() -> None:
    import json
    from concurrent.futures import ProcessPoolExecutor

    out = {}
    with ProcessPoolExecutor(12) as ex:
        fs = {N: ex.submit(free_stream, N) for N in (16, 24)}
        runs = {(tr, N): ex.submit(run, N, tr, samples=(0.05,)) for tr in ("geo", "galerkin", "corr")
                for N in (16, 24, 32, 48)}
        for N, f in fs.items():
            out[f"free_stream {N}"] = f.result()
            print("free stream", N, out[f"free_stream {N}"], flush=True)
        for (tr, N), f in runs.items():
            out[f"run {tr} {N}"] = f.result()
            print("run", tr, N, out[f"run {tr} {N}"], flush=True)
    json.dump(out, open("docs/figures/galerkin2d.json", "w"), indent=1)


if __name__ == "__main__":
    main()

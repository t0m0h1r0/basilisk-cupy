"""docs/notes/89 (P1-2): a circular interface translated by a constant velocity, exact geometry.

A heavy disc (radius R0, periodic copies) in a light fluid. The reconstruction of docs/notes/88 (interface-centred
window, radius 5.5 cells, angle-dependent switch, velocity differences) with two hinge variables (the user's
comparison):
* "exact": max(d_Gamma(x)/h, 0) with d_Gamma = |x - c| - R0 the exact signed distance (positive outside; the kink's
  side is not flipped: d is not a polynomial, a flipped hinge would differ by d P1);
* "tangent": max(xi, 0), xi the coordinate along the normal of the tangent line at the cell's foot (note 88);
* "quadratic": max(xi + (h / 2 R0) eta^2, 0), the osculating parabola (the user's C; unflipped, like "exact");
* "height": the parabola of the height functions of the volume fractions (3 x 7 stencil; the practical form of D,
  docs/notes/90): only the fractions, no exact geometry, define the hinge (the switch still uses the exact distance).

Cells cut by the circle are integrated piecewise: the x-range is split where the circle crosses the cell's
horizontal sides and at the circle's extreme abscissae; on each sub-interval the y-limits are smooth functions of x,
Gauss in y (exact for the polynomial inner integrands) and Gauss in t with x = a + (b - a)(1 - cos pi t)/2 (which
removes the square-root behaviour at a tangency).

    python -m cases.galerkin2d_circle [--workers W] [--jobs J] [--hinges exact,tangent,quadratic] [--remap 1]
    python -m cases.galerkin2d_circle --hinges height --remap 0 --out docs/figures/galerkin2d_circle_height.json
    python -m cases.galerkin2d_circle --hinges exact --remap 0 --exactness 0 --transports poly4,poly6,poly8 \
        --out docs/figures/galerkin2d_poly.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from cases.galerkin2d import P0_EXP, P1_EXP, Reconstruction2D, assemble, averages

_tx, _twx = np.polynomial.legendre.leggauss(16)
TX, TWX = 0.5 * (_tx + 1.0), 0.5 * _twx
_ty, _twy = np.polynomial.legendre.leggauss(6)
TY, TWY = 0.5 * (_ty + 1.0), 0.5 * _twy
RK = {"window": "interface", "switch": "angle", "diff": True, "R": 5.5}


class DistanceHinge:
    """The signed distance to a circle, in cells (a picklable callable: the fits run in worker processes)."""

    def __init__(self, ctr, R0, h):
        self.ctr, self.R0, self.h = np.asarray(ctr), R0, h

    def __call__(self, X):
        return (np.linalg.norm(X - self.ctr, axis=1) - self.R0) / self.h


class QuadraticHinge:
    """The local quadratic approximation of the distance (user's C): xi + (h / 2 R0) eta^2, in cells, about the foot
    (a polynomial, so the outside function P0 + hinge P1 stays a polynomial)."""

    def __init__(self, xg, n, tau, R0, h):
        self.xg, self.n, self.tau, self.c, self.h = np.asarray(xg), np.asarray(n), np.asarray(tau), h / (2 * R0), h

    def __call__(self, X):
        Y = (X - self.xg) / self.h
        return Y @ self.n + self.c * (Y @ self.tau) ** 2


class HeightHinge:
    """The height-function parabola from the volume fractions (the user's D, its practical form): with vertical
    heights Y(x) = a + b xi + c xi^2 (xi = (x - x_I) / h, matching the three column averages), the defining function
    side * (y - Y(x)) / (h sqrt(1 + Y'(x_G)^2)), positive on the light side; ``swap``: horizontal heights."""

    def __init__(self, xI, a, b, c, side, swap, h, scale):
        self.xI, self.a, self.b, self.c, self.side, self.swap, self.h, self.scale = xI, a, b, c, side, swap, h, scale

    def Y(self, x):
        xi = (x - self.xI) / self.h
        return self.a + self.h * (self.b * xi + self.c * xi * xi)

    def __call__(self, X):
        x, y = (X[:, 1], X[:, 0]) if self.swap else (X[:, 0], X[:, 1])
        return self.side * (y - self.Y(x)) / (self.h * self.scale)


def fractions(geom, N):
    h = 1.0 / N
    F = np.zeros((N, N))
    for iy in range(N):
        for ix in range(N):
            F[iy, ix] = sum(w.sum() for _, w, hv in geom.pieces(ix * h, iy * h, (ix + 1) * h, (iy + 1) * h) if hv)
    return F / (h * h)


def height_frame(F, xg, c, h):
    """The frame of the height-function parabola of the interface cell holding x_G (only used to locate the
    cell; the curve itself comes from the fractions F[iy, ix] of a 3 x 7 stencil)."""
    N = F.shape[0]
    I, J = int(math.floor(xg[0] / h)), int(math.floor(xg[1] / h))
    f = lambda i, j: F[j % N, i % N]  # noqa: E731
    gx = sum(f(I + 1, J + dj) - f(I - 1, J + dj) for dj in (-1, 0, 1))
    gy = sum(f(I + di, J + 1) - f(I + di, J - 1) for di in (-1, 0, 1))
    swap = abs(gx) > abs(gy)                   # the interface is closer to vertical: horizontal heights
    if swap:
        f = lambda i, j: F[i % N, j % N]  # noqa: E731,F811  (rows and columns exchanged)
        I, J, gy = J, I, gx
        x0, y0 = c[1], c[0]
    else:
        x0, y0 = c[0], c[1]
    heavy_below = gy < 0
    ys = []
    for di in (-1, 0, 1):
        H = sum(f(I + di, J + dj) for dj in range(-3, 4))
        ys.append((J - 3) * h + H * h if heavy_below else (J + 4) * h - H * h)
    cc = (ys[2] - 2 * ys[1] + ys[0]) / 2 / h
    b = (ys[2] - ys[0]) / 2 / h
    a = ys[1] - cc * h / 12
    xI = (I + 0.5) * h
    side = 1.0 if heavy_below else -1.0
    # the foot: the nearest point of the parabola to the cell centre (Newton on xi)
    xi = (x0 - xI) / h
    for _ in range(30):
        Yv = a / h + b * xi + cc * xi * xi
        Yp = b + 2 * cc * xi
        r = (xi - (x0 - xI) / h) + (Yv - y0 / h) * Yp
        dr = 1 + Yp * Yp + (Yv - y0 / h) * 2 * cc
        xi -= r / dr
    xf, yf = xI + h * xi, a + h * (b * xi + cc * xi * xi)
    slope = b + 2 * cc * xi
    n = side * np.array([-slope, 1.0]) / math.hypot(slope, 1.0)      # towards the light side
    hinge = HeightHinge(xI, a, b, cc, side, swap, h, math.hypot(slope, 1.0))
    if swap:
        return np.array([yf, xf]), n[::-1].copy(), hinge
    return np.array([xf, yf]), n, hinge


class Circle:
    def __init__(self, cx=0.5, cy=0.5, R0=0.25, ratio=1e3):
        self.cx, self.cy, self.R0, self.ratio = cx, cy, R0, ratio

    def moved(self, U, t):
        return Circle(self.cx + U[0] * t, self.cy + U[1] * t, self.R0, self.ratio)

    def _copy(self, x, y):
        """The copy of the centre nearest to (x, y)."""
        return (x - ((x - self.cx + 0.5) % 1.0 - 0.5), y - ((y - self.cy + 0.5) % 1.0 - 0.5))

    def heavy(self, x, y):
        dx = (x - self.cx + 0.5) % 1.0 - 0.5
        dy = (y - self.cy + 0.5) % 1.0 - 0.5
        return dx * dx + dy * dy < self.R0 ** 2

    def nearest(self, x, y):
        cx, cy = self._copy(x, y)
        return math.hypot(x - cx, y - cy) - self.R0, None

    def frame(self, c, s, h, hinge):
        cx, cy = self._copy(*c)
        v = np.array([c[0] - cx, c[1] - cy])
        r = float(np.linalg.norm(v))
        n = v / r if r > 0 else np.array([1.0, 0.0])
        xg = np.array([cx, cy]) + self.R0 * n
        tau = np.array([-n[1], n[0]])
        if hinge == "exact":
            return (xg, n, tau, DistanceHinge([cx, cy], self.R0, h))
        if hinge == "quadratic":
            return (xg, n, tau, QuadraticHinge(xg, n, tau, self.R0, h))
        if hinge == "height":
            N = int(round(1.0 / h))
            if getattr(self, "_F", (None, None))[0] != N:
                self._F = (N, fractions(self, N))
            xf, nf, hz = height_frame(self._F[1], xg, c, h)
            return (xf, nf, np.array([-nf[1], nf[0]]), hz)
        sgn = -1.0 if s > 0 else 1.0
        return (xg, n * sgn, tau, None)

    def pieces(self, x0, y0, x1, y1):
        cx, cy = self._copy(0.5 * (x0 + x1), 0.5 * (y0 + y1))
        R = self.R0
        bx = [x0, x1]
        for xe in (cx - R, cx + R):
            if x0 < xe < x1:
                bx.append(xe)
        for yy in (y0, y1):
            dd = R * R - (yy - cy) ** 2
            if dd > 0:
                for xe in (cx - math.sqrt(dd), cx + math.sqrt(dd)):
                    if x0 < xe < x1:
                        bx.append(xe)
        bx = sorted(set(bx))
        out = []
        for a, b in zip(bx[:-1], bx[1:]):
            if b - a < 1e-15:
                continue
            xm = 0.5 * (a + b)
            inside = abs(xm - cx) < R
            t = TX
            x = a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * t))
            wx = TWX * (b - a) * 0.5 * np.pi * np.sin(np.pi * t)
            half = np.sqrt(np.maximum(R * R - (x - cx) ** 2, 0.0)) if inside else np.zeros_like(x)
            lo = np.clip(cy - half, y0, y1)
            hi = np.clip(cy + half, y0, y1)
            spans = []
            if inside and (np.mean(hi - lo) > 0):
                spans.append((lo, hi, True))
            if not inside:
                spans.append((np.full_like(x, y0), np.full_like(x, y1), False))
            else:
                if np.mean(lo - y0) > 0:
                    spans.append((np.full_like(x, y0), lo, False))
                if np.mean(y1 - hi) > 0:
                    spans.append((hi, np.full_like(x, y1), False))
            for ylo, yhi, hv in spans:
                Y = ylo[:, None] + (yhi - ylo)[:, None] * TY[None, :]
                W = wx[:, None] * (yhi - ylo)[:, None] * TWY[None, :]
                X = np.stack([np.repeat(x, TY.size), Y.ravel()], axis=1)
                w = W.ravel()
                keep = w > 0
                if keep.any():
                    out.append((X[keep], w[keep], hv))
        return out


def poly_nodes(G1, m, k):
    """Richardson step sizes 1/lam_i for the residual polynomial q(lam) = (1 - lam) prod_j (1 - lam/lam_j) on the
    spectrum of A = M^{-1/2} G1 M^{-1/2}, which lies in [1, b] (docs/notes/90 E):
    * lambda_min >= 1: the shift keeps every cell's mass-weighted mean, so int_K rho P^2 >= m_K (mean)^2 (Cauchy-
      Schwarz) and G1 >= M; G1 1 = m makes M^{1/2} 1 an eigenvector with eigenvalue 1 (the root of q at 1);
    * b: the Gershgorin bound (rigorous: |q| <= sqrt 2 must hold on all of spec A for the energy bound);
    the other k - 1 roots are the Chebyshev roots of [1, b]."""
    s = 1.0 / np.sqrt(m)
    b = float((np.abs(G1) * s[:, None] * s[None, :]).sum(1).max())
    j = np.arange(1, k)
    lam = 0.5 * (1.0 + b) + 0.5 * (b - 1.0) * np.cos((2 * j - 1) * np.pi / (2 * (k - 1))) if k > 1 else np.array([])
    return np.concatenate([[1.0], lam]), (1.0, b)


def poly_transport(G1, C, m, k):
    """The polynomial-approximated Galerkin map (docs/notes/90 E): two Richardson sweeps preconditioned by the mass,
    x <- x + (1/lam_i) M^{-1}(C u - G1 x), give the residual polynomial r = q^2 in [0, 2]: the energy cannot grow,
    and r(1) = 0 keeps the free stream and the total momentum exact. Only products with G1, C and the diagonal M."""
    nodes, _ = poly_nodes(G1, m, k)
    X = np.zeros_like(C)
    for _ in range(2):
        for lam in nodes:
            X = X + (C - G1 @ X) / (lam * m[:, None])
    return X


def field(c0):
    """q at t = 0: smooth outside, + 3 (R0^2 - r^2) cos 2 pi (x - y) inside (continuous, normal gradient jumps)."""
    def q(x, y):
        dx = (x - c0.cx + 0.5) % 1.0 - 0.5
        dy = (y - c0.cy + 0.5) % 1.0 - 0.5
        r2 = dx * dx + dy * dy
        base = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        return base + np.where(r2 < c0.R0 ** 2, 3.0 * (c0.R0 ** 2 - r2) * np.cos(2 * np.pi * (x - y)), 0.0)
    return q


def exactness(N, hinge, ratio=1e3, seed=0, near=2.5, workers=1):
    """Random P0 + max(d/h, 0) P1 (polynomials in x/h, y/h about the centre, d the exact signed distance) against
    the reconstruction in the cells within ``near`` cells of the circle (relative error)."""
    g = Circle(ratio=ratio)
    rec = Reconstruction2D(N, g, hinge=hinge, workers=workers, **RK)
    h = 1.0 / N
    rng = np.random.default_rng(seed)
    c0, c1 = rng.normal(size=15), rng.normal(size=10)
    err, scale = 0.0, 0.0
    for iy in range(N):
        for ix in range(N):
            c = np.array([(ix + 0.5) * h, (iy + 0.5) * h])
            s, _ = g.nearest(*c)
            if abs(s) / h > near:
                continue
            ctr = np.array(g._copy(*c))

            def q(X):
                xi, eta = (X[:, 0] - ctr[0]) / h, (X[:, 1] - ctr[1]) / h
                d = (np.linalg.norm(X - ctr, axis=1) - g.R0) / h
                p0 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c0, P0_EXP)) / 10.0 ** 4
                p1 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c1, P1_EXP)) / 10.0 ** 3
                return p0 + np.maximum(d, 0.0) * p1
            i = iy * N + ix
            idx, V = rec.basis(i, rec.qx[i])
            u = {}
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    Xc, W, j = rec._cell_points(ix + dx, iy + dy)
                    u[j] = float(W @ q(Xc)) / W.sum()
            ref = q(rec.qx[i])
            err = max(err, float(np.abs(V @ np.array([u[j] for j in idx]) - ref).max()))
            scale = max(scale, float(np.abs(ref).max()))
    return err / scale


def run(N, hinge, transport="galerkin", ratio=1e3, U=(1.0, 0.5), t_end=0.1, cfl=0.37, samples=(0.05,),
        workers=1):
    h = 1.0 / N
    steps = int(math.ceil(t_end / (cfl * h / max(abs(U[0]), abs(U[1])))))
    dt = t_end / steps
    g0 = Circle(ratio=ratio)
    q0 = field(g0)
    exact = lambda t: averages(N, g0.moved(U, t), lambda x, y: q0(x - U[0] * t, y - U[1] * t))  # noqa: E731
    rk = {**RK, "hinge": hinge, "workers": workers}
    rec0 = Reconstruction2D(N, g0, **rk)
    u = exact(0.0)
    u0max = float(np.abs(u).max())
    E0, errs, inf, condm, umax = None, {}, 0.0, 0.0, 0.0
    at = {int(round(t / dt)): t for t in samples}
    for k in range(steps):
        rec1 = Reconstruction2D(N, g0.moved(U, (k + 1) * dt), **rk)
        G0, G1, C, Cg = assemble(N, rec0, rec1, U, dt, workers)
        if E0 is None:
            E0 = float(u @ G0 @ u)
        if transport == "geo":
            T = Cg / rec1.mass[:, None]
        elif transport.startswith("poly"):
            T = poly_transport(G1, C, rec1.mass, int(transport[4:]))
        else:
            T = np.linalg.solve(G1, C)
        inf = max(inf, float(np.abs(T).sum(1).max()))
        Mh = np.diag(1 / np.sqrt(rec1.mass))
        ev = np.linalg.eigvalsh(Mh @ G1 @ Mh)
        condm = max(condm, float(ev.max() / ev.min()))
        u = T @ u
        umax = max(umax, float(np.abs(u).max()))
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(u - exact((k + 1) * dt)).max())
        rec0, Gl = rec1, G1
    errs[t_end] = float(np.abs(u - exact(t_end)).max())
    one = np.ones(N * N)
    return {"errors": errs, "energy": float(u @ Gl @ u) / E0, "inf_step": inf, "cond_max": condm,
            "umax_over_run": umax / u0max, "steps": steps, "free_stream_last": float(np.abs(T @ one - 1).max())}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--hinges", default="exact,tangent,quadratic")
    ap.add_argument("--remap", type=int, default=1)
    ap.add_argument("--exactness", type=int, default=1)
    ap.add_argument("--transports", default="galerkin")
    ap.add_argument("--out", default="docs/figures/galerkin2d_circle.json")
    a = ap.parse_args()
    hinges = a.hinges.split(",")
    jobs = [("exact", exactness, {"N": N, "hinge": hz}) for hz in hinges for N in (24, 32, 48)] if a.exactness else []
    for N in (64, 48, 32, 24):
        for hz in hinges:
            for tr in a.transports.split(","):
                jobs.append(("run", run, {"N": N, "hinge": hz, "transport": tr}))
        if a.remap:
            jobs.append(("run", run, {"N": N, "hinge": "exact", "transport": "geo"}))
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [(n, kw, ex.submit(fn, **kw, workers=a.workers)) for n, fn, kw in jobs]
        for n, kw, f in futs:
            r = f.result()
            res.append({"test": n, "args": kw, "result": r})
            print(f"{time.time() - t0:7.0f}s", n, kw, r, flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

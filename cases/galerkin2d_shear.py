"""P1-3 (prototype): a disc deformed by an unsteady shear with an exact flow map (no directional splitting).

    u = (A sin(2 pi y) cos(pi t / T), 0),   Phi_t(X, Y) = (X + S(t) sin 2 pi Y, Y),   S(t) = (A T / pi) sin(pi t / T)

The map is area preserving, the interface at time t is the image of the circle |X - c| = R0, and the flow reverses at
t = T / 2 (the disc returns at t = T). Between two times the old point of x is (x - (S1 - S0) sin 2 pi y, y): the
moved old cells are no longer rectangles, their vertical faces become the curves x = j h + dS sin 2 pi y.

Quadrature (both for the cell averages and for the common partition): y outside, x inside. For a fixed y every
boundary of the partition (cell faces, moved old faces, the two interface points) is an x location given in closed
form; the y-range is split where two of them cross (and at the interface's tangencies), then Gauss in u with
y = y_t +- u^2 from the nearer tangency y_t (inside the disc's y-range: the square root becomes smooth in u, also on a
piece that ends just short of the tangency), and Gauss in x on each x-interval between consecutive boundaries.

Hinges: "exact" max(phi / (h |grad phi(x_G)|), 0) with phi = |Phi_t^{-1}(x) - c| - R0 (a smooth defining function, not
the distance: by Hadamard's lemma it gives the same approximation, docs/notes/89 B), "tangent", "quadratic"
(xi + (kappa h / 2) eta^2, kappa the curve's curvature at the foot).

    python -m cases.galerkin2d_shear [--workers W] [--jobs J]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from cases.galerkin2d import P0_EXP, P1_EXP, Reconstruction2D, _parallel
from cases.galerkin2d_circle import RK, Circle, field

_t, _tw = np.polynomial.legendre.leggauss(12)
TY, TWY = 0.5 * (_t + 1.0), 0.5 * _tw
_x, _xw = np.polynomial.legendre.leggauss(6)
TX, TWX = 0.5 * (_x + 1.0), 0.5 * _xw
TH = np.linspace(0.0, 2 * np.pi, 721)[:-1]


def _roots(g, a, b, n=64, monotone=False):
    """Zeros of g on [a, b]: first split at the extrema of g (zeros of its difference quotient), so that a pair of
    close roots around an extremum is not missed, then sign changes on each monotone piece, refined by bisection."""
    if not monotone:
        e = 1e-7 * (b - a)
        brk = [a] + _roots(lambda y: (g(y + e) - g(y - e)) / (2 * e), a, b, n, monotone=True) + [b]
        out = []
        for lo, hi in zip(brk[:-1], brk[1:]):
            if hi - lo > 1e-14:
                out += _roots(g, lo, hi, 8, monotone=True)
        return out + brk[1:-1]
    ys = np.linspace(a, b, n + 1)
    v = g(ys)
    out = []
    for k in range(n):
        va, vb = v[k], v[k + 1]
        if not (np.isfinite(va) and np.isfinite(vb)) or va * vb > 0 or va == vb:
            continue
        lo, hi, flo = ys[k], ys[k + 1], va
        for _ in range(60):
            m = 0.5 * (lo + hi)
            fm = float(g(np.array([m]))[0])
            if not np.isfinite(fm):
                break
            if fm * flo > 0:
                lo, flo = m, fm
            else:
                hi = m
        r = 0.5 * (lo + hi)
        if a < r < b:
            out.append(r)
    return out


def strips(x0, x1, y0, y1, curves, ybreaks, label, tangency=None):
    """Quadrature of the rectangle partitioned by the x-curves f(y) (NaN where absent): {label: (X, w)}.
    ``label(x, y)`` gives an integer label to each x-interval (evaluated at its midpoint); ``tangency`` = (y_lo, y_hi),
    the ordinates where the curves end with a square-root behaviour."""
    base = sorted(set([y0, y1] + [y for y in ybreaks if y0 < y < y1]))
    edges = [lambda y, c=x0: np.full_like(y, c), lambda y, c=x1: np.full_like(y, c)]
    allc = curves + edges
    cut = list(base)
    for ya, yb in zip(base[:-1], base[1:]):          # the roots between the tangencies (curves defined at them)
        for a in range(len(curves)):
            for b in range(a + 1, len(allc)):
                cut += _roots(lambda y, fa=allc[a], fb=allc[b]: fa(y) - fb(y), ya, yb)
    cut = sorted(set(cut))
    out = {}
    for ya, yb in zip(cut[:-1], cut[1:]):
        if yb - ya < 1e-15:
            continue
        if tangency is not None and tangency[0] <= ya and yb <= tangency[1]:
            # y = y_t +- u^2 from the nearer tangency: the half-width sqrt(R0^2 - (y - c)^2) = u sqrt(2 R0 - u^2) is
            # smooth in u, also on sub-intervals that end just short of the tangency (a nearby root)
            yt, sg = (tangency[0], 1.0) if ya + yb < tangency[0] + tangency[1] else (tangency[1], -1.0)
            ua, ub = sorted((math.sqrt(abs(ya - yt)), math.sqrt(abs(yb - yt))))
            uu = ua + (ub - ua) * TY
            y = yt + sg * uu * uu
            wy = TWY * (ub - ua) * 2.0 * uu
        else:
            y = ya + (yb - ya) * TY
            wy = TWY * (yb - ya)
        F = np.stack([np.nan_to_num(f(y), nan=x0) for f in curves], axis=1) if curves else np.zeros((y.size, 0))
        F = np.sort(np.clip(np.concatenate([np.full((y.size, 1), x0), F, np.full((y.size, 1), x1)], axis=1), x0, x1),
                    axis=1)
        lo, hi = F[:, :-1], F[:, 1:]                         # (ny, nint)
        L = hi - lo
        lab = label(0.5 * (lo + hi), np.repeat(y[:, None], lo.shape[1], axis=1))
        X = lo[:, :, None] + L[:, :, None] * TX[None, None, :]
        W = wy[:, None, None] * L[:, :, None] * TWX[None, None, :]
        Y = np.broadcast_to(y[:, None, None], X.shape)
        lab = np.broadcast_to(lab[:, :, None], X.shape)
        keep = W > 0
        for v in np.unique(lab[keep]):
            m = keep & (lab == v)
            P = np.stack([X[m], Y[m]], axis=1)
            if v in out:
                out[v] = (np.vstack([out[v][0], P]), np.concatenate([out[v][1], W[m]]))
            else:
                out[v] = (P, W[m])
    return out


class PhiHinge:
    """phi(x) / (h |grad phi(x_G)|), phi = |Phi^{-1}(x) - c_k| - R0 for one periodic copy c_k (picklable)."""

    def __init__(self, ck, R0, S, h, g0):
        self.ck, self.R0, self.S, self.h, self.g0 = np.asarray(ck), R0, S, h, g0

    def __call__(self, X):
        q = np.stack([X[:, 0] - self.S * np.sin(2 * np.pi * X[:, 1]), X[:, 1]], axis=1)
        return (np.linalg.norm(q - self.ck, axis=1) - self.R0) / (self.h * self.g0)


class QuadHinge:
    def __init__(self, xg, n, tau, kappa, h):
        self.xg, self.n, self.tau, self.c, self.h = np.asarray(xg), np.asarray(n), np.asarray(tau), kappa * h / 2, h

    def __call__(self, X):
        Y = (X - self.xg) / self.h
        return Y @ self.n + self.c * (Y @ self.tau) ** 2


class ShearedDisc:
    def __init__(self, cx=0.5, cy=0.5, R0=0.25, ratio=1e3, S=0.0):
        self.cx, self.cy, self.R0, self.ratio, self.S = cx, cy, R0, ratio, S

    def at(self, S):
        return ShearedDisc(self.cx, self.cy, self.R0, self.ratio, S)

    def back(self, X):
        return np.stack([X[..., 0] - self.S * np.sin(2 * np.pi * X[..., 1]), X[..., 1]], axis=-1)

    def heavy(self, x, y):
        dx = (x - self.S * np.sin(2 * np.pi * y) - self.cx + 0.5) % 1.0 - 0.5
        dy = (y - self.cy + 0.5) % 1.0 - 0.5
        return dx * dx + dy * dy < self.R0 ** 2

    def _copy(self, x, y):
        """The periodic copy of the centre nearest to the pre-image of (x, y)."""
        xb = x - self.S * math.sin(2 * math.pi * y)
        return (self.cx + round(xb - self.cx), self.cy + round(y - self.cy))

    def curve(self, th, ck):
        y = ck[1] + self.R0 * np.sin(th)
        return np.stack([ck[0] + self.R0 * np.cos(th) + self.S * np.sin(2 * np.pi * y), y], axis=-1)

    def _foot(self, x, y):
        ck = self._copy(x, y)
        P = np.array([x, y])
        d2 = lambda th: np.sum((self.curve(np.atleast_1d(th), ck) - P) ** 2, axis=1)  # noqa: E731
        k = int(np.argmin(d2(TH)))
        a, b = TH[k] - 2 * np.pi / 720, TH[k] + 2 * np.pi / 720
        g = (math.sqrt(5) - 1) / 2
        for _ in range(60):
            c1, c2 = b - g * (b - a), a + g * (b - a)
            if d2(c1)[0] < d2(c2)[0]:
                b = c2
            else:
                a = c1
        th = 0.5 * (a + b)
        return th, ck

    def _geom_at(self, th, ck):
        R0, S = self.R0, self.S
        y = ck[1] + R0 * math.sin(th)
        w = 2 * math.pi
        xp = -R0 * math.sin(th) + S * w * math.cos(w * y) * R0 * math.cos(th)
        yp = R0 * math.cos(th)
        xpp = -R0 * math.cos(th) + S * (-(w ** 2) * math.sin(w * y) * (R0 * math.cos(th)) ** 2
                                        - w * math.cos(w * y) * R0 * math.sin(th))
        ypp = -R0 * math.sin(th)
        sp = math.hypot(xp, yp)
        n = np.array([yp, -xp]) / sp
        kappa = (xp * ypp - yp * xpp) / sp ** 3
        return n, kappa

    def nearest(self, x, y):
        th, ck = self._foot(x, y)
        xg = self.curve(np.array([th]), ck)[0]
        s = math.hypot(x - xg[0], y - xg[1])
        return (s if self.heavy(x, y) == False else -s), None  # noqa: E712

    def frame(self, c, s, h, hinge):
        th, ck = self._foot(*c)
        xg = self.curve(np.array([th]), ck)[0]
        n, kappa = self._geom_at(th, ck)
        tau = np.array([-n[1], n[0]])
        if hinge == "exact":
            q = self.back(xg) - np.asarray(ck)
            J = np.array([[1.0, -2 * np.pi * self.S * math.cos(2 * np.pi * xg[1])], [0.0, 1.0]])
            g0 = float(np.linalg.norm(J.T @ (q / np.linalg.norm(q))))
            return (xg, n, tau, PhiHinge(ck, self.R0, self.S, h, g0))
        if hinge == "quadratic":
            return (xg, n, tau, QuadHinge(xg, n, tau, kappa, h))
        sgn = -1.0 if s > 0 else 1.0
        return (xg, n * sgn, tau, None)

    def _iface(self, x0, y0, x1, y1):
        """The two interface x-curves of the copy relevant to the cell, and the tangency ordinates."""
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        ck = self._copy(xm, ym)
        R0, S = self.R0, self.S

        def side(sg):
            def f(y):
                d = R0 * R0 - (y - ck[1]) ** 2
                return np.where(d >= -1e-12 * R0 * R0, ck[0] + S * np.sin(2 * np.pi * y) + sg * np.sqrt(np.maximum(d, 0.0)), np.nan)
            return f
        return [side(-1.0), side(1.0)], [ck[1] - R0, ck[1] + R0]

    def pieces(self, x0, y0, x1, y1):
        curves, yb = self._iface(x0, y0, x1, y1)
        parts = strips(x0, x1, y0, y1, curves, yb, lambda x, y: self.heavy(x, y).astype(int), tangency=yb)
        return [(P, w, bool(v)) for v, (P, w) in parts.items()]


class _ShearAssembler:
    def __init__(self, N, rec0, rec1, dS):
        self.N, self.rec0, self.rec1, self.dS = N, rec0, rec1, dS

    def row(self, iy):
        N, rec0, rec1, dS = self.N, self.rec0, self.rec1, self.dS
        h = 1.0 / N
        geom = rec1.geom
        trip = {k: ([], [], []) for k in ("G0", "G1", "C", "Cg")}

        def add(k, r, c, v):
            R, Cc = np.meshgrid(r, c, indexing="ij")
            trip[k][0].append(R.ravel())
            trip[k][1].append(Cc.ravel())
            trip[k][2].append(np.asarray(v).ravel())

        for ix in range(N):
            i = iy * N + ix
            x0, y0, x1, y1 = ix * h, iy * h, (ix + 1) * h, (iy + 1) * h
            curves, yb = geom._iface(x0, y0, x1, y1)
            faces = [lambda y, c=(ix + o) * h: c + dS * np.sin(2 * np.pi * y) for o in (-1, 0, 1, 2)]

            def label(x, y):
                jx = np.floor((x - dS * np.sin(2 * np.pi * y)) / h).astype(int) - (ix - 2)   # 0..4
                return 2 * jx + geom.heavy(x, y).astype(int)
            for v, (X, w) in strips(x0, x1, y0, y1, curves + faces, yb, label, tangency=yb).items():
                hv = v % 2 == 1
                jx = v // 2 + ix - 2
                wr = w * (1.0 if hv else 1.0 / geom.ratio)
                j = iy * N + (jx % N)
                Xo = np.stack([X[:, 0] - dS * np.sin(2 * np.pi * X[:, 1]) - (jx - jx % N) * h, X[:, 1]], axis=1)
                i1, V1 = rec1.basis(i, X)
                i0, V0 = rec0.basis(j, Xo)
                add("G1", i1, i1, (V1 * wr[:, None]).T @ V1)
                add("G0", i0, i0, (V0 * wr[:, None]).T @ V0)
                add("C", i1, i0, (V1 * wr[:, None]).T @ V0)
                add("Cg", [i], i0, (wr @ V0)[None, :])
        n2 = N * N
        return {k: (np.concatenate(v[0]) * n2 + np.concatenate(v[1]), np.concatenate(v[2])) for k, v in trip.items()}

    def rows(self, iys):
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


def assemble(N, rec0, rec1, dS, workers=1):
    n2 = N * N
    w = max(workers, 1)
    chunks = [c for c in (list(range(N))[k::w] for k in range(w)) if c]
    out = _parallel(_ShearAssembler(N, rec0, rec1, dS), "rows", chunks, workers)
    mats = []
    for k in ("G0", "G1", "C", "Cg"):
        M = np.zeros(n2 * n2)
        for t in out:
            idx, v = t[k]
            M[idx] += v
        mats.append(M.reshape(n2, n2))
    return tuple(mats)


def averages(N, geom, f):
    h = 1.0 / N
    out = np.zeros(N * N)
    for iy in range(N):
        for ix in range(N):
            num = den = 0.0
            for X, w, hv in geom.pieces(ix * h, iy * h, (ix + 1) * h, (iy + 1) * h):
                r = 1.0 if hv else 1.0 / geom.ratio
                num += r * float(w @ f(X[:, 0], X[:, 1]))
                den += r * float(w.sum())
            out[iy * N + ix] = num / den
    return out


def exact_field(g, q0):
    return lambda x, y: q0(x - g.S * np.sin(2 * np.pi * y), y)


def exactness(N, hinge, S=0.15, ratio=1e3, seed=0, near=2.5, workers=1):
    """Random P0 + max(phi / h, 0) P1 (polynomials in x/h, y/h about the disc's centre copy) against the
    reconstruction near the interface (relative error; with S = 0 the same test as cases.galerkin2d_circle)."""
    g = ShearedDisc(ratio=ratio, S=S)
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
            ck = np.array(g._copy(*c))

            def q(X):
                xi, eta = (X[:, 0] - ck[0]) / h, (X[:, 1] - ck[1]) / h     # about the centre (as in note 89's circle)
                qb = g.back(X) - ck
                d = (np.linalg.norm(qb, axis=1) - g.R0) / h
                p0 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c0, P0_EXP)) / 10.0 ** 4
                p1 = sum(cc * xi ** p * eta ** qq for cc, (p, qq) in zip(c1, P1_EXP)) / 10.0 ** 3
                return p0 + np.maximum(d, 0.0) * p1
            i = iy * N + ix
            idx, V = rec.basis(i, rec.qx[i])
            u = {}
            for dy in range(-9, 10):
                for dx in range(-9, 10):
                    Xc, W, j = rec._cell_points(ix + dx, iy + dy)
                    u[j] = float(W @ q(Xc)) / W.sum()
            ref = q(rec.qx[i])
            err = max(err, float(np.abs(V @ np.array([u[j] for j in idx]) - ref).max()))
            scale = max(scale, float(np.abs(ref).max()))
    return err / scale


def run(N, hinge, transport="galerkin", ratio=1e3, A=1.0, T=0.5, t_end=0.25, cfl=0.37, samples=(0.125,),
        workers=1):
    h = 1.0 / N
    steps = int(math.ceil(t_end / (cfl * h / A)))
    dt = t_end / steps
    S = lambda t: A * T / math.pi * math.sin(math.pi * t / T)  # noqa: E731
    g0 = ShearedDisc(ratio=ratio)
    q0 = field(Circle(ratio=ratio))
    exact = lambda t: averages(N, g0.at(S(t)), exact_field(g0.at(S(t)), q0))  # noqa: E731
    rk = {**RK, "hinge": hinge, "workers": workers}
    rec0 = Reconstruction2D(N, g0, **rk)
    u = exact(0.0)
    u0max = float(np.abs(u).max())
    E0, errs, inf, condm, umax = None, {}, 0.0, 0.0, 0.0
    at = {int(round(t / dt)): t for t in samples}
    one = np.ones(N * N)
    fs = 0.0
    for k in range(steps):
        rec1 = Reconstruction2D(N, g0.at(S((k + 1) * dt)), **rk)
        G0, G1, C, Cg = assemble(N, rec0, rec1, S((k + 1) * dt) - S(k * dt), workers)
        if E0 is None:
            E0 = float(u @ G0 @ u)
        Tm = Cg / rec1.mass[:, None] if transport == "geo" else np.linalg.solve(G1, C)
        inf = max(inf, float(np.abs(Tm).sum(1).max()))
        fs = max(fs, float(np.abs(Tm @ one - 1).max()))
        Mh = np.diag(1 / np.sqrt(rec1.mass))
        ev = np.linalg.eigvalsh(Mh @ G1 @ Mh)
        condm = max(condm, float(ev.max() / ev.min()))
        u = Tm @ u
        umax = max(umax, float(np.abs(u).max()))
        if k + 1 in at:
            errs[at[k + 1]] = float(np.abs(u - exact((k + 1) * dt)).max())
        rec0, Gl = rec1, G1
    errs[t_end] = float(np.abs(u - exact(t_end)).max())
    return {"errors": errs, "energy": float(u @ Gl @ u) / E0, "inf_step": inf, "cond_max": condm,
            "umax_over_run": umax / u0max, "steps": steps, "free_stream_max": fs}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [("exact", exactness, {"N": 32, "hinge": hz}) for hz in ("exact", "tangent", "quadratic")]
    for N in (64, 48, 32, 24):
        for hz in ("exact", "quadratic", "tangent"):
            jobs.append(("run", run, {"N": N, "hinge": hz}))
        jobs.append(("run", run, {"N": N, "hinge": "exact", "transport": "geo"}))
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [(n, kw, ex.submit(fn, **kw, workers=a.workers)) for n, fn, kw in jobs]
        for n, kw, f in futs:
            r = f.result()
            res.append({"test": n, "args": kw, "result": r})
            print(f"{time.time() - t0:7.0f}s", n, kw, r, flush=True)
    json.dump(res, open("docs/figures/galerkin2d_shear.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

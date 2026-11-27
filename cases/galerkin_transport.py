"""P0-4b-3 (docs/notes/80, 81): the user's geometry-consistent Galerkin transport, G^{n+1} u^{n+1} = C u^n, in 1-D with
a discontinuous density (ratio 1000) and an exact interface, local polynomial reconstructions (no CCD yet).

Periodic [0, 1), N cells, a heavy band rho = 1 on (0.213, 0.647) and 0.001 elsewhere, carried by a uniform flow
together with q (exact: rho(x - s), q(x - s), s the displacement; every function takes the displacement, so the
mass map, the density and the interfaces agree for any velocity). The unknowns are the mass-weighted cell averages
u_i = int_{K_i} rho q / m_i.

Reconstructions (``piecewise``):
* False (note 80): on cell i the polynomial of degree p whose mass-weighted averages over the stencil's cells are the
  stencil's u (cubic: cells i-1 .. i+2; quartic: i-2 .. i+2);
* True (note 81, the user's design): the cells within p of an interfacial cell use one piecewise polynomial of the
  interface, P_-(xi) = a0 + sum a_r^- xi^r on the left, P_+ = a0 + sum a_r^+ xi^r on the right (xi = (x - x_G)/h),
  sharing the interface value a0 (continuous velocity, free gradients), fitted to the mass-weighted averages of the
  2p + 1 cells centred on the interfacial cell; the other cells as above;
* "blend": the reconstructions of the stencils centred on the interfacial cell cz and on cz + 1 mixed, cz alone for
  theta <= 1/4 and cz + 1 alone for theta >= 3/4 (theta the interface's place in its cell, both stencils well posed
  there), linearly between, so that the basis is continuous when the interface crosses a cell boundary (each
  component keeps the averages and the constants, hence the mix).
Either way sum_j phi_j = 1 and int_{K_i} rho phi_j = m_i delta_ij, and

    G_ij = int rho^{n+1} phi_i^{n+1} phi_j^{n+1},    C_ij = int rho^n(x) phi_i^{n+1}(x + ds) phi_j^n(x) dx,

integrated exactly (Gauss on the pieces between the cell boundaries before and after the move and the interfaces).

    python -m cases.galerkin_transport [--piecewise]
"""

from __future__ import annotations

import math

import numpy as np

A0, B0, RHO_IN, RHO_OUT = 0.213, 0.647, 1.0, 1e-3
STENCILS = {2: (-1, 0, 1), 3: (-1, 0, 1, 2), 4: (-2, -1, 0, 1, 2), 5: (-2, -1, 0, 1, 2, 3)}


def rho_at(x, s, ratio: float = RHO_IN / RHO_OUT):
    z = np.mod(x - s, 1.0)
    return np.where((z > A0) & (z < B0), RHO_IN, RHO_IN / ratio)


def interfaces(s):
    return [float(np.mod(A0 + s, 1.0)), float(np.mod(B0 + s, 1.0))]


def _pieces(lo, hi, cuts):
    pts = sorted({lo, hi, *[c for c in cuts if lo < c < hi]})
    return list(zip(pts[:-1], pts[1:]))


def _integrate(f, lo, hi, cuts, order):
    """int_lo^hi f(x) dx with f smooth between the cuts (Gauss-Legendre, ``order`` points per piece)."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    s = 0.0
    for a, b in _pieces(lo, hi, cuts):
        x = 0.5 * (b - a) * xg + 0.5 * (a + b)
        s = s + 0.5 * (b - a) * np.tensordot(wg, f(x), axes=(0, 0))
    return s


def _mono_integral(a, b, z, h, r):
    """int_a^b ((x - z)/h)^r dx."""
    return h / (r + 1) * (((b - z) / h) ** (r + 1) - ((a - z) / h) ** (r + 1))


class Reconstruction:
    """The basis at displacement s: on cell i, u_h(x) = sum_k phi_{i,k}(x) u_{stencil_i[k]}."""

    def __init__(self, n: int, p: int, s: float, piecewise: bool = False, ratio: float = RHO_IN / RHO_OUT,
                 alpha: float = 0.0, extra: int = 0):
        self.n, self.p, self.s = n, p, s
        self.alpha = alpha
        self.extra = extra  # "ls": the window reaches `extra` more cells on each side (over-determined everywhere)
        self.sigma_min = np.inf  # "ls": min over the constrained fits of sigma_min(W^1/2 A Z)
        self.h = h = 1.0 / n
        self.ratio = ratio
        self.offsets = STENCILS[p]
        cuts = [z + m for z in interfaces(s) for m in (-1, 0, 1)]
        self.mass = np.array([_integrate(lambda x: rho_at(x, s, ratio), k * h, (k + 1) * h, cuts, 4)
                              for k in range(n)])
        # every cell's basis: a weighted sum of components ("c", stencil, coef) or ("pw", stencil, (zz, W))
        gammas = []
        for z in interfaces(s):
            for m in (-1, 0, 1):
                zz = z + m
                cz = int(math.floor(zz / h))
                gammas.append((zz, cz, zz / h - cz))
        pw = {}

        def piece(zz, c):
            if (zz, c) not in pw:
                pw[(zz, c)] = self._piecewise(zz, c)
            return ("pw", np.arange(c - p, c + p + 1), (zz, pw[(zz, c)]))

        self.parts = []
        if piecewise == "ls":
            self._build_ls(gammas)
            return
        for i in range(n):
            near = None
            if piecewise:
                for g in gammas:
                    if abs(g[1] - i) <= p + 1 and (near is None or abs(g[1] - i) < abs(near[1] - i)):
                        near = g
            if near is None:
                self.parts.append([(1.0, ("c", i + np.array(self.offsets), self._centred(i)))])
                continue
            zz, cz, theta = near
            # "blend": the stencil centred on cz alone for theta <= 1/4, on cz + 1 alone for theta >= 3/4 (both well
            # posed there), a linear mix between; theta -> 1 gives the stencil the next cell starts with (continuous)
            wb = min(max((theta - 0.25) / 0.5, 0.0), 1.0)
            centres = [(1.0, cz)] if piecewise is True else [(1.0 - wb, cz), (wb, cz + 1)]
            comps = []
            for w, c in centres:
                if w == 0.0:
                    continue
                if abs(c - i) <= p:
                    comps.append((w, piece(zz, c)))
                else:
                    comps.append((w, ("c", i + np.array(self.offsets), self._centred(i))))
            self.parts.append(comps)

    @staticmethod
    def eta(t):
        """The smooth step 3t^2 - 2t^3 on [0, 1]."""
        t = min(max(t, 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _build_ls(self, gammas):
        """"ls" (docs/notes/82): the window of the interface at zz (in cell cz, at theta in it) is cz - p .. cz + p + 1,
        the end cells weighted 1 - eta(theta) (left) and eta(theta) (right), the others 1, all times
        (m_j / max m)^alpha; the window at theta = 1 is the next cell's at theta = 0, so the weights are continuous in
        zz. Cell i (in the window) gets its own constrained fit: the shared-interface-value piecewise polynomial whose
        mass-weighted average over cell i is exactly u_i and whose weighted squared misfit over the window is least.
        The cells that enter and leave the set of piecewise cells (the window's end cells) mix this fit with the
        centred single polynomial by the same weight (both keep the cell's average and the constants), so the basis
        is continuous in the interface position."""
        p, n = self.p, self.n
        lam = {}  # cell -> (weight of the piecewise fit, gamma)
        for g in gammas:
            zz, cz, theta = g
            for c in range(cz - p, cz + p + 2):
                w = 1.0 - self.eta(theta) if c == cz - p else self.eta(theta) if c == cz + p + 1 else 1.0
                if w > 0.0 and (c not in lam or w > lam[c][0]):
                    lam[c] = (w, g)
        bycell = {}
        for c, (w, g) in lam.items():
            ci = c % n
            if ci not in bycell or w > bycell[ci][0]:
                bycell[ci] = (w, g, c)
        for i in range(n):
            comps = []
            if i in bycell:
                w, (zz, cz, theta), c = bycell[i]
                W = self._ls_fit(zz, cz, theta, c)
                e = self.extra
                comps.append((w, ("pw", np.arange(cz - p - e, cz + p + 2 + e) + (i - c), (zz + (i - c) * self.h, W))))
                if w < 1.0:
                    comps.append((1.0 - w, ("c", i + np.array(self.offsets), self._centred(i))))
            else:
                comps.append((1.0, ("c", i + np.array(self.offsets), self._centred(i))))
            self.parts.append(comps)

    def _rows(self, zz, cells):
        """Mass-weighted averages of the basis [1, xi^r left, xi^r right] over the (unwrapped) cells, and the
        cells' masses."""
        h, p = self.h, self.p
        rl, rr = float(rho_at(np.array([zz - 1e-12]), self.s, self.ratio)[0]), \
            float(rho_at(np.array([zz + 1e-12]), self.s, self.ratio)[0])
        A = np.zeros((len(cells), 2 * p + 1))
        mass = np.zeros(len(cells))
        for row, c in enumerate(cells):
            a, b = c * h, (c + 1) * h
            la, lb = a, min(b, zz)
            ra, rb = max(a, zz), b
            wl = rl * max(lb - la, 0.0)
            wr = rr * max(rb - ra, 0.0)
            m = wl + wr
            mass[row] = m
            A[row, 0] = 1.0
            if lb > la:
                A[row, 1:p + 1] = [rl * _mono_integral(la, lb, zz, h, r) / m for r in range(1, p + 1)]
            if rb > ra:
                A[row, p + 1:] = [rr * _mono_integral(ra, rb, zz, h, r) / m for r in range(1, p + 1)]
        return A, mass

    def _ls_fit(self, zz, cz, theta, c):
        """W (coefficients = W u_window) of cell c's constrained fit (see _build_ls)."""
        p = self.p
        e = self.extra
        cells = np.arange(cz - p - e, cz + p + 2 + e)
        A, mass = self._rows(zz, cells)
        geo = np.ones(len(cells))
        geo[0] = 1.0 - self.eta(theta)
        geo[-1] = self.eta(theta)
        wts = geo * (mass / mass.max()) ** self.alpha
        k = int(np.where(cells == c)[0][0])
        b = A[k]
        # null space of b^T and a particular solution: a = b (u_c)/(b.b) + Z y
        _, _, Vt = np.linalg.svd(b[None, :])
        Z = Vt[1:].T
        ap = b / (b @ b)  # times u_c
        sw = np.sqrt(wts)
        B = sw[:, None] * (A @ Z)
        sv = np.linalg.svd(B, compute_uv=False)
        self.sigma_min = min(self.sigma_min, float(sv[-1] / sv[0]))
        # y = B^+ sw (u - A ap e_k^T u)
        E = np.eye(len(cells))
        Pk = np.outer(ap, E[k])  # a_p = Pk u
        rhs = sw[:, None] * (E - A @ Pk)
        y = np.linalg.lstsq(B, rhs, rcond=None)[0]
        return Pk + Z @ y  # (2p + 1, window)

    def _centred(self, i):
        h, p, s = self.h, self.p, self.s
        xi0 = (i + 0.5) * h
        cuts = [z + m for z in interfaces(s) for m in (-2, -1, 0, 1, 2)]
        A = np.zeros((len(self.offsets), p + 1))
        for kk, o in enumerate(self.offsets):
            c = i + o
            mom = _integrate(lambda x: rho_at(x, s, self.ratio)[:, None] * ((x[:, None] - xi0) / h) ** np.arange(p + 1),
                             c * h, (c + 1) * h, cuts, p + 2)
            A[kk] = mom / self.mass[c % self.n]
        return np.linalg.solve(A, np.eye(len(self.offsets)))  # (p + 1, stencil)

    def _piecewise(self, zz, c0):
        """The shared-interface-value polynomial of the interface at zz over the cells c0 - p .. c0 + p (the
        interface in one of them): coefficients [a0, a1^- .. ap^-, a1^+ .. ap^+] = W @ u, every cell's
        mass-weighted average matched."""
        h, p = self.h, self.p
        rl, rr = float(rho_at(np.array([zz - 1e-12]), self.s, self.ratio)[0]), \
            float(rho_at(np.array([zz + 1e-12]), self.s, self.ratio)[0])
        A = np.zeros((2 * p + 1, 2 * p + 1))
        for row, c in enumerate(range(c0 - p, c0 + p + 1)):
            a, b = c * h, (c + 1) * h
            la, lb = a, min(b, zz)  # the left phase's part
            ra, rb = max(a, zz), b
            wl = rl * max(lb - la, 0.0)
            wr = rr * max(rb - ra, 0.0)
            m = wl + wr
            A[row, 0] = 1.0
            if lb > la:
                A[row, 1:p + 1] = [rl * _mono_integral(la, lb, zz, h, r) / m for r in range(1, p + 1)]
            if rb > ra:
                A[row, p + 1:] = [rr * _mono_integral(ra, rb, zz, h, r) / m for r in range(1, p + 1)]
        self.last_matrix = A
        return np.linalg.solve(A, np.eye(2 * p + 1))

    def basis(self, x):
        """For points x (unwrapped, within one cell and on one side of the interface): the cells and basis values."""
        h = self.h
        cell = int(math.floor(float(np.mean(x)) / h))
        i = cell % self.n
        shift = (cell - i) * h  # the period the points lie in (cell - i is a multiple of n)
        cols, vals = {}, {}
        for w, (kind, st, data) in self.parts[i]:
            if kind == "c":
                xi = (x - shift - (i + 0.5) * h) / h
                v = (xi[:, None] ** np.arange(self.p + 1)[None, :]) @ data
            else:
                zz, W = data
                xi = (x - shift - zz) / h
                V = np.zeros((x.size, 2 * self.p + 1))
                V[:, 0] = 1.0
                left = xi < 0
                for r in range(1, self.p + 1):
                    V[:, r] = np.where(left, xi ** r, 0.0)
                    V[:, self.p + r] = np.where(left, 0.0, xi ** r)
                v = V @ W
            for k, c in enumerate(np.mod(st, self.n)):
                vals[c] = vals.get(c, 0.0) + w * v[:, k]
        idx = np.array(sorted(vals))
        return idx, np.stack([vals[c] for c in idx], axis=1)


def matrices(n: int, p: int, s: float, ds: float, piecewise: bool = False, ratio: float = RHO_IN / RHO_OUT,
             remap: bool = False, alpha: float = 0.0, extra: int = 0):
    """G^{n+1} and C for the displacement s -> s + ds, and the reconstructions; ``remap``: also the direct geometric
    remap's matrix Cg_ij = int rho^n(x) 1_{K_i}(x + ds) phi_j^n(x) dx (u_g = M^{-1} Cg u^n)."""
    old = Reconstruction(n, p, s, piecewise, ratio, alpha, extra)
    new = Reconstruction(n, p, s + ds, piecewise, ratio, alpha, extra)
    h = 1.0 / n
    order = 2 * p + 2
    xg, wg = np.polynomial.legendre.leggauss(order)
    cut_new = [k * h for k in range(-1, n + 2)] + [z + m for z in interfaces(s + ds) for m in (-1, 0, 1)]
    cut_old = ([k * h for k in range(-1, n + 2)] + [k * h - ds for k in range(-1, n + 3)]
               + [z + m for z in interfaces(s) for m in (-1, 0, 1)])
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    Cg = np.zeros((n, n))
    for i in range(n):
        for a, b in _pieces(i * h, (i + 1) * h, cut_new):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, s + ds, ratio)
            idx, v = new.basis(x)
            G[np.ix_(idx, idx)] += (v * w[:, None]).T @ v
        for a, b in _pieces(i * h, (i + 1) * h, cut_old):
            x = 0.5 * (b - a) * xg + 0.5 * (a + b)
            w = 0.5 * (b - a) * wg * rho_at(x, s, ratio)
            io, vo = old.basis(x)
            inew, vn = new.basis(x + ds)
            C[np.ix_(inew, io)] += (vn * w[:, None]).T @ vo
            if remap:
                target = int(math.floor(float(np.mean(x + ds)) / h)) % n
                Cg[target, io] += w @ vo
    return (G, C, old, new, Cg) if remap else (G, C, old, new)


def profile(z, kind: str = "smooth"):
    """The transported q at the co-moving coordinate z: "smooth" sin 2 pi z; "kink": + 3 (z - A0)(B0 - z) inside the
    band (continuous, its gradient jumps at both interfaces); "phases": exp(z) cos 3z inside, sin 2 pi z outside, made
    continuous at both interfaces by a linear correction inside (different non-polynomial functions per phase)."""
    z = np.mod(z, 1.0)
    q = np.sin(2 * np.pi * z)
    if kind == "kink":
        q = q + np.where((z > A0) & (z < B0), 3.0 * (z - A0) * (B0 - z), 0.0)
    elif kind == "phases":
        f = lambda t: np.exp(t) * np.cos(3 * t)  # noqa: E731
        g = lambda t: np.sin(2 * np.pi * t)  # noqa: E731
        da, db = g(A0) - f(A0), g(B0) - f(B0)
        inner = f(z) + da + (db - da) * (z - A0) / (B0 - A0)
        q = np.where((z > A0) & (z < B0), inner, q)
    elif kind == "grid":  # docs/notes/84: + a wave of 1/16 (four cells at N = 64), the dissipation's probe
        q = q + 0.5 * np.sin(32 * np.pi * z)
    return q


def exact_averages(n: int, s: float, mass, kind: str = "smooth", ratio: float = RHO_IN / RHO_OUT):
    h = 1.0 / n
    cuts = [z + m for z in interfaces(s) for m in (-1, 0, 1)]
    return np.array([_integrate(lambda x: rho_at(x, s, ratio) * profile(x - s, kind), k * h, (k + 1) * h, cuts, 10)
                     for k in range(n)]) / mass


def _sqrtm(A):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1 / np.sqrt(w)) @ V.T


def one_step(n: int, p: int, cfl: float = 0.37, piecewise: bool = False, kind: str = "smooth"):
    ds = cfl / n
    G, C, old, new = matrices(n, p, 0.0, ds, piecewise)
    u0 = exact_averages(n, 0.0, old.mass, kind)
    u1 = np.linalg.solve(G, C @ u0)
    err = float(np.abs(u1 - exact_averages(n, ds, new.mass, kind)).max())
    G0 = matrices(n, p, -ds, ds, piecewise)[0]  # G at displacement 0
    fs = float(np.abs(np.linalg.solve(G, C @ np.ones(n)) - 1.0).max())
    mom = abs(new.mass @ u1 - old.mass @ u0)
    _, G0ih = _sqrtm(G0)
    G1h, _ = _sqrtm(G)
    amp = np.linalg.norm(G1h @ np.linalg.solve(G, C) @ G0ih, 2)
    E0, E1 = u0 @ G0 @ u0, u1 @ G @ u1
    Mh = np.diag(1 / np.sqrt(new.mass))
    ev = np.linalg.eigvalsh(Mh @ G @ Mh)
    return {"error": err, "free_stream": fs, "momentum": float(mom), "amplification": float(amp),
            "dE": float(E1 / E0 - 1), "min_eig_G": float(np.linalg.eigvalsh(G).min()),
            "mass_eig": (float(ev.min()), float(ev.max()))}


def run(n: int, p: int, t_end: float = 0.137, cfl: float = 0.37, kind: str = "smooth", piecewise: bool = False,
        samples=None):
    """Steps of displacement <= cfl/n up to t_end (U = 1); max error at t_end (and at ``samples`` if given)."""
    steps = int(math.ceil(t_end / (cfl / n)))
    ds = t_end / steps
    u = None
    out = {}
    sample_steps = {} if samples is None else {int(round(t / ds)): t for t in samples}
    for k in range(steps):
        G, C, old, new = matrices(n, p, k * ds, ds, piecewise)
        if u is None:
            u = exact_averages(n, 0.0, old.mass, kind)
        u = np.linalg.solve(G, C @ u)
        if k + 1 in sample_steps:
            out[sample_steps[k + 1]] = float(np.abs(u - exact_averages(n, (k + 1) * ds, new.mass, kind)).max())
    err = float(np.abs(u - exact_averages(n, t_end, new.mass, kind)).max())
    return (err, steps) if samples is None else (err, steps, out)


def locality(n: int, p: int, cfl: float = 0.37, piecewise: bool = False):
    """The largest |(G^-1 C)_ij| at each cell distance |i - j| (periodic)."""
    G, C, _, _ = matrices(n, p, 0.0, cfl / n, piecewise)
    T = np.linalg.solve(G, C)
    d = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)])
    return [float(np.abs(T[d == k]).max()) for k in range(n // 2 + 1)]


def exactness(n: int, p: int, ratio: float = RHO_IN / RHO_OUT, seed: int = 0, piecewise=True, alpha: float = 0.0,
              s: float = 0.0, extra: int = 0):
    """P0-2: a piecewise polynomial of degree p (different in the two phases, continuous at both interfaces) is
    reproduced by the piecewise reconstruction: max |u_h - q| at points of every cell."""
    rng = np.random.default_rng(seed)
    cin = rng.normal(size=p + 1)
    cout = rng.normal(size=p + 1)
    qin = lambda z: np.polyval(cin, z - 0.43)  # noqa: E731
    # outside on the unwrapped (B0, A0 + 1): add a linear function so that it meets qin at both interfaces
    base = lambda z: np.polyval(cout, z - 0.93)  # noqa: E731
    da, db = qin(B0) - base(B0), qin(A0) - base(A0 + 1.0)
    qout = lambda z: base(z) + da + (db - da) * (z - B0) / (A0 + 1.0 - B0)  # noqa: E731

    def q(x):
        z = np.mod(x, 1.0)
        zo = np.where(z <= A0, z + 1.0, z)
        return np.where((z > A0) & (z < B0), qin(z), qout(zo))

    rec = Reconstruction(n, p, s, piecewise, ratio, alpha, extra)
    h = 1.0 / n
    cuts = [z + m for z in interfaces(s) for m in (-1, 0, 1)]
    u = np.array([_integrate(lambda x: rho_at(x, s, ratio) * q(x - s), k * h, (k + 1) * h, cuts, 12)
                  for k in range(n)]) / rec.mass
    err = 0.0
    for i in range(n):
        for a, b in _pieces(i * h, (i + 1) * h, cuts):
            x = a + (b - a) * np.linspace(0.05, 0.95, 7)
            idx, v = rec.basis(x)
            err = max(err, float(np.abs(v @ u[idx] - q(x - s)).max()))
    return err, float(np.abs(q(np.linspace(0, 1, 101))).max())


def sweep(p: int, ratio: float, samples: int = 501):
    """P0-3: the interface at theta in its cell (cell width 1, interface cell [0, 1], p full cells on each side),
    heavy on the right (rho = 1, ratio on the left: 1 / ratio): max condition number of the reconstruction matrix,
    min of the Schur denominator sum_s rho_s int L_s / m (the lower bound 1/(p + 1)), max over the interfacial cell of
    the sum of |basis values| (the reconstruction's amplification)."""
    rl, rr = 1.0 / ratio, 1.0
    out = {"cond": 0.0, "schur": np.inf, "amp": 0.0}
    for theta in np.linspace(0.0, 1.0, samples):
        A = np.zeros((2 * p + 1, 2 * p + 1))
        for row, c in enumerate(range(-p, p + 1)):
            a, b = float(c), float(c + 1)
            if c < 0:
                A[row, 0] = 1.0
                A[row, 1:p + 1] = [_mono_integral(a, b, theta, 1.0, r) for r in range(1, p + 1)]
            elif c > 0:
                A[row, 0] = 1.0
                A[row, p + 1:] = [_mono_integral(a, b, theta, 1.0, r) for r in range(1, p + 1)]
            else:
                m = rl * theta + rr * (1.0 - theta)
                A[row, 0] = 1.0
                A[row, 1:p + 1] = [rl * _mono_integral(0.0, theta, theta, 1.0, r) / m for r in range(1, p + 1)]
                A[row, p + 1:] = [rr * _mono_integral(theta, 1.0, theta, 1.0, r) / m for r in range(1, p + 1)]
        out["cond"] = max(out["cond"], float(np.linalg.cond(A)))
        W = np.linalg.inv(A)
        # L_s: a0 = 1 and zero averages on the side's full cells
        den = 0.0
        m = rl * theta + rr * (1.0 - theta)
        for side, (lo, hi, rho_s, cells) in enumerate(((0.0, theta, rl, range(-p, 0)), (theta, 1.0, rr, range(1, p + 1)))):
            B = np.array([[_mono_integral(c, c + 1, theta, 1.0, r) for r in range(1, p + 1)] for c in cells])
            coef = np.linalg.solve(B, -np.ones(p))  # 1 + sum coef_r xi^r has zero averages there
            den += rho_s * ((hi - lo) + sum(cf * _mono_integral(lo, hi, theta, 1.0, r)
                                            for r, cf in zip(range(1, p + 1), coef)))
        out["schur"] = min(out["schur"], den / m)
        for xx in np.linspace(0.0, 1.0, 41):
            xi = xx - theta
            V = np.zeros(2 * p + 1)
            V[0] = 1.0
            for r in range(1, p + 1):
                V[(r if xi < 0 else p + r)] = xi ** r
            out["amp"] = max(out["amp"], float(np.abs(V @ W).sum()))
    return out


def switching(n: int, p: int, eps: float, across: bool = True, piecewise: bool = True, kind: str = "kink"):
    """P0-4: the interface at A0 + s moves by 2 eps h, from k h - eps h to k h + eps h (``across``: over a cell
    boundary, where the stencils switch) or from (k + 1/2) h - eps h to (k + 1/2) h + eps h (within a cell).
    Returns ||G^-1 C - I||_2, ||u^1 - u^0||_inf, (E^1 - E^0)/E^0 (q ``kind``)."""
    h = 1.0 / n
    k = int(round(A0 / h)) + 1
    start = (k * h if across else (k + 0.5) * h) - eps * h
    s0, ds = start - A0, 2 * eps * h
    G, C, old, new = matrices(n, p, s0, ds, piecewise)
    T = np.linalg.solve(G, C)
    u0 = exact_averages(n, s0, old.mass, kind)
    u1 = T @ u0
    G0 = matrices(n, p, s0 - ds, ds, piecewise)[0]
    return (float(np.linalg.norm(T - np.eye(n), 2)), float(np.abs(u1 - u0).max()),
            float((u1 @ G @ u1) / (u0 @ G0 @ u0) - 1))


def local_correction(n: int, p: int, piecewise=True, kind: str = "smooth", cfl: float = 0.37):
    """P0-5: the Galerkin step against the direct geometric remap of the same reconstruction, u_g = M^-1 Cg u^n:
    delta p_i = m_i (u^{n+1}_i - u_g,i) (sums to zero). Returns max |delta u| in the cells within 2 of an interface
    and elsewhere, |sum delta p|, and the errors of u_g and u^{n+1} against the exact averages."""
    ds = cfl / n
    G, C, old, new, Cg = matrices(n, p, 0.0, ds, piecewise, remap=True)
    u0 = exact_averages(n, 0.0, old.mass, kind)
    u1 = np.linalg.solve(G, C @ u0)
    ug = (Cg @ u0) / new.mass
    du = u1 - ug
    h = 1.0 / n
    xc = (np.arange(n) + 0.5) * h
    d = np.min([np.minimum(np.abs(xc - z), 1 - np.abs(xc - z)) for z in interfaces(ds)], axis=0) / h
    ex = exact_averages(n, ds, new.mass, kind)
    return {"near": float(np.abs(du[d <= 2.5]).max()), "away": float(np.abs(du[d > 2.5]).max()),
            "sum": float(abs(np.sum(new.mass * du))), "err_remap": float(np.abs(ug - ex).max()),
            "err_galerkin": float(np.abs(u1 - ex).max())}


def amplification(n: int, p: int, piecewise=True, steps: int = 40, cfl: float = 0.37):
    """P0-6: per-step ||T||_inf and ||T||_2 (T = G^-1 C), and those of the product T_{k-1} ... T_0 over ``steps``
    steps (the interface crosses about steps * cfl cells)."""
    ds = cfl / n
    P = np.eye(n)
    one = {"inf": 0.0, "two": 0.0}
    prod = {"inf": 0.0, "two": 0.0}
    for k in range(steps):
        G, C, _, _ = matrices(n, p, k * ds, ds, piecewise)
        T = np.linalg.solve(G, C)
        one["inf"] = max(one["inf"], float(np.abs(T).sum(1).max()))
        one["two"] = max(one["two"], float(np.linalg.norm(T, 2)))
        P = T @ P
        prod["inf"] = max(prod["inf"], float(np.abs(P).sum(1).max()))
        prod["two"] = max(prod["two"], float(np.linalg.norm(P, 2)))
    return one, prod


def main(piecewise: bool = False) -> None:
    ns = (16, 32, 64, 128)
    tag = "piecewise" if piecewise else "single polynomial"
    print(f"-- {tag}: one step, CFL 0.37: max error of the mass-weighted cell averages; structure")
    for p in (3, 4):
        rs = [one_step(n, p, piecewise=piecewise) for n in ns]
        print(f"   degree {p}: " + " ".join(f"{r['error']:.2e}" for r in rs) + "; free stream "
              f"{max(r['free_stream'] for r in rs):.1e}, momentum {max(r['momentum'] for r in rs):.1e}, amplification "
              f"{max(r['amplification'] for r in rs):.12f}, dE/E {min(r['dE'] for r in rs):.1e} .. "
              f"{max(r['dE'] for r in rs):.1e}, eig(M^-1/2 G M^-1/2) in [{min(r['mass_eig'][0] for r in rs):.6f}, "
              f"{max(r['mass_eig'][1] for r in rs):.3f}]", flush=True)
    for kind in ("smooth", "kink", "phases"):
        print(f"-- {tag}: q {kind}, CFL 0.37, max error at t = 0.05, 0.1, 0.137")
        for p in (3, 4):
            res = [run(n, p, kind=kind, piecewise=piecewise, samples=(0.05, 0.1)) for n in ns + (256,)]
            for t in (0.05, 0.1, 0.137):
                e = [r[2][t] if t in r[2] else r[0] for r in res] if t != 0.137 else [r[0] for r in res]
                print(f"   degree {p} t={t:5.3f}: " + " ".join(f"{x:.2e}" for x in e) + "  orders "
                      + " ".join(f"{math.log2(e[k] / e[k + 1]):.2f}" for k in range(len(e) - 1)), flush=True)


if __name__ == "__main__":
    import sys

    main(piecewise="--piecewise" in sys.argv)

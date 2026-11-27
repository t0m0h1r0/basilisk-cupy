"""docs/notes/95 (P1-a, P1-b): two-phase PCCD reconstructions without a domain-wide jump function.

P1-a "local" (the user's section 4): the CCD system A z = B q + b_G, with b_G = A z_J - B q_J only on the rows whose
stencil straddles an interface (J = H(d) T(d), T the degree-4 jump polynomial; on any row whose stencil stays on one
side, J is a polynomial there and the CCD reproduces it: b_G = 0). The interface cell's polynomial is
H_m(z, q_m) + J - H_m(z_J, q_J,m) (H the mean-preserving Hermite). No cut-off function. Mixed cells: the plain
average from the mass-weighted one (note 92); jumps estimated jointly (note 94 D) or given (Oracle).

P1-b "split" (the user's section 5): PCCD on each phase's run of pure cells separately (interior faces: the CCD rows;
the two end faces: the explicit 5-cell one-sided values and gradients, exact for quartics); the mixed cell holds the
neighbouring pure cells' polynomials extended, the light one shifted by a constant to the heavy interface value; then
the shift u_h = ubar_i + u~ - (mass-weighted mean of u~ over K_i) in every cell. Light data never enter the heavy
reconstruction, and G = M + R~* (I - P_c) R~.

    (used by cases.pccd_gram and cases.pccd_floquet)
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from cases.pccd import BUB, H_G0, H_G1, H_V0, H_V1, MEAN, WG, XG, _mono, ccd_matrices
from cases.pccd_density import pieces
from cases.pccd_jump import estimate_terms_joint, jump_terms


def hb(v0, g0, v1, g1, q):
    """s-coefficients of the mean-preserving Hermite (cubic Hermite + bubble)."""
    c = v0 * H_V0 + g0 * H_G0 + v1 * H_V1 + g1 * H_G1
    return c + (q - c @ MEAN) * BUB / (BUB @ MEAN)


@lru_cache(maxsize=8)
def _ccd(N):
    A, B = ccd_matrices(N)
    return A, B, np.linalg.inv(A)


def _poly(T, d):
    return sum(c * d ** k for k, c in enumerate(T))


def _dpoly(T, d):
    return sum(k * c * d ** (k - 1) for k, c in enumerate(T) if k > 0)


def _int_poly(T, d0, d1):
    return sum(c * (d1 ** (k + 1) - d0 ** (k + 1)) / (k + 1) for k, c in enumerate(T))


# --------------------------------------------------------------------------------------------- P1-a
class Local:
    def __init__(self, N, ratio, a, b, ubar, terms=None):
        self.N, self.ratio, self.a, self.b = N, ratio, a, b
        h = self.h = 1.0 / N
        P = self.P = pieces(N, a, b)
        if terms is None:
            terms = estimate_terms_joint(ubar, N, a, b, P, ratio)
        self.terms = [(xg, list(T)) for xg, T, *_ in terms]
        A, B, Ainv = _ccd(N)
        mixed = [i for i in range(N) if len(P[i]) > 1]
        bG = np.zeros(2 * N)
        self.cellJ = {}                       # interface cell -> (xg, T, zJ at its faces, qJ)
        for xg, T in self.terms:
            m = int(math.floor(xg / h))
            zJ = np.zeros(2 * N)
            qJ = np.zeros(N)
            for f in range(m - 1, m + 3):
                d = f * h - xg
                if d > 0:
                    zJ[2 * (f % N)] = _poly(T, d)
                    zJ[2 * (f % N) + 1] = h * _dpoly(T, d)
            for c in range(m - 1, m + 2):
                d0, d1 = c * h - xg, (c + 1) * h - xg
                lo = max(d0, 0.0)
                if d1 > lo:
                    qJ[c % N] = _int_poly(T, lo, d1) / h
            r = A @ zJ - B @ qJ
            for row in (2 * (m % N), 2 * (m % N) + 1, 2 * ((m + 1) % N), 2 * ((m + 1) % N) + 1):
                bG[row] += r[row]
            self.cellJ[m % N] = (xg, T, zJ[2 * (m % N)], zJ[2 * (m % N) + 1], zJ[2 * ((m + 1) % N)],
                                 zJ[2 * ((m + 1) % N) + 1], qJ[m % N])
        self.bG, self.Ainv, self.B = bG, Ainv, B
        # mixed cells: plain averages from the mass-weighted ones (linear: superposition over the unknowns)
        q = ubar.astype(float).copy()
        if mixed:
            def resid(qM):
                qq = q.copy()
                qq[mixed] = qM
                self._set(qq)
                return np.array([self._wmean(i) for i in mixed]) - ubar[mixed]
            r0 = resid(np.zeros(len(mixed)))
            Smat = np.stack([resid(np.eye(len(mixed))[k]) - r0 for k in range(len(mixed))], axis=1)
            q[mixed] = np.linalg.solve(Smat, -r0)
        self._set(q)

    def _set(self, q):
        z = self.Ainv @ (self.B @ q + self.bG)
        v, g = z[0::2], z[1::2]
        self.coef = np.array([hb(v[i], g[i], v[(i + 1) % self.N], g[(i + 1) % self.N], q[i]) for i in range(self.N)])
        for m, (xg, T, v0, g0, v1, g1, qj) in self.cellJ.items():
            self.coef[m] = self.coef[m] - hb(v0, g0, v1, g1, qj)

    def _cellval(self, i, s):
        x = (i + s) * self.h
        out = _mono(s) @ self.coef[i]
        if i in self.cellJ:
            xg, T = self.cellJ[i][0], self.cellJ[i][1]
            d = x - xg
            out = out + np.where(d > 0, _poly(T, d), 0.0)
        return out

    def _wmean(self, i):
        num = den = 0.0
        for s0, s1, hv in self.P[i]:
            r = 1.0 if hv else 1.0 / self.ratio
            s = s0 + (s1 - s0) * XG
            num += r * (s1 - s0) * (WG @ self._cellval(i, s))
            den += r * (s1 - s0)
        return num / den

    def __call__(self, x):
        cell = np.floor(x / self.h).astype(int)
        s = x / self.h - cell
        out = np.empty_like(x)
        for i in np.unique(cell):
            k = cell == i
            out[k] = self._cellval(int(i) % self.N, s[k])
        return out


# --------------------------------------------------------------------------------------------- P1-b
@lru_cache(maxsize=1)
def end_closure():
    """(cv, cg): v(0) = cv . q, h u'(0) = cg . q for the quartic whose averages over cells [k, k+1] (k = 0..4) are q_k."""
    M = np.array([[((k + 1) ** (p + 1) - k ** (p + 1)) / (p + 1) for p in range(5)] for k in range(5)])
    Mi = np.linalg.inv(M)
    return Mi[0], Mi[1]


def explicit_closure():
    """The default end closure as rows: (row_v, c_v, row_g, c_g), row = coefficients of (v0, g0, v1, g1)."""
    cv, cg = end_closure()
    pad = lambda c: np.concatenate([c, [0.0]])  # noqa: E731
    return (np.array([1.0, 0.0, 0.0, 0.0]), pad(cv), np.array([0.0, 1.0, 0.0, 0.0]), pad(cg))


class Split:
    def __init__(self, N, ratio, a, b, ubar, closure=None):
        self.N, self.ratio, self.a, self.b, self.h = N, ratio, a, b, 1.0 / N
        self.closure = closure or explicit_closure()
        P = self.P = pieces(N, a, b)
        self.mixed = [i for i in range(N) if len(P[i]) > 1]
        heavy = [i for i in range(N) if len(P[i]) == 1 and P[i][0][2]]
        light = [i for i in range(N) if len(P[i]) == 1 and not P[i][0][2]]
        self.coef = np.zeros((N, 5))
        self.phase = {}
        for cells in (self._runs(heavy), self._runs(light)):
            for run in cells:
                self._solve_run(run, ubar)
                for i in run:
                    self.phase[i % N] = run
        # mixed cells: both neighbours' polynomials extended; the light one shifted to the heavy interface value
        self.mix = {}
        for m in self.mixed:
            left, right = (m - 1) % N, (m + 1) % N
            sides = []
            for s0, s1, hv in P[m]:
                nb = left if s0 == 0.0 else right
                sides.append((s0, s1, hv, nb, -1.0 if nb == left else 1.0))
            xg_s = [s1 for s0, s1, hv in P[m]][:-1][0]          # the interface position in the cell (s)
            ext = {}
            for s0, s1, hv, nb, off in sides:
                ext[hv] = (nb, off)
            hv_nb, hv_off = ext[True]
            lt_nb, lt_off = ext[False]
            uH = _mono(np.array([xg_s - hv_off]))[0] @ self.coef[hv_nb]
            uL = _mono(np.array([xg_s - lt_off]))[0] @ self.coef[lt_nb]
            self.mix[m] = (ext, xg_s, uH - uL)
        # the shift: every cell's mass-weighted mean -> ubar (pure cells already hold it)
        self.shift = np.zeros(N)
        for i in range(N):
            self.shift[i] = ubar[i] - self._wmean(i)

    def _runs(self, cells):
        """Maximal runs of consecutive (periodic) cells, as lists of unwrapped indices."""
        N = self.N
        cs = set(cells)
        runs = []
        seen = set()
        for c in sorted(cells):
            if c in seen:
                continue
            start = c
            while (start - 1) % N in cs and (start - 1) % N != c:
                start -= 1
            run = []
            k = start
            while k % N in cs and k % N not in seen:
                run.append(k)
                seen.add(k % N)
                k += 1
            runs.append(run)
        return runs

    def _solve_run(self, run, ubar):
        N = self.N
        n = len(run)
        if n < 5:
            raise ValueError("a phase run shorter than 5 cells")
        q = ubar[np.array(run) % N]
        nf = n + 1
        A = np.zeros((2 * nf, 2 * nf))
        rhs = np.zeros(2 * nf)
        rv, cv, rg, cg = self.closure
        k = cv.size
        # left end face 0 (cells 0..k-1): rows over (v0, g0, v1, g1); right end face n mirrored: v even, g odd
        A[0, 0:4], rhs[0] = rv, cv @ q[:k]
        A[1, 0:4], rhs[1] = rg, cg @ q[:k]
        mir = np.array([1.0, -1.0, 1.0, -1.0])           # (v_n, g_n, v_{n-1}, g_{n-1}) <- (v0, -g0, v1, -g1)
        idx = [2 * n, 2 * n + 1, 2 * (n - 1), 2 * (n - 1) + 1]
        A[2 * n, idx], rhs[2 * n] = rv * mir, cv @ q[::-1][:k]
        A[2 * n + 1, idx], rhs[2 * n + 1] = -rg * mir, -(cg @ q[::-1][:k])
        for f in range(1, n):
            r = 2 * f
            A[r, 2 * f] += 1.0
            A[r, 2 * (f - 1)] += 7 / 16
            A[r, 2 * (f + 1)] += 7 / 16
            A[r, 2 * (f + 1) + 1] += -1 / 16
            A[r, 2 * (f - 1) + 1] += 1 / 16
            rhs[r] = 15 / 16 * (q[f - 1] + q[f])
            r += 1
            A[r, 2 * f + 1] += 1.0
            A[r, 2 * (f - 1) + 1] += -1 / 8
            A[r, 2 * (f + 1) + 1] += -1 / 8
            A[r, 2 * (f + 1)] += 9 / 8
            A[r, 2 * (f - 1)] += -9 / 8
            rhs[r] = 3 * (q[f] - q[f - 1])
        z = np.linalg.solve(A, rhs)
        v, g = z[0::2], z[1::2]
        for k, c in enumerate(run):
            self.coef[c % N] = hb(v[k], g[k], v[k + 1], g[k + 1], q[k])

    def _raw(self, i, s):
        """u~ in cell i at s (before the shift)."""
        if i not in self.mix:
            return _mono(s) @ self.coef[i]
        ext, xg_s, jump = self.mix[i]
        out = np.empty_like(s)
        heavy_side_left = ext[True][1] < 0
        left = s < xg_s
        for hv in (True, False):
            nb, off = ext[hv]
            k = left if (hv == heavy_side_left) else ~left
            val = _mono(s[k] - off) @ self.coef[nb]
            out[k] = val if hv else val + jump
        return out

    def _wmean(self, i):
        num = den = 0.0
        for s0, s1, hv in self.P[i]:
            r = 1.0 if hv else 1.0 / self.ratio
            s = s0 + (s1 - s0) * XG
            num += r * (s1 - s0) * (WG @ self._raw(i, s))
            den += r * (s1 - s0)
        return num / den

    def __call__(self, x):
        cell = np.floor(x / self.h).astype(int)
        s = x / self.h - cell
        out = np.empty_like(x)
        for i in np.unique(cell):
            k = cell == i
            ii = int(i) % self.N
            out[k] = self._raw(ii, s[k]) + self.shift[ii]
        return out


def basis_functions(N, ratio, a, b, mode):
    cls = Local if mode == "local" else Split
    fs = []
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        fs.append(cls(N, ratio, a, b, e))
    return lambda j, x: fs[j](x)


def oracle_local(N, ratio, a, b, ubar):
    return Local(N, ratio, a, b, ubar, terms=jump_terms(a, b))

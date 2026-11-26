"""Phase-wise CCD face gradient for the pressure jump with a density ratio (docs/notes/55, 61).

FCCD's face gradient is a global compact operator: across an interface where the pressure jumps and every
derivative jumps in proportion to the density ratio, it does not converge (note 55). :class:`PhaseCCDFaces`
keeps FCCD everywhere except that the CCD line systems are closed phase by phase:

* every grid line is split at its cut faces (``H = [psi > 1/2]`` differs across the face) into segments; on a
  segment of at least 6 cells the CCD second derivative ``q`` is solved with the extrapolation closure at the cut
  ends (no datum) and the field's own wall closure at walls, and the inner faces take FCCD's gradient
  ``(u_i - u_{i-1})/h - (h/24)(q_i - q_{i-1})``; shorter segments keep the two-point difference with ``q`` from
  second differences;
* on a cut face the two-point GFM difference gets the one-sided Taylor terms of both phases up to the third
  derivative (``[p] = j`` and a continuous flux at the crossing, ``theta`` the crossing's fraction from the lower
  cell), and is moved from the crossing to the face centre in the phase that holds the centre:

      g = (p_R - p_L)/h + (-(bh)^2 q_R(G)/2 - (bh)^3 c_R/6 + (ah)^2 q_L(G)/2 - (ah)^3 c_L/6)/h
          + ((1/2 - theta) h q_s(G) + ((1/2 - theta) h)^2 c_s/2) / (rho_s alpha_f),      a = theta, b = 1 - theta,

  with ``c`` the one-sided third derivative (difference of ``q`` within the phase) and ``q(G)`` moved to the
  crossing. The jump itself, ``alpha_f B(j)``, stays in the face acceleration (TwoPhaseCLS), whose crossing value
  should be interpolated with :func:`hermite_at_crossings` (linear interpolation leaves an error proportional to
  the density ratio, note 55).

Everything else (divergence, face values, face-to-cell) is FCCD's. The lines are processed on the host.
Uniform 2-D grids.
"""

from __future__ import annotations

import numpy as np

from ..backend import array_module, to_numpy
from .ccd import ccd_matrices

MIN_SEGMENT = 6


def _kind(bc, axis: int, side: int) -> str:
    from ..boundary import Dirichlet

    c = bc.get(axis, side)
    if c.periodic:
        return "periodic"
    return "dirichlet" if isinstance(c, Dirichlet) else "neumann"


class PhaseCCDFaces:
    name = "phase-ccd"

    def __init__(self, fccd, transfer: str = "fccd", taylor: bool = True) -> None:
        """``transfer``: the cell <-> face transfers (``interp_f``, ``face_to_cell``) of FCCD (``"fccd"``) or of fd2
        (``"fd2"``, the two-point means; the cell-centred correction g jumps with the density at the interface,
        docs/notes/67); the gradient is the phase-wise CCD's, the divergence FCCD's either way. ``taylor``: the
        one-sided Taylor terms and the shift to the face centre on the cut faces (3rd order on a static manufactured
        solution, notes 55/61); without them the cut faces take the GFM difference alone, which removes an O(1)
        damping of moving interfaces at a density ratio (docs/notes/67)."""
        self.fccd = fccd
        if transfer not in ("fccd", "fd2"):
            raise ValueError("transfer is 'fccd' or 'fd2'")
        self.transfer = transfer
        self.taylor = bool(taylor)
        if transfer == "fd2":
            from .fd2 import FD2FaceComplex  # noqa: PLC0415

            fd2 = FD2FaceComplex()
            self.interp_f, self.face_to_cell = fd2.interp_f, fd2.face_to_cell
        self.cuts = self.thetas = self.alphas = self.rho_lower = None
        self.rho_pair = (1.0, 1.0)

    def update(self, tp, alpha) -> None:
        """Cut faces, crossing fractions, face coefficients and the lower-side density from ``TwoPhaseCLS``."""
        self.cuts, self.thetas, self.rho_lower = [], [], []
        for ax in range(2):
            hl, hr, cut, theta = tp._cut(ax)
            self.cuts.append(to_numpy(cut))
            self.thetas.append(to_numpy(theta))
            self.rho_lower.append(np.where(to_numpy(hl), tp.rho1, tp.rho2))
        self.alphas = [to_numpy(alpha[ax]) for ax in range(2)]
        self.rho_pair = (tp.rho1, tp.rho2)
        self._plans = {}

    def _plan(self, axis: int, n: int, m: int, periodic: bool, wall):
        """Segments of every line grouped by (length, closures), and the cut faces, for the current interface."""
        key = (axis, n, m, periodic, wall)
        cache = getattr(self, "_plans", {})
        if key in cache:
            return cache[key]
        cut = np.moveaxis(self.cuts[axis], axis, 0)
        groups: dict = {}
        cut_faces = []  # (face s0, line j)
        lines = []
        for j in range(m):
            c = cut[:n + (0 if periodic else 1), j]
            starts = np.nonzero(c[: n] if periodic else c[1:n])[0]
            if starts.size == 0:
                continue
            if not periodic:
                starts = starts + 1
            lines.append(j)
            bounds = list(starts)
            if periodic:
                segs = [(s0, (bounds[(k + 1) % len(bounds)] - s0) % n or n, "extrapolate", "extrapolate")
                        for k, s0 in enumerate(bounds)]
            else:
                edges = [0] + bounds + [n]
                segs = [(edges[k], edges[k + 1] - edges[k], wall[0] if edges[k] == 0 else "extrapolate",
                         wall[1] if edges[k + 1] == n else "extrapolate") for k in range(len(edges) - 1)]
            for s0, length, klo, khi in segs:
                groups.setdefault((length, klo, khi), []).append((s0, j))
            cut_faces += [(s0, j) for s0 in bounds]
        plan = (groups, np.array(cut_faces, dtype=int).reshape(-1, 2), np.array(lines, dtype=int))
        cache[key] = plan
        self._plans = cache
        return plan

    def grad_f(self, a, axis: int):
        """Batched form of :meth:`_grad_f_loop` (segments of equal length and closures in one product)."""
        if self.cuts is None:
            return self.fccd.grad_f(a, axis)
        xp = array_module(a.interior)
        base = to_numpy(self.fccd.grad_f(a, axis))
        p = np.moveaxis(to_numpy(a.interior), axis, 0)
        out = np.moveaxis(base.copy(), axis, 0)
        n, m = p.shape
        h = float(a.grid.h[axis])
        periodic = a.bc.is_periodic(axis)
        wall = (_kind(a.bc, axis, 0), _kind(a.bc, axis, 1))
        groups, cf, lines = self._plan(axis, n, m, periodic, wall)
        if lines.size == 0:
            return xp.asarray(np.moveaxis(out, 0, axis))
        q = np.full((n, m), np.nan)
        for (length, klo, khi), segs in groups.items():
            s0 = np.array([sj[0] for sj in segs])
            jj = np.array([sj[1] for sj in segs])
            idx = (s0[None, :] + np.arange(length)[:, None]) % n          # (length, count)
            seg = p[idx, jj[None, :]]
            if length >= MIN_SEGMENT:
                _, D2, *_ = ccd_matrices(length, h, klo, khi)
                qs = D2 @ seg
                out[idx[1:], jj[None, :]] = (seg[1:] - seg[:-1]) / h - (h / 24.0) * (qs[1:] - qs[:-1])
            else:
                out[idx[1:], jj[None, :]] = (seg[1:] - seg[:-1]) / h
                if length < 3:
                    continue
                qs = np.empty_like(seg)
                qs[1:-1] = (seg[2:] - 2 * seg[1:-1] + seg[:-2]) / h ** 2
                qs[0], qs[-1] = qs[1], qs[-2]
            q[idx, jj[None, :]] = qs
        s0, jj = cf[:, 0], cf[:, 1]
        L, R = (s0 - 1) % n, s0 % n
        th = np.moveaxis(self.thetas[axis], axis, 0)[s0, jj]
        af = np.moveaxis(self.alphas[axis], axis, 0)[s0, jj]
        rl = np.moveaxis(self.rho_lower[axis], axis, 0)[s0, jj]
        g = (p[R, jj] - p[L, jj]) / h
        if not self.taylor:
            out[s0, jj] = g
            if periodic:
                out[n, :] = out[0, :]
            return xp.asarray(np.moveaxis(out, 0, axis))
        qL = np.nan_to_num(q[L, jj], nan=0.0)
        qR = np.nan_to_num(q[R, jj], nan=0.0)
        Lm, Rp = (L - 1) % n, (R + 1) % n
        inside = np.ones_like(s0, dtype=bool) if periodic else (L - 1 >= 0) & (R + 1 < n)
        ok = inside & np.isfinite(q[Lm, jj]) & np.isfinite(q[Rp, jj])
        c3L = np.where(ok, (qL - np.nan_to_num(q[Lm, jj])) / h, 0.0)
        c3R = np.where(ok, (np.nan_to_num(q[Rp, jj]) - qR) / h, 0.0)
        bL = qL + th * h * c3L
        bR = qR - (1 - th) * h * c3R
        aa, bb = th * h, (1 - th) * h
        g += (-bb * bb * bR / 2 - bb ** 3 * c3R / 6 + aa * aa * bL / 2 - aa ** 3 * c3L / 6) / h
        r1, r2 = self.rho_pair
        rho_side = np.where(th > 0.5, rl, np.where(rl == r1, r2, r1))
        b_side = np.where(th > 0.5, bL, bR)
        c_side = np.where(th > 0.5, c3L, c3R)
        d = (0.5 - th) * h
        g += (d * b_side + d * d * c_side / 2) / (rho_side * af)
        out[s0, jj] = g
        if periodic:
            out[n, :] = out[0, :]
        return xp.asarray(np.moveaxis(out, 0, axis))

    def _grad_f_loop(self, a, axis: int):
        """Reference implementation, line by line (kept for the tests)."""
        if self.cuts is None:
            return self.fccd.grad_f(a, axis)
        xp = array_module(a.interior)
        base = to_numpy(self.fccd.grad_f(a, axis))  # wall faces and segment-free lines
        p = np.moveaxis(to_numpy(a.interior), axis, 0)
        cut = np.moveaxis(self.cuts[axis], axis, 0)
        th = np.moveaxis(self.thetas[axis], axis, 0)
        af = np.moveaxis(self.alphas[axis], axis, 0)
        rl = np.moveaxis(self.rho_lower[axis], axis, 0)
        out = np.moveaxis(base.copy(), axis, 0)
        n, m = p.shape
        h = float(a.grid.h[axis])
        periodic = a.bc.is_periodic(axis)
        wall = (_kind(a.bc, axis, 0), _kind(a.bc, axis, 1))
        for j in range(m):
            c = cut[:n + (0 if periodic else 1), j]
            starts = np.nonzero(c[: n] if periodic else c[1:n])[0]
            if starts.size == 0:
                continue  # FCCD's own gradient (no interface on the line)
            if not periodic:
                starts = starts + 1
            line = p[:, j]
            g = out[:, j]
            q = np.full(n, np.nan)
            # segments: [s_k, s_{k+1}) between cut faces (face s is between cells s-1 and s)
            bounds = list(starts)
            if periodic:
                segs = [(s0, (bounds[(k + 1) % len(bounds)] - s0) % n or n, "extrapolate", "extrapolate")
                        for k, s0 in enumerate(bounds)]
            else:
                edges = [0] + bounds + [n]
                segs = []
                for k in range(len(edges) - 1):
                    s0, s1 = edges[k], edges[k + 1]
                    segs.append((s0, s1 - s0, wall[0] if s0 == 0 else "extrapolate",
                                 wall[1] if s1 == n else "extrapolate"))
            for s0, length, klo, khi in segs:
                idx = (s0 + np.arange(length)) % n
                seg = line[idx]
                if length >= MIN_SEGMENT:
                    # wall data: the homogeneous condition of the field (the pressure increment: Neumann 0)
                    _, D2, *_ = ccd_matrices(length, h, klo, khi)
                    qs = D2 @ seg
                    q[idx] = qs
                    g[idx[1:]] = (seg[1:] - seg[:-1]) / h - (h / 24.0) * (qs[1:] - qs[:-1])
                else:
                    g[idx[1:]] = (seg[1:] - seg[:-1]) / h
                    if length >= 3:
                        d2 = np.empty(length)
                        d2[1:-1] = (seg[2:] - 2 * seg[1:-1] + seg[:-2]) / h ** 2
                        d2[0], d2[-1] = d2[1], d2[-2]
                        q[idx] = d2
            for s0 in bounds:
                L, R = (s0 - 1) % n, s0 % n
                tt = float(th[s0, j])
                gg = (line[R] - line[L]) / h
                qL = q[L] if np.isfinite(q[L]) else 0.0
                qR = q[R] if np.isfinite(q[R]) else 0.0
                Lm, Rp = (L - 1) % n, (R + 1) % n
                inside = periodic or (L - 1 >= 0 and R + 1 < n)
                c3L = c3R = 0.0
                if inside and np.isfinite(q[Lm]) and np.isfinite(q[Rp]):
                    c3L = (qL - q[Lm]) / h
                    c3R = (q[Rp] - qR) / h
                bL = qL + tt * h * c3L
                bR = qR - (1 - tt) * h * c3R
                aa, bb = tt * h, (1 - tt) * h
                gg += (-bb * bb * bR / 2 - bb ** 3 * c3R / 6 + aa * aa * bL / 2 - aa ** 3 * c3L / 6) / h
                rho_lo = float(rl[s0, j])
                rho_side = rho_lo if tt > 0.5 else (self.rho_pair[1] if rho_lo == self.rho_pair[0] else self.rho_pair[0])
                b_side, c_side = (bL, c3L) if tt > 0.5 else (bR, c3R)
                d = (0.5 - tt) * h
                gg += (d * b_side + d * d * c_side / 2) / (rho_side * float(af[s0, j]))
                g[s0] = gg
            if periodic:
                g[n] = g[0]
            out[:, j] = g
        return xp.asarray(np.moveaxis(out, 0, axis))

    def __getattr__(self, key):
        return getattr(self.fccd, key)


def _extend_outside_band(v, band, axis: int):
    """Along every line of ``axis``, replace the values outside ``band`` by the nearest band value on that line
    (constant extension; lines without band cells are left as they are)."""
    v = np.moveaxis(v.copy(), axis, 0)
    b = np.moveaxis(band, axis, 0)
    n = v.shape[0]
    idx = np.arange(n)[:, None] * np.ones(v.shape[1], dtype=int)[None, :]
    has = b.any(axis=0)
    fwd = np.where(b, idx, -1)
    fwd = np.maximum.accumulate(fwd, axis=0)
    bwd = np.where(b, idx, n)
    bwd = np.minimum.accumulate(bwd[::-1], axis=0)[::-1]
    near = np.where(fwd < 0, bwd, np.where(bwd >= n, fwd, np.where(idx - fwd <= bwd - idx, fwd, bwd)))
    near = np.clip(near, 0, n - 1)
    ext = np.take_along_axis(v, near, axis=0)
    v = np.where(has[None, :], np.where(b, v, ext), v)
    return np.moveaxis(v, 0, axis)


def hermite_at_crossings(jcell, theta, axis: int, h: float, periodic: bool, band=None):
    """``j`` at the crossing of every face of ``axis`` (fraction ``theta`` from the lower cell): quintic Hermite
    interpolation with the CCD jet ``(j, j', j'')`` along the line (compact; no wider stencil). ``band``: cells where
    ``j`` is defined (e.g. the curvature band); outside it ``j`` is first extended with the nearest band value on the
    line. Face arrays have ``n + 1`` entries along ``axis``."""
    xp = array_module(jcell)
    jm = to_numpy(jcell)
    if band is not None:
        jm = _extend_outside_band(jm, to_numpy(band), axis)
    jm = np.moveaxis(jm, axis, 0)
    n = jm.shape[0]
    kind = "periodic" if periodic else "extrapolate"
    D1, D2, *_ = ccd_matrices(n, h, kind, kind)
    d1, d2 = D1 @ jm, D2 @ jm
    if periodic:
        L, L1, L2 = (np.concatenate([v[-1:], v]) for v in (jm, d1, d2))
        R, R1, R2 = (np.concatenate([v, v[:1]]) for v in (jm, d1, d2))
    else:  # wall faces: no crossing; reuse the first / last cell
        L, L1, L2 = (np.concatenate([v[:1], v]) for v in (jm, d1, d2))
        R, R1, R2 = (np.concatenate([v, v[-1:]]) for v in (jm, d1, d2))
    t = np.moveaxis(to_numpy(theta), axis, 0)
    t2, t3, t4, t5 = t * t, t ** 3, t ** 4, t ** 5
    H0 = 1 - 10 * t3 + 15 * t4 - 6 * t5
    H1 = t - 6 * t3 + 8 * t4 - 3 * t5
    H2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
    H3 = 10 * t3 - 15 * t4 + 6 * t5
    H4 = -4 * t3 + 7 * t4 - 3 * t5
    H5 = 0.5 * t3 - t4 + 0.5 * t5
    val = H0 * L + H1 * h * L1 + H2 * h * h * L2 + H3 * R + H4 * h * R1 + H5 * h * h * R2
    return xp.asarray(np.moveaxis(val, 0, axis))

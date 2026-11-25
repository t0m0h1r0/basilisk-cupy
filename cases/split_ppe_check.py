"""Variable-density projection with the pressure jump: a manufactured solution (docs/notes/56, phase 1-2).

Periodic unit square, drop of radius R = 0.25 at the centre, densities rho1 (inside) / rho2. With any smooth
``s`` the pressure ``p_k = rho_k s + j0 H`` has a continuous normal flux (``alpha grad p = grad s`` in both
phases) and the jump ``[p] = (rho1 - rho2) s + j0``, varying along the interface; ``div(alpha grad p) =
lap s``. Here ``s = cos(2 pi x) cos(4 pi y)`` (kx != ky) and ``j0 = 4``.

The projection solves ``D_f(alpha_f G_f p) = D_f u*`` with ``u* = alpha_f B(j) + w_f`` (``w_f`` the exact face
normal derivative of ``s``; step 1), so the projected face velocity ``u_f = w_f - alpha_f (G_f p - B(j))``
vanishes exactly: its size is the error of the jump-corrected pressure gradient (the parasitic velocity of
this problem). Variants:

* ``fd2``: Basilisk's 2nd-order faces, GFM coefficient on cut faces, sharp jump (the default of TwoPhaseIMEX);
* ``fccd-smooth``: FCCD faces solving for ``p~ = p - H j_ext`` (docs/notes/32; with a density ratio ``p~``
  keeps a kink, ``[d_n p~] = (rho1/rho2 - 1) d_n p_2``);
* ``fccd-sharp``: FCCD faces across the jump;
* ``fccd-c1``: option C of note 55 with the exact normal-derivative jump ``g`` (an oracle): FCCD on
  ``p - H (j_ext - phi g_ext)``, continuous with its normal derivative;
* ``hyb<m>-sharp`` / ``hyb<m>-smooth``: option A, the fd2 (GFM) face gradient within ``m`` faces of a cut face
  and FCCD elsewhere (:class:`HybridFaces`), on the pressure with its jump or on ``p~``.

    python -m cases.split_ppe_check [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np

from basilisk_cupy import CartesianGrid, CellField
from basilisk_cupy.boundary import BoundaryConditions
from basilisk_cupy.fields import FaceField
from basilisk_cupy.projection.approximate import project


def s_func(x, y):
    return np.cos(2 * math.pi * x) * np.cos(4 * math.pi * y)


def ds(x, y, ax):
    if ax == 0:
        return -2 * math.pi * np.sin(2 * math.pi * x) * np.cos(4 * math.pi * y)
    return -4 * math.pi * np.cos(2 * math.pi * x) * np.sin(4 * math.pi * y)


class PlainDivergence:
    """FCCD's face gradient with the plain face difference as the divergence (diagnostic for note 55)."""

    name = "fccd-plain-div"

    def __init__(self, fccd) -> None:
        from basilisk_cupy.discretization.fd2 import FD2FaceComplex

        self.fccd, self.fd2 = fccd, FD2FaceComplex()

    def div_f(self, flux, periodic=None):
        return self.fd2.div_f(flux, periodic=periodic)

    def __getattr__(self, key):
        return getattr(self.fccd, key)


class PhaseCCDFaces:
    """Option A' of note 55: phase-wise CCD along every grid line (segments between cut faces, extrapolation
    closures at the cut ends), FCCD face gradients inside the segments, and on the cut faces the GFM two-point
    gradient with the one-sided curvature terms ``(-((1-theta)h)^2 f''_R + (theta h)^2 f''_L)/(2h)`` (from the
    Taylor expansions of each phase to the crossing, with [p] = j and the continuous flux). Segments shorter
    than 6 cells keep fd2. Periodic lines only (this prototype)."""

    name = "phase-ccd"

    def __init__(self, fccd, cuts, thetas, h, alphas=None, rho_l=None) -> None:
        """``alphas``: the face coefficients (GFM on cut faces); ``rho_l``: per axis, the density of the cell on
        the lower side of every face. With them the cut-face value is the flux at the face centre, not at the
        crossing: ``F(x_f) = F_Gamma + (1/2 - theta) h (alpha p'')_side`` (the side that holds the face centre)."""
        self.fccd, self.cuts, self.thetas, self.h = fccd, cuts, thetas, h
        self.alphas, self.rho_l = alphas, rho_l
        self.third = True  # include the one-sided third-derivative Taylor terms on the cut faces

    def grad_f(self, a, axis: int):
        from basilisk_cupy.discretization.ccd import ccd_matrices

        p = np.moveaxis(np.asarray(a.interior), axis, 0)  # (n, m): lines along axis 0
        cut = np.moveaxis(self.cuts[axis], axis, 0)      # (n + 1, m) faces
        th = np.moveaxis(self.thetas[axis], axis, 0)
        af = np.moveaxis(np.asarray(self.alphas[axis]), axis, 0) if self.alphas is not None else None
        rl = np.moveaxis(self.rho_l[axis], axis, 0) if self.rho_l is not None else None
        n, m = p.shape
        h = self.h
        out = np.zeros((n + 1, m))
        for j in range(m):
            line, c, t = p[:, j], cut[:n, j], th[:n, j]  # face i between cells i-1 and i (periodic)
            g = np.zeros(n)
            q = np.full(n, np.nan)
            starts = np.nonzero(c)[0]
            if starts.size == 0:
                D1, D2, *_ = ccd_matrices(n, h, "periodic", "periodic")
                q = D2 @ line
                g = (line - np.roll(line, 1)) / h - (h / 24.0) * (q - np.roll(q, 1))
            else:
                for k, s0 in enumerate(starts):
                    s1 = starts[(k + 1) % len(starts)]
                    length = (s1 - s0) % n or n
                    idx = (s0 + np.arange(length)) % n  # cells of the segment
                    seg = line[idx]
                    if length >= 6:
                        _, D2, *_ = ccd_matrices(length, h, "extrapolate", "extrapolate")
                        qs = D2 @ seg
                        q[idx] = qs
                        g[idx[1:]] = (seg[1:] - seg[:-1]) / h - (h / 24.0) * (qs[1:] - qs[:-1])
                    else:
                        g[idx[1:]] = (seg[1:] - seg[:-1]) / h
                        if length >= 3:  # second differences (one-sided copies at the ends) for the cut terms
                            d2 = np.empty(length)
                            d2[1:-1] = (seg[2:] - 2 * seg[1:-1] + seg[:-2]) / h ** 2
                            d2[0], d2[-1] = d2[1], d2[-2]
                            q[idx] = d2
                for s0 in starts:  # cut faces: GFM two-point difference + one-sided Taylor terms
                    L, R = (s0 - 1) % n, s0
                    tt = t[s0]
                    g[s0] = (line[R] - line[L]) / h
                    qL = q[L] if np.isfinite(q[L]) else 0.0
                    qR = q[R] if np.isfinite(q[R]) else 0.0
                    c3L = c3R = 0.0
                    if self.third and np.isfinite(q[(L - 1) % n]) and np.isfinite(q[(R + 1) % n]):
                        c3L = (qL - q[(L - 1) % n]) / h  # one-sided third derivatives within each phase
                        c3R = (q[(R + 1) % n] - qR) / h
                    bL = qL + tt * h * c3L          # second derivatives at the crossing
                    bR = qR - (1 - tt) * h * c3R
                    a, b = tt * h, (1 - tt) * h
                    g[s0] += (-b * b * bR / 2 - b ** 3 * c3R / 6 + a * a * bL / 2 - a ** 3 * c3L / 6) / h
                    if af is not None:
                        # from the crossing to the face centre, in the phase that holds the centre
                        rho_side = rl[s0, j] if tt > 0.5 else self._rho_other(rl[s0, j])
                        b_side, c_side = (bL, c3L) if tt > 0.5 else (bR, c3R)
                        d = (0.5 - tt) * h
                        g[s0] += (d * b_side + d * d * c_side / 2) / (rho_side * af[s0, j])
            out[:n, j] = g
            out[n, j] = g[0]
        return np.moveaxis(out, 0, axis)

    def _rho_other(self, r):
        return self.rho_pair[1] if r == self.rho_pair[0] else self.rho_pair[0]

    def __getattr__(self, key):
        return getattr(self.fccd, key)


class HybridFaces:
    """Option A of note 55: FCCD's face gradient away from the interface, the fd2 (GFM) two-point gradient on
    the faces within ``m`` faces of a cut face (both axes); every other operation is FCCD's."""

    name = "hybrid"

    def __init__(self, fccd, masks) -> None:
        from basilisk_cupy.discretization.fd2 import FD2FaceComplex

        self.fccd, self.fd2, self.masks = fccd, FD2FaceComplex(), masks

    def grad_f(self, a, axis: int):
        return np.where(self.masks[axis], self.fd2.grad_f(a, axis), self.fccd.grad_f(a, axis))

    def __getattr__(self, key):
        return getattr(self.fccd, key)


def _near_cut(cut, m: int, periodic_axes=(True, True)):
    """Faces within ``m`` faces (along both axes) of a cut face; ``cut`` is a face array of one axis."""
    near = cut.copy()
    for _ in range(m):
        grown = near.copy()
        for ax in range(2):
            grown |= np.roll(near, 1, ax) | np.roll(near, -1, ax)
        near = grown
    return near


def _hermite_jump(jcell, theta, ax, h):
    """``j`` at the crossings by quintic Hermite interpolation between the two cells of every face, with the
    CCD jet ``(j, j', j'')`` along the face's axis (periodic lines): compact, no wider stencil."""
    from basilisk_cupy.discretization.ccd import ccd_matrices

    jm = np.moveaxis(jcell, ax, 0)
    n = jm.shape[0]
    D1, D2, *_ = ccd_matrices(n, h, "periodic", "periodic")
    d1, d2 = D1 @ jm, D2 @ jm
    L = np.concatenate([jm[-1:], jm], axis=0)  # face i: cells i-1 (L) and i (R), periodic
    R = np.concatenate([jm, jm[:1]], axis=0)
    L1, R1 = np.concatenate([d1[-1:], d1]), np.concatenate([d1, d1[:1]])
    L2, R2 = np.concatenate([d2[-1:], d2]), np.concatenate([d2, d2[:1]])
    t = np.moveaxis(theta, ax, 0)
    # quintic Hermite on [0, 1] with values, h-scaled first and h^2-scaled second derivatives
    t2, t3, t4, t5 = t * t, t ** 3, t ** 4, t ** 5
    H0 = 1 - 10 * t3 + 15 * t4 - 6 * t5
    H1 = t - 6 * t3 + 8 * t4 - 3 * t5
    H2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
    H3 = 10 * t3 - 15 * t4 + 6 * t5
    H4 = -4 * t3 + 7 * t4 - 3 * t5
    H5 = 0.5 * t3 - t4 + 0.5 * t5
    val = H0 * L + H1 * h * L1 + H2 * h * h * L2 + H3 * R + H4 * h * R1 + H5 * h * h * R2
    return np.moveaxis(val, 0, ax)


def solve(n: int, ratio: float, variant: str, j0: float = 4.0, tolerance: float = 1e-11, exact_jump: bool = False,
          jump_interp: str = "linear"):
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    warnings.filterwarnings("ignore")
    rho1, rho2 = ratio, 1.0
    grid = CartesianGrid((n, n))
    x, y = grid.meshgrid()
    faces = "fccd" if variant.startswith(("fccd", "hyb", "phase-ccd", "lib")) else "fd2"
    # the stepper is only used for its geometry and operators (equal densities pass its check)
    ns = TwoPhaseIMEX(grid, 0.25 - np.hypot(x - 0.5, y - 0.5), rho1=1.0, rho2=1.0, periodic=(0, 1), faces=faces)
    tp = ns.tp
    tp.rho1, tp.rho2 = rho1, rho2
    tp._properties(ns)  # GFM alpha with the density ratio
    alpha = ns.alpha
    H = (ns.psi > 0.5).astype(float)
    jcell = (rho1 - rho2) * s_func(x, y) + j0
    h = float(grid.h[0])
    ustar = []
    smooth = variant in ("fccd-smooth", "fccd-c1", "fccd-c1-plaindiv", "fccd-smooth-plaindiv", "fccd-exact-ext",
                         "fccd-exact-c1") or variant.endswith("-smooth") and variant.startswith("hyb")
    jext = tp.extend_jump(jcell) if smooth else None
    if variant in ("fccd-exact-ext", "fccd-exact-c1"):
        # exact closest-point extension of the jump (constant along the normal of the circle)
        r = np.hypot(x - 0.5, y - 0.5)
        xg, yg = 0.5 + 0.25 * (x - 0.5) / r, 0.5 + 0.25 * (y - 0.5) / r
        jext = (rho1 - rho2) * s_func(xg, yg) + j0
        if variant == "fccd-exact-c1":
            dn = (ds(xg, yg, 0) * (xg - 0.5) + ds(xg, yg, 1) * (yg - 0.5)) / 0.25
            jext = jext - (0.25 - r) * (rho1 - rho2) * dn
    if variant in ("fccd-c1", "fccd-c1-plaindiv"):
        # option C with the exact normal-derivative jump g = [d_n p] = (rho1 - rho2) d_n s (n outward):
        # p - H (j_ext - phi g_ext) is C^1 across the interface (phi = distance, positive inside)
        r = np.hypot(x - 0.5, y - 0.5)
        dn = (ds(x, y, 0) * (x - 0.5) + ds(x, y, 1) * (y - 0.5)) / np.maximum(r, 1e-12)
        gext = tp.extend_jump((rho1 - rho2) * dn)
        jext = jext - (0.25 - r) * gext
    for ax in range(2):
        hl, hr, cut, theta = tp._cut(ax)
        jl, jr = tp._faces(jcell, ax)
        jg = (1.0 - theta) * jl + theta * jr
        if jump_interp == "hermite":
            jg = _hermite_jump(jcell, theta, ax, h)
        if variant == "lib":  # the library's operator and crossing interpolation
            from basilisk_cupy.discretization.phase_ccd import hermite_at_crossings

            jg = hermite_at_crossings(jcell, theta, ax, h, True)
        if exact_jump:  # the jump at the crossing itself instead of its linear interpolation
            if ax == 0:
                xc = np.concatenate([x[:1] - h, x], axis=0) + theta * h
                yc = np.concatenate([y[:1], y], axis=0)
            else:
                xc = np.concatenate([x[:, :1], x], axis=1)
                yc = np.concatenate([y[:, :1] - h, y], axis=1) + theta * h
            jg = np.where(cut, (rho1 - rho2) * s_func(xc, yc) + j0, 0.0)
        B = np.where(cut, jg * (hr.astype(float) - hl.astype(float)), 0.0) / h
        if smooth:
            el, er = tp._faces(jext, ax)
            B = B - (hr * er - hl * el) / h
        # exact face-normal derivative of s at the face centres
        fx, fy = grid.meshgrid()
        if ax == 0:
            xf = np.concatenate([x[:1] - 0.5 * h, x + 0.5 * h], axis=0)
            yf = np.concatenate([y[:1], y], axis=0)
        else:
            xf = np.concatenate([x[:, :1], x], axis=1)
            yf = np.concatenate([y[:, :1] - 0.5 * h, y + 0.5 * h], axis=1)
        del fx, fy
        ustar.append(alpha[ax] * B + ds(xf, yf, ax))
    uf = FaceField(grid, ustar, "uf")
    p = CellField(grid, bc=BoundaryConditions.periodic(2))
    ops = ns.ops
    if variant == "lib":
        from basilisk_cupy.discretization.phase_ccd import PhaseCCDFaces as LibPhaseCCD

        ops = LibPhaseCCD(ns.ops)
        ops.update(tp, alpha)
    elif variant.startswith("phase-ccd"):
        cuts, thetas = [], []
        for ax in range(2):
            hl, hr, cut, theta = tp._cut(ax)
            cuts.append(cut)
            thetas.append(theta)
        rho_l = [np.where(tp._cut(ax)[0], rho1, rho2) for ax in range(2)]
        ops = PhaseCCDFaces(ns.ops, cuts, thetas, h, alphas=[alpha[ax] for ax in range(2)], rho_l=rho_l)
        ops.rho_pair = (rho1, rho2)
        ops.third = not variant.endswith("-2nd")
        if variant.endswith("plaindiv"):
            ops = PlainDivergence(ops)
    elif variant.endswith("plaindiv"):
        ops = PlainDivergence(ns.ops)
    if variant.startswith("hyb"):
        m = int(variant[3:].split("-")[0])
        ops = HybridFaces(ns.ops, [_near_cut(tp._cut(ax)[2], m) for ax in range(2)])
    project(uf, p, alpha, 1.0, ops=ops, tolerance=tolerance)
    pfull = p.interior + (H * jext if smooth else 0.0)
    exact = np.where(H > 0, rho1, rho2) * s_func(x, y) + j0 * H
    err = pfull - exact
    err = err - err.mean()
    band = (ns.psi > 0.01) & (ns.psi < 0.99)
    res = max(float(np.abs(uf[ax]).max()) for ax in range(2))
    return {"p_err": float(np.abs(err).max()), "p_err_far": float(np.abs(np.where(band, 0, err)).max()),
            "u_res": res, "p_scale": float(np.abs(exact).max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(n=n, ratio=r, variant=v) for v in ("fd2", "fccd-smooth", "fccd-sharp")
             for r in (1.0, 10.0, 1000.0) for n in (32, 64, 128)]
    res = pmap(solve, items, args.jobs)
    for it, r in zip(items, res):
        print(f"{it['variant']:12s} ratio {it['ratio']:6g} N {it['n']:4d}: p err {r['p_err']:.3e} "
              f"(outside the band {r['p_err_far']:.3e}, scale {r['p_scale']:.0f}), |u_f| {r['u_res']:.3e}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = {"fd2": "o-", "fccd-smooth": "s--", "fccd-sharp": "^:"}
    colors = {1.0: "C0", 10.0: "C1", 1000.0: "C3"}
    for v in styles:
        for ratio in colors:
            pts = [(it["n"], r["u_res"]) for it, r in zip(items, res) if it["variant"] == v and it["ratio"] == ratio]
            ax.loglog(*zip(*pts), styles[v], color=colors[ratio], label=f"{v}, density ratio {ratio:g}")
    ns_ = np.array([32, 128])
    ax.loglog(ns_, 2.4e-2 * (ns_ / 32.0) ** -2, "k-", lw=0.6)
    ax.loglog(ns_, 1.3e-4 * (ns_ / 32.0) ** -4, "k-", lw=0.6)
    ax.text(90, 4e-3, "h²", fontsize=8)
    ax.text(90, 2e-6, "h⁴", fontsize=8)
    ax.set_xlabel("N")
    ax.set_ylabel("residual face velocity max |u_f|")
    ax.set_title("manufactured p_k = ρ_k s + j₀H: projected face velocity")
    ax.legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(f"{args.out}/density_ratio_fccd.png", dpi=150, bbox_inches="tight")
    print("wrote", f"{args.out}/density_ratio_fccd.png")


if __name__ == "__main__":
    main()

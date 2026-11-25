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


def solve(n: int, ratio: float, variant: str, j0: float = 4.0, tolerance: float = 1e-11):
    from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX

    warnings.filterwarnings("ignore")
    rho1, rho2 = ratio, 1.0
    grid = CartesianGrid((n, n))
    x, y = grid.meshgrid()
    faces = "fccd" if variant.startswith(("fccd", "hyb")) else "fd2"
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
    smooth = variant in ("fccd-smooth",) or variant.endswith("-smooth") and variant.startswith("hyb")
    jext = tp.extend_jump(jcell) if smooth else None
    for ax in range(2):
        hl, hr, cut, theta = tp._cut(ax)
        jl, jr = tp._faces(jcell, ax)
        jg = (1.0 - theta) * jl + theta * jr
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
    args = ap.parse_args()
    from cases.parallel import pmap

    items = [dict(n=n, ratio=r, variant=v) for v in ("fd2", "fccd-smooth", "fccd-sharp")
             for r in (1.0, 10.0, 1000.0) for n in (32, 64, 128)]
    for it, r in zip(items, pmap(solve, items, args.jobs)):
        print(f"{it['variant']:12s} ratio {it['ratio']:6g} N {it['n']:4d}: p err {r['p_err']:.3e} "
              f"(outside the band {r['p_err_far']:.3e}, scale {r['p_scale']:.0f}), |u_f| {r['u_res']:.3e}")


if __name__ == "__main__":
    main()

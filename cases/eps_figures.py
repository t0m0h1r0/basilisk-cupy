"""Interface width eps / h (docs/notes/28): ``eps_width.png``.

Top: errors against eps / h (1.5, 2, 3, 4): ACLS (FCCD, 2008 form) flux on a planar profile, curvature of
psi on an exact circle (N = 64 / 128 / 256), psi and curvature after a periodic translation with the
corrected divergence and 6th-order face values (N = 32 / 64 / 128). Bottom: psi - exact after the
translation (N = 64) and the reversed vortex at T (N = 64) for eps / h = 1.5 and 3.

    python -m cases.eps_figures [--out docs/figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid, FaceField
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_curvature import psi_curvature

C = (1.5, 2.0, 3.0, 4.0)


def planar(c, n=64):
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    d = x
    cls = CLSInterface.from_level_set(g, d, BoundaryConditions(), eps=c / n)
    core = (np.abs(d) < 3 * cls.eps) & (np.abs(y) < 0.3)
    fn = cls._acls_face_normals(np.ones_like(d), np.zeros_like(d))
    return float(np.abs(cls._acls_fccd_divergence(cls.psi.interior, fn)[core]).max()) / n


def exact_curvature(c, n):
    R, cx, cy = 0.2, -0.0123, 0.0217
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    r = np.hypot(x - cx, y - cy)
    cls = CLSInterface.from_level_set(g, R - r, BoundaryConditions(), eps=c / n)
    core = np.abs(R - r) < cls.eps
    return float(np.abs(psi_curvature(cls.psi) - 1 / r)[core].max()) * R


def translation(c, n):
    R = 0.2
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    bc = BoundaryConditions.periodic(2)
    cls = CLSInterface.from_level_set(g, R - np.hypot(x, y), bc, eps=c / n, divergence="fccd",
                                      ops=FCCDFaceComplex(order=6))
    uf = FaceField(g, [np.ones(g.face_shape(0)), 0.5 * np.ones(g.face_shape(1))])
    for _ in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
    ye = ((y + 1) % 1) - 0.5
    r = np.hypot(x, ye)
    ex = CLSInterface.from_level_set(g, R - r, bc, eps=c / n).psi.interior
    core = np.abs(R - r) < cls.eps
    k = float(np.abs(psi_curvature(cls.psi) - 1 / r)[core].max()) * R
    return float(np.abs(cls.psi.interior - ex).sum()) / n / n, k, cls.psi.interior - ex, x, y


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases import reversed_cls as rc

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig = plt.figure(figsize=(17, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    a0 = fig.add_subplot(gs[0, 0])
    a0.semilogy(C, [planar(c) for c in C], "o-")
    a0.set_title("ACLS (FCCD) flux, planar profile: max|div F| h")
    a1 = fig.add_subplot(gs[0, 1])
    for n in (64, 128, 256):
        a1.semilogy(C, [exact_curvature(c, n) for c in C], "o-", label=f"N = {n}")
    a1.set_title("psi curvature, exact circle: max|kappa R - R/r|")
    a1.legend()
    a2 = fig.add_subplot(gs[0, 2])
    a3 = fig.add_subplot(gs[0, 3])
    fields = {}
    for n in (32, 64, 128):
        res = [translation(c, n) for c in C]
        a2.semilogy(C, [r_[0] for r_ in res], "o-", label=f"N = {n}")
        a3.semilogy(C, [r_[1] for r_ in res], "o-", label=f"N = {n}")
        if n == 64:
            fields = {c: (r_[2], r_[3], r_[4]) for c, r_ in zip(C, res)}
    a2.set_title("translation: psi L1 error")
    a3.set_title("translation: curvature max error")
    a2.legend()
    a3.legend()
    for a in (a0, a1, a2, a3):
        a.set_xlabel("eps / h")
        a.grid(alpha=0.3)
    for k, c in enumerate((1.5, 3.0)):
        d, x, y = fields[c]
        a = fig.add_subplot(gs[1, k])
        im = a.imshow(d.T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="RdBu_r", vmin=-2e-3, vmax=2e-3)
        a.set_title(f"translation N = 64, eps/h = {c}\npsi - exact", fontsize=10)
        fig.colorbar(im, ax=a, shrink=0.8)
    n = 64
    g = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    for k, c in enumerate((1.5, 3.0)):
        r = rc.run(n, "numpy", divergence="fccd", face_order=6, eps_factor=c)
        a = fig.add_subplot(gs[1, 2 + k])
        a.imshow(r["psi"].T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="Blues", vmin=0, vmax=1)
        a.contour(x, y, r["psi"], [0.5], colors="r", linewidths=0.8)
        tt = np.linspace(0, 2 * np.pi, 200)
        a.plot(-0.2 + 0.2 * np.cos(tt), -0.236338 + 0.2 * np.sin(tt), "k--", lw=0.6)
        a.set_title(f"reversed vortex N = 64, eps/h = {c}\nsharp error {r['sharp'][0]:.4f}", fontsize=10)
    fig.suptitle("Interface width eps / h")
    fig.savefig(f"{args.out}/eps_width.png", dpi=100)
    print("wrote", f"{args.out}/eps_width.png")


if __name__ == "__main__":
    main()

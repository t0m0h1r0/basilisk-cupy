"""2-D passive scalar advection with the compact BCG flux (docs/notes/74): periodic [0, 1]^2, face velocities from a
stream function (discretely divergence-free), direction-split sweeps (Strang: x-y then y-x on alternate steps), each
sweep ``q -= dt/h (F_{i+1/2} - F_{i-1/2}) - dt/h q (u_{i+1/2} - u_{i-1/2})`` (the dilation of the 1-D sweep, as vof.h's
``cc`` term). Face values from BCG (centred slope) or the compact flux (``cases.compact_flux_1d.CompactFlux``,
alpha = 1/3, j = -2..2). Tests:

* uniform translation (1, 0.5) for one period (the sweeps commute: no splitting error);
* a deformation vortex (after LeVeque) psi = sin(2 pi x) sin(2 pi y) / (2 pi) on the periodic square, reversed by
  cos(pi t / T), back to the initial state at T = 0.5 (T = 2 deforms the Gaussian into filaments that N = 128 does not
  resolve: errors 0.46 / 0.18 for BCG / compact, not converging).

    python -m cases.compact_flux_2d
"""

from __future__ import annotations

import math

import numpy as np


def faces_along(q, uf, dt, h, axis, scheme):
    """Face values r on faces k = 0..n-1 along ``axis`` (face k between cells k - 1 and k), every line at once."""
    qa = np.moveaxis(q, axis, 0)
    ua = np.moveaxis(uf, axis, 0)
    out = np.empty_like(qa)
    for j in range(qa.shape[1]):
        out[:, j] = scheme(qa[:, j], ua[:, j], dt, h)
    return np.moveaxis(out, 0, axis)


def sweep(q, uf, dt, h, axis, scheme):
    r = faces_along(q, uf, dt, h, axis, scheme)
    flux = uf * r
    dflux = np.roll(flux, -1, axis=axis) - flux
    du = np.roll(uf, -1, axis=axis) - uf
    return q - dt / h * dflux + dt / h * q * du


def face_velocities(psi, n):
    """u on x-faces (x = i h, y cell centres), v on y-faces, from psi at the vertices: exactly divergence-free."""
    h = 1.0 / n
    xv = np.arange(n) * h
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    P = psi(X, Y)  # vertices (i h, j h)
    u = (np.roll(P, -1, axis=1) - P) / h       # x-face (i h, (j + 1/2) h): psi_y
    v = -(np.roll(P, -1, axis=0) - P) / h      # y-face ((i + 1/2) h, j h): -psi_x
    return u, v


def cell_average(f, n, order: int = 6):
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(order)
    c = (np.arange(n) + 0.5) * h
    out = np.zeros((n, n))
    for a, wa in zip(xg, wg):
        for b, wb in zip(xg, wg):
            X, Y = np.meshgrid(c + 0.5 * h * a, c + 0.5 * h * b, indexing="ij")
            out += 0.25 * wa * wb * f(X, Y)
    return out


def run(scheme, n, velocity, t_end, cfl, q0):
    h = 1.0 / n
    umax = max(np.abs(u).max() for u in velocity(0.0, n))
    dt = cfl * h / umax
    steps = int(math.ceil(t_end / dt - 1e-9))
    dt = t_end / steps
    q = q0.copy()
    for s in range(steps):
        u, v = velocity((s + 0.5) * dt, n)
        order = (0, 1) if s % 2 == 0 else (1, 0)
        for axis in order:
            q = sweep(q, u if axis == 0 else v, dt, h, axis, scheme)
    return q


def main() -> None:
    from cases.compact_flux_1d import CompactFlux, bcg_faces

    schemes = {"bcg": bcg_faces, "compact a=1/3 j=-2..2": CompactFlux(1.0 / 3.0, (-2, -1, 0, 1, 2)).faces}
    bump = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)  # noqa: E731

    def uniform(t, n):
        return 1.0 + np.zeros((n, n)), 0.5 + np.zeros((n, n))

    def vortex(t, n, T=0.5):
        u, v = face_velocities(lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) / (2 * np.pi), n)
        c = math.cos(math.pi * t / T)
        return c * u, c * v

    gauss = lambda x, y: np.exp(-((x - 0.5) ** 2 + (y - 0.3) ** 2) / 0.01)  # noqa: E731
    for label, vel, t_end, f in (("uniform (1, 0.5), t = 2", uniform, 2.0, bump),
                                 ("deformation vortex, t = T = 0.5", vortex, 0.5, gauss)):
        print(f"-- {label}, CFL 0.5")
        for name, sch in schemes.items():
            errs = []
            ns = (32, 64, 128) if vel is uniform else (32, 64, 128, 256)
            for n in ns:
                q0 = cell_average(f, n)
                errs.append(float(np.abs(run(sch, n, vel, t_end, 0.5, q0) - q0).max()))
            orders = [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
            print(f"   {name:22s} " + " ".join(f"{e:.2e}" for e in errs) + "  order " + " ".join(f"{o:.2f}" for o in orders),
                  flush=True)


if __name__ == "__main__":
    main()

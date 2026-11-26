"""1-D prototype of the compact BCG flux (docs/notes/74): q_t + (u q)_x = 0 on the periodic [0, 1), cell averages,
face fluxes u_f r_f with r_f from

    alpha r_{f-1} + r_f + alpha r_{f+1} = sum_j a_j(C_f) qbar_{up(f) + j}        (C_f = |u_f| dt / h)

(fixed left-hand side, solved by FFT; right-hand coefficients from ``cases.compact_flux_theory.compact_fit`` per face,
the stencil mirrored for u_f < 0). Compared with BCG (r = q + (1 - C)/2 * centred slope) and first-order upwind:
the convergence of a sine wave advected one period at constant velocity, the same with a variable velocity
u(x) = 1 + 0.5 sin(2 pi x) (against a fine-grid solution), and the overshoot of a square wave.

    python -m cases.compact_flux_1d
"""

from __future__ import annotations

import math

import numpy as np


class CompactFlux:
    def __init__(self, alpha: float, stencil, ncoef: int = 101):
        from cases.compact_flux_theory import compact_fit

        self.alpha, self.stencil = float(alpha), np.array(stencil)
        self.Cs = np.linspace(0.0, 1.0, ncoef)
        self.table = np.array([compact_fit(max(C, 1e-6), alpha, self.stencil) for C in self.Cs])  # (ncoef, m)

    def coefficients(self, C):
        """Linear interpolation of a_j(C) in C (the coefficients are smooth in C)."""
        return np.stack([np.interp(C, self.Cs, self.table[:, k]) for k in range(len(self.stencil))], axis=-1)

    def faces(self, q, uf, dt, h):
        """r on faces k = 0..n-1 (face k between cells k - 1 and k, periodic)."""
        n = q.size
        C = np.abs(uf) * dt / h
        a = self.coefficients(C)
        rhs = np.zeros(n)
        pos = uf > 0
        for idx, j in enumerate(self.stencil):
            # upwind cell of face k: k - 1 (u > 0) or k (u < 0, the stencil mirrored)
            rhs += np.where(pos, a[:, idx] * np.roll(q, -(j - 1)), a[:, idx] * np.roll(q, j))
        # alpha r_{k-1} + r_k + alpha r_{k+1} = rhs, periodic: FFT
        k = np.fft.fftfreq(n) * n
        symbol = 1.0 + 2.0 * self.alpha * np.cos(2 * np.pi * k / n)
        return np.real(np.fft.ifft(np.fft.fft(rhs) / symbol))


def bcg_faces(q, uf, dt, h):
    C = uf * dt / h
    slope = 0.5 * (np.roll(q, -1) - np.roll(q, 1))
    left = np.roll(q, 1) + 0.5 * (1.0 - C) * np.roll(slope, 1)   # from cell k - 1
    right = q - 0.5 * (1.0 + C) * slope                           # from cell k
    return np.where(uf > 0, left, right)


def upwind_faces(q, uf, dt, h):
    return np.where(uf > 0, np.roll(q, 1), q)


def cell_average(f, n):
    x = np.linspace(0.0, 1.0, n + 1)
    xg, wg = np.polynomial.legendre.leggauss(8)
    out = np.zeros(n)
    for k in range(n):
        xs = 0.5 * (x[k + 1] - x[k]) * xg + 0.5 * (x[k + 1] + x[k])
        out[k] = 0.5 * np.sum(wg * f(xs))
    return out


def advect(scheme, q0, velocity, n, t_end, cfl):
    h = 1.0 / n
    xf = np.arange(n) * h  # faces
    uf = velocity(xf)
    dt = cfl * h / np.abs(uf).max()
    steps = int(math.ceil(t_end / dt - 1e-9))
    dt = t_end / steps
    q = q0.copy()
    for _ in range(steps):
        r = scheme(q, uf, dt, h)
        flux = uf * r
        q = q - dt / h * (np.roll(flux, -1) - flux)
    return q


def main() -> None:
    schemes = {"upwind": upwind_faces, "bcg": bcg_faces}
    for alpha, st in ((0.25, (-2, -1, 0, 1, 2)), (1.0 / 3.0, (-2, -1, 0, 1, 2))):
        cf = CompactFlux(alpha, st)
        schemes[f"compact a={alpha:.3f} j=-2..2"] = cf.faces
    sine = lambda x: np.sin(2 * np.pi * x)  # noqa: E731
    const = lambda x: np.ones_like(x)  # noqa: E731
    for cfl in (0.5, 0.8):
        print(f"-- constant velocity, sine, one period, CFL {cfl}")
        for name, sch in schemes.items():
            errs = []
            ns = (16, 32, 64, 128)
            for n in ns:
                q0 = cell_average(sine, n)
                errs.append(float(np.abs(advect(sch, q0, const, n, 1.0, cfl) - q0).max()))
            orders = [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
            print(f"   {name:24s} " + " ".join(f"{e:.2e}" for e in errs) + "  order " + " ".join(f"{o:.2f}" for o in orders))
    print("-- variable velocity u = 1 + 0.5 sin(2 pi x), sine, t = 0.5, CFL 0.6 (against N = 1024 of each scheme)")
    var = lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x)  # noqa: E731
    for name, sch in schemes.items():
        ref = advect(sch, cell_average(sine, 1024), var, 1024, 0.5, 0.6)
        errs = []
        for n in (32, 64, 128, 256):
            q = advect(sch, cell_average(sine, n), var, n, 0.5, 0.6)
            refc = ref.reshape(n, -1).mean(axis=1)
            errs.append(float(np.abs(q - refc).max()))
        orders = [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
        print(f"   {name:24s} " + " ".join(f"{e:.2e}" for e in errs) + "  order " + " ".join(f"{o:.2f}" for o in orders))
    print("-- square wave (0.25 < x < 0.75), constant velocity, one period, N = 128, CFL 0.5: min, max")
    square = lambda x: ((x > 0.25) & (x < 0.75)).astype(float)  # noqa: E731
    for name, sch in schemes.items():
        q = advect(sch, cell_average(square, 128), const, 128, 1.0, 0.5)
        print(f"   {name:24s} min {q.min():+.4f} max {q.max():.4f}")


if __name__ == "__main__":
    main()

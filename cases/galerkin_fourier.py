"""docs/notes/83 (P0-8a, P0-9a): the Galerkin transport G u^{n+1} = C u^n at constant density, periodic, uniform
velocity U, as a Fourier (Bloch) symbol, for two reconstructions from cell averages:

* "poly": the cell-wise quartic of note 80 (cell i's polynomial matches the averages of cells i-2 .. i+2);
* "spline": the continuous quartic B-spline u_h = sum_j a_j B4((x - x_j)/h), the coefficients from the cell averages
  (a_{i-2} + 26 a_{i-1} + 66 a_i + 26 a_{i+1} + a_{i+2})/120 = u_i (the user's proposal).

For u_j = e^{i j theta} the reconstruction is a Bloch function phi_theta; translation keeps the Bloch class, so the
Galerkin step is the projection onto span(phi_theta):

    T(C, theta) = <phi_theta, S_C phi_theta> / <phi_theta, phi_theta>     (S_C: shift by C cells, inner product on
                                                                            one cell, exact Gauss quadrature)

and the energy ratio per step is |T|^2 (h = 1, Delta t = C h / U).

    python -m cases.galerkin_fourier
"""

from __future__ import annotations

import json
import math

import numpy as np

XG, WG = np.polynomial.legendre.leggauss(12)


def bspline4(x):
    """The centred cardinal quartic B-spline (support [-5/2, 5/2], unit integral)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k in range(6):
        out = out + (-1) ** k * math.comb(5, k) * np.maximum(x + 2.5 - k, 0.0) ** 4
    return out / 24.0


def poly_coefficients(theta):
    """Coefficients c_r (xi = x - x_i in cells) of cell 0's quartic for the averages e^{i o theta}, o = -2..2."""
    offs = np.arange(-2, 3)
    A = np.array([[((o + 0.5) ** (r + 1) - (o - 0.5) ** (r + 1)) / (r + 1) for r in range(5)] for o in offs])
    return np.linalg.solve(A.astype(complex), np.exp(1j * offs * theta))


def spline_symbol(theta):
    """A(theta) = (66 + 52 cos theta + 2 cos 2 theta)/120: the cell averages of the B-spline coefficients."""
    return (66 + 52 * np.cos(theta) + 2 * np.cos(2 * theta)) / 120.0


def phi(kind, theta, x):
    """The reconstruction of u_j = e^{i j theta} at points x (cell j = [j - 1/2, j + 1/2])."""
    x = np.asarray(x, dtype=float)
    if kind == "poly":
        c = poly_coefficients(theta)
        m = np.floor(x + 0.5)
        xi = x - m
        return np.exp(1j * m * theta) * np.polyval(c[::-1], xi)
    a0 = 1.0 / spline_symbol(theta)
    j0 = np.floor(x + 0.5)
    out = np.zeros(x.shape, dtype=complex)
    for d in range(-3, 4):
        j = j0 + d
        out = out + a0 * np.exp(1j * j * theta) * bspline4(x - j)
    return out


def _cell_integral(f, a, b):
    """int_a^b f over [a, b] with Gauss (f smooth on [a, b])."""
    x = 0.5 * (b - a) * XG + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(WG * f(x))


def symbol(kind, C, theta):
    """T(C, theta): <phi, S_C phi>/<phi, phi> on cell 0 = [-1/2, 1/2] (pieces split where x - C crosses a cell
    boundary; the B-spline's knots are the cell boundaries too)."""
    f = lambda x: phi(kind, theta, x)  # noqa: E731
    cut = -0.5 + C - math.floor(C)
    pts = sorted({-0.5, 0.5, *([cut] if -0.5 < cut < 0.5 else [])})
    num = sum(_cell_integral(lambda x: np.conj(f(x)) * f(x - C), a, b) for a, b in zip(pts[:-1], pts[1:]))
    den = _cell_integral(lambda x: np.abs(f(x)) ** 2, -0.5, 0.5)
    return num / den


def dissipation(kind, C, theta, beta: float = 0.0):
    """d = (1 - |T|^2)/C and gamma h/U = -log|T|^2 / C (energy decay per unit time in units U/h). ``beta``: the
    adjoint correction T (1 + beta (1 - |T|^2)) (docs/notes/84; at constant density T^dagger = conj(T))."""
    T = symbol(kind, C, theta)
    T = T * (1 + beta * (1 - abs(T) ** 2))
    a = abs(T) ** 2
    return {"T": [T.real, T.imag], "d": (1 - a) / C, "gamma": -math.log(max(a, 1e-300)) / C}


def jump_rate(theta):
    """Small-C energy decay of the cell-wise quartic from its face jumps: gamma h/U = |J|^2 / int_cell |phi|^2."""
    J = phi("poly", theta, np.array([0.5 + 1e-13]))[0] - phi("poly", theta, np.array([0.5 - 1e-13]))[0]
    g = _cell_integral(lambda x: np.abs(phi("poly", theta, x)) ** 2, -0.5, 0.5)
    return abs(J) ** 2 / g, abs(J) ** 2


def main() -> None:
    thetas = {"16h": math.pi / 8, "8h": math.pi / 4, "6h": math.pi / 3, "4h": math.pi / 2, "3h": 2 * math.pi / 3,
              "2h": math.pi}
    cfls = (0.001, 0.01, 0.1, 0.37, 0.5, 0.75, 1.0)
    out = {"jump": {}, "table": {}}
    print("-- face jump of the cell-wise quartic: gamma h/U from |J|^2 (small C); |J|^2 vs 1024/900 sin^10(theta/2)")
    for name, th in thetas.items():
        g, J2 = jump_rate(th)
        out["jump"][name] = {"gamma": g, "J2": J2, "formula": 1024 / 900 * math.sin(th / 2) ** 10,
                             "nu_num": g / (2 * th * th)}
        print(f"   {name}: gamma {g:.3e}  |J|^2 {J2:.6e}  formula {1024 / 900 * math.sin(th / 2) ** 10:.6e}  "
              f"nu_num/(U h) {g / (2 * th * th):.3e}")
    for kind in ("poly", "spline"):
        print(f"-- {kind}: d = (1 - |T|^2)/C  [gamma = -log|T|^2 / C]")
        for name, th in thetas.items():
            row = [dissipation(kind, C, th) for C in cfls]
            out["table"][f"{kind} {name}"] = {str(C): r for C, r in zip(cfls, row)}
            print(f"   {name}: " + "  ".join(f"{r['d']:.3e} [{r['gamma']:.3e}]" for r in row))
    # the whole wavenumber range for the figure
    ths = np.linspace(1e-3, math.pi, 200)
    out["curve"] = {kind: {str(C): [dissipation(kind, C, th)["gamma"] for th in ths] for C in (0.001, 0.37)}
                    for kind in ("poly", "spline")}
    out["curve"]["theta"] = ths.tolist()
    # accuracy: one-step error |T - e^{-i C theta}| at C = 0.37 for small theta
    print("-- one-step error |T - e^{-iC theta}|, C = 0.37, theta = 2 pi / (8, 16, 32, 64) cells, and orders")
    for kind in ("poly", "spline"):
        errs = []
        for m in (8, 16, 32, 64):
            th = 2 * math.pi / m
            T = symbol(kind, 0.37, th)
            errs.append(abs(T - np.exp(-1j * 0.37 * th)))
        print(f"   {kind}: " + " ".join(f"{e:.2e}" for e in errs) + "  orders "
              + " ".join(f"{math.log2(errs[i] / errs[i + 1]):.2f}" for i in range(3)))
        out[f"accuracy {kind}"] = errs
    with open("docs/figures/galerkin_fourier.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()

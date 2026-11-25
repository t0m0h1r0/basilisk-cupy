"""Modified dispersion relation: continuous Navier--Stokes, discrete interface kinematics (docs/notes/41).

The normal mode of two identical viscous fluids (:mod:`cases.capwave_modes`) with the kinematic condition
``s eta = v(0)`` replaced by ``s eta = sum_f w_f v(y_f)``, the weights with which the discrete CLS transport
moves the ``psi = 1/2`` crossing (:func:`cases.capwave_jacobian.kinematic_weights`, faces ``y_f = f h`` about
the interface on a face). ``v(y)`` depends on the eigenvector, so ``s`` is found by Newton on the
determinant with the weights applied to the mode's own profile.

    python -m cases.capwave_kinematic_theory
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def _matrix(s, k, sigma, rho, nu, yw):
    """4x4 system for (A, B, C, D); ``yw = (y_f, w_f)``: the kinematic gain enters the normal-stress row."""
    mu = rho * nu
    m = cmath.sqrt(k * k + s / nu)
    if m.real < 0:
        m = -m
    i = 1j
    y, w = yw

    def vprof(coef):  # v at the faces for unit A, B, C, D
        A, B, C, D = coef
        up = y >= 0
        return np.where(up, -k * A * np.exp(-k * y) - i * k * B * np.exp(-m * y),
                        k * C * np.exp(k * y) - i * k * D * np.exp(m * y))

    # eta = sum w v(y_f) / s, linear in (A, B, C, D): row of coefficients
    kin = np.array([np.sum(w * vprof(e)) for e in np.eye(4)])
    M = np.array([
        [i * k, -m, -i * k, -m],
        [-k, -i * k, -k, i * k],
        [-2 * i * k * k, m * m + k * k, -2 * i * k * k, -(m * m + k * k)],
        [rho * s + 2 * mu * k * k, 2 * i * mu * k * m, -(rho * s + 2 * mu * k * k), 2 * i * mu * k * m],
    ], dtype=complex)
    M[3] = M[3] + sigma * k ** 3 / s * kin / (-k)  # sigma k^2 eta with eta = kin . coef / s (v = -k A ... convention)
    return M


def growth(k, sigma, rho, nu, yw, s0):
    s = s0
    for _ in range(100):
        f = np.linalg.det(_matrix(s, k, sigma, rho, nu, yw))
        ds = 1e-7 * max(abs(s), 1.0)
        df = (np.linalg.det(_matrix(s + ds, k, sigma, rho, nu, yw)) - f) / ds
        step = f / df
        s = s - step
        if abs(step) < 1e-12 * abs(s):
            break
    return s


def main() -> None:
    from cases import capillary_wave as cw
    from cases.capwave_jacobian import CapwaveMap, kinematic_weights
    from cases.capwave_modes import growth as exact_growth

    K = 2 * math.pi
    for n, f in ((32, 0.5), (32, 1.0), (32, 2.0), (64, 0.5), (64, 1.0), (64, 2.0)):
        nu = cw.MU * f
        s = exact_growth(K, 1.0, 1.0, nu)
        cm = CapwaveMap(n, f, curvature="exact")
        yf, w = kinematic_weights(cm.tp)
        sharp = growth(K, 1.0, 1.0, nu, (np.array([0.0]), np.array([1.0])), s)
        sh = growth(K, 1.0, 1.0, nu, (yf, w), s)
        print(f"N={n} x{f}: delta/h {math.sqrt(2 * nu / math.sqrt(K ** 3 / 2)) / (2.0 / n):.2f}  check (w = delta_0) "
              f"{sharp.real / s.real - 1:+.1e}  theory damping {sh.real / s.real - 1:+.4f} frequency {sh.imag / s.imag - 1:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()

"""Normal mode of a capillary wave between two identical viscous fluids (unbounded, no gravity).

Upper fluid ``phi = A e^{-ky}``, ``psi = B e^{-my}``; lower ``phi = C e^{ky}``, ``psi = D e^{my}`` (times
``e^{ikx + st}``), ``m^2 = k^2 + s / nu``; velocity continuous, shear stress continuous, normal stress jump
``sigma k^2 eta`` with ``s eta = v(0)``, pressure ``-rho s phi``. ``growth(k, sigma, rho, nu)`` returns the
complex ``s`` (Newton from the inviscid ``i sqrt(sigma k^3 / 2 rho)``).
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def _det(s, k, sigma, rho, nu):
    mu = rho * nu
    m = cmath.sqrt(k * k + s / nu)
    if m.real < 0:
        m = -m
    i = 1j
    M = np.array([
        [i * k, -m, -i * k, -m],
        [-k, -i * k, -k, i * k],
        [-2 * i * k * k, m * m + k * k, -2 * i * k * k, -(m * m + k * k)],
        [rho * s + 2 * mu * k * k + sigma * k ** 3 / s, 2 * i * mu * k * m + i * sigma * k ** 3 / s * 1.0,
         -(rho * s + 2 * mu * k * k), 2 * i * mu * k * m],
    ], dtype=complex)
    return np.linalg.det(M)


def growth(k: float, sigma: float, rho: float, nu: float) -> complex:
    s = 1j * math.sqrt(sigma * k ** 3 / (2 * rho))
    for _ in range(100):
        f = _det(s, k, sigma, rho, nu)
        ds = 1e-7 * max(abs(s), 1.0)
        df = (_det(s + ds, k, sigma, rho, nu) - f) / ds
        step = f / df
        s = s - step
        if abs(step) < 1e-13 * abs(s):
            break
    return s


def period_decay(k: float, sigma: float, rho: float, nu: float):
    """Period and decay per period of |a| (half the oscillation period between peaks), in omega_0 t."""
    s = growth(k, sigma, rho, nu)
    w0 = math.sqrt(sigma * k ** 3 / (2 * rho))
    half = math.pi / s.imag  # |a| peaks every half period
    return half * w0, math.exp(s.real * half)

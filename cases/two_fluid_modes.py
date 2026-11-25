"""Normal mode of a perturbed flat interface between two viscous fluids (unbounded, with gravity), docs/notes/55.

Lower fluid 1 (``y < 0``: ``rho1, mu1``), upper fluid 2 (``rho2, mu2``), gravity ``-g`` along ``y``. With
``u = grad(phi) + curl(psi e_z)`` (``u = phi_x + psi_y``, ``v = phi_y - psi_x``) and the factor
``e^{ikx + st}``::

    upper: phi = A e^{-ky}, psi = B e^{-m2 y};   lower: phi = C e^{ky}, psi = D e^{m1 y};   m_k^2 = k^2 + s/nu_k

Conditions at ``y = 0``: continuous ``u``, ``v`` and shear stress ``mu (u_y + v_x)``; normal stress
``(p1 - 2 mu1 v1_y) - (p2 - 2 mu2 v2_y) = sigma k^2 eta`` with the pressures ``p_k = -rho_k (s phi + g eta)``
and the kinematic condition ``s eta = v(0)``. Inviscid limits: ``s^2 = -sigma k^3 / (rho1 + rho2)`` (capillary
wave) and ``s^2 = ((rho2 - rho1) g k - sigma k^3) / (rho1 + rho2)`` (Rayleigh--Taylor, heavy fluid on top).

``mode(k, sigma, rho1, mu1, rho2, mu2, g)`` returns the complex ``s`` (Newton from the inviscid root).

    python -m cases.two_fluid_modes      # the paper's capillary wave and Rayleigh--Taylor settings
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def _matrix(s, k, sigma, rho1, mu1, rho2, mu2, g):
    i = 1j
    m1 = cmath.sqrt(k * k + s * rho1 / mu1) if mu1 else None
    m2 = cmath.sqrt(k * k + s * rho2 / mu2) if mu2 else None
    if m1 is not None and m1.real < 0:
        m1 = -m1
    if m2 is not None and m2.real < 0:
        m2 = -m2
    T = sigma * k * k + (rho1 - rho2) * g
    return np.array([
        [i * k, -m2, -i * k, -m1],
        [-k, -i * k, -k, i * k],
        [-2 * i * k * k * mu2, mu2 * (m2 * m2 + k * k), -2 * i * k * k * mu1, -mu1 * (m1 * m1 + k * k)],
        [rho2 * s + 2 * mu2 * k * k, 2 * i * mu2 * k * m2, -rho1 * s - 2 * mu1 * k * k - T * k / s,
         2 * i * mu1 * k * m1 + T * i * k / s],
    ], dtype=complex)


def _det(s, *args):
    return np.linalg.det(_matrix(s, *args))


def inviscid(k, sigma, rho1, rho2, g=0.0) -> complex:
    s2 = ((rho2 - rho1) * g * k - sigma * k ** 3) / (rho1 + rho2)
    return cmath.sqrt(s2) if s2 >= 0 else 1j * math.sqrt(-s2)


def mode(k: float, sigma: float, rho1: float, mu1: float, rho2: float, mu2: float, g: float = 0.0,
         guess: complex | None = None) -> complex:
    """Complex growth rate ``s`` (``Re s`` the damping/growth, ``Im s`` the angular frequency)."""
    if not (mu1 > 0 and mu2 > 0):
        raise ValueError("both viscosities must be positive (use inviscid() otherwise)")
    args = (k, sigma, rho1, mu1, rho2, mu2, g)
    s = guess if guess is not None else inviscid(k, sigma, rho1, rho2, g)
    if abs(s) == 0:
        s = 1e-3
    for _ in range(200):
        f = _det(s, *args)
        ds = 1e-7 * max(abs(s), 1e-6)
        step = f / ((_det(s + ds, *args) - f) / ds)
        s = s - step
        if abs(step) < 1e-13 * abs(s):
            break
    return s


def main() -> None:
    # the paper's capillary wave (14a): water / air, lambda = 10 mm, T = 0.046742983863 s
    k = 2 * math.pi / 0.01
    s = mode(k, 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    T = 0.046742983863
    s0 = inviscid(k, 0.0728, 998.2, 1.204)
    print(f"capillary wave: inviscid omega {s0.imag:.6f}; viscous s = {s.real:.6f} {s.imag:+.6f}i")
    print(f"  amplitude ratio after the paper's period: normal mode {math.exp(s.real * T):.6f} "
          f"(paper 0.793553; 2 nu k^2 estimate {math.exp(-2 * 1.002e-3 / 998.2 * k * k * T):.6f})")
    # the paper's Rayleigh--Taylor (14d): oil (950, 5e-3) below water (1000, 1e-3), sigma 0.04, k = 4 pi
    k = 4 * math.pi
    s = mode(k, 0.040, 950.0, 5e-3, 1000.0, 1e-3, 9.81)
    print(f"Rayleigh-Taylor: inviscid {inviscid(k, 0.040, 950.0, 1000.0, 9.81).real:.6f}, sigma = 0: "
          f"{inviscid(k, 0.0, 950.0, 1000.0, 9.81).real:.6f}; viscous {s.real:.6f} {s.imag:+.2e}i "
          f"(paper gamma = 1.78)")


if __name__ == "__main__":
    main()

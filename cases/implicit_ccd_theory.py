"""1-D linear theory of implicit space-time compact schemes for q_t + u q_x = 0 (docs/notes/74): the step
q^{n+1} = P(-Z)/P(Z)-type Pade approximants of e^{-Z}, Z = u dt d/dx, with Z -> C d1(theta) and Z^2 -> C^2 d2(theta)
(the independent CCD second derivative) or C^2 d1(theta)^2 (the first derivative squared); d1, d2 the symbols of the
implemented periodic CCD matrices. For each approximant and CFL number: max over theta of |G| - 1 (stability), and
the amplitude and phase-speed ratio at theta = pi/2.

    python -m cases.implicit_ccd_theory
"""

from __future__ import annotations

import numpy as np

PADE = {  # numerator(Z), denominator(Z) as polynomial coefficients in Z (constant first): e^{-Z} ~ N(Z)/M(Z)
    "[1/1] CN": ((1.0, -1.0 / 2.0), (1.0, 1.0 / 2.0)),
    "[1/2] L-stable": ((1.0, -1.0 / 3.0), (1.0, 2.0 / 3.0, 1.0 / 6.0)),
    "[2/2]": ((1.0, -1.0 / 2.0, 1.0 / 12.0), (1.0, 1.0 / 2.0, 1.0 / 12.0)),
}


def symbols(n: int = 512):
    from basilisk_cupy.discretization.ccd import ccd_matrices

    d1, d2 = ccd_matrices(n, 1.0, "periodic", "periodic")[:2]
    theta = 2 * np.pi * np.arange(1, n // 2 + 1) / n
    v = np.exp(1j * np.outer(np.arange(n), theta))
    return theta, (d1 @ v)[0] / v[0], (d2 @ v)[0] / v[0]


def poly(coeffs, Z1, Z2):
    """sum c_k Z^k with Z -> Z1 and Z^2 -> Z2 (only up to Z^2 occurs)."""
    out = coeffs[0] + (coeffs[1] * Z1 if len(coeffs) > 1 else 0.0)
    if len(coeffs) > 2:
        out = out + coeffs[2] * Z2
    return out


def main() -> None:
    theta, d1, d2 = symbols()
    k = int(np.argmin(np.abs(theta - np.pi / 2)))
    for name, (num, den) in PADE.items():
        for second in ("d2 (CCD)", "d1^2"):
            rows = []
            for C in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
                Z1 = C * d1
                Z2 = C * C * (d2 if second.startswith("d2") else d1 * d1)
                G = poly(num, Z1, Z2) / poly(den, Z1, Z2)
                exact_phase = -C * theta[k]
                amp = abs(G[k])
                speed = np.angle(G[k]) / exact_phase if exact_phase != 0 else 1.0
                rows.append(f"C {C:3.1f}: {np.abs(G).max() - 1:+.1e} amp {amp:.3f} speed {speed:.3f}")
            print(f"{name:15s} Z^2 -> {second:9s} | " + " | ".join(rows), flush=True)


if __name__ == "__main__":
    main()

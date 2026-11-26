"""Linear stability of BCG-type face predictors with CCD derivatives (docs/notes/72, P1): 1-D periodic constant
advection, cell values q_j, face value from the upwind cell q_f = q + (1/2)(1 - C) q' + c2(C) q'' (h = 1), update
q_j^{n+1} = q_j - C (q_{f,j+1/2} - q_{f,j-1/2}); amplification G(theta, C) = 1 - C (1 - e^{-i theta}) r(theta, C)
with r = 1 + (1 - C)/2 d1(theta) + c2(C) d2(theta), d1 / d2 the symbols of the derivative operators (from the
implemented CCD matrices). Variants: BCG (centred slope, c2 = 0), CCD slope (c2 = 0), the point-value Taylor term
c2 = (1 - C)^2/8 (the first ccd2 prototype) and the mean-preserving parabola c2 = (1 - C)(1 - 2C)/12.

    python -m cases.ccd_bcg_stability
"""

from __future__ import annotations

import numpy as np


def symbols(n: int = 256):
    from basilisk_cupy.discretization.ccd import ccd_matrices

    d1, d2 = ccd_matrices(n, 1.0, "periodic", "periodic")[:2]
    theta = 2 * np.pi * np.arange(n) / n
    v = np.exp(1j * np.outer(np.arange(n), theta))  # columns: Fourier modes
    s1 = (d1 @ v)[0] / v[0]
    s2 = (d2 @ v)[0] / v[0]
    return theta, s1, s2


def amplification(theta, d1, d2, C, c2):
    r = 1.0 + 0.5 * (1.0 - C) * d1 + c2 * d2
    return 1.0 - C * (1.0 - np.exp(-1j * theta)) * r


def main() -> None:
    theta, s1, s2 = symbols()
    centred = 1j * np.sin(theta)
    variants = {
        "BCG (centred slope)": (centred, 0.0 * s2, lambda C: 0.0),
        "CCD slope": (s1, s2, lambda C: 0.0),
        "CCD + (1-C)^2/8 q'' (point value)": (s1, s2, lambda C: (1 - C) ** 2 / 8),
        "CCD + (1-C)(1-2C)/12 q'' (mean parabola)": (s1, s2, lambda C: (1 - C) * (1 - 2 * C) / 12),
    }
    Cs = np.linspace(0.01, 1.0, 100)
    for name, (d1, d2, c2) in variants.items():
        worst = [(float(np.abs(amplification(theta, d1, d2, C, c2(C))).max()) - 1.0, float(C)) for C in Cs]
        stable = [C for g, C in worst if g <= 1e-12]
        g_max, C_at = max(worst)
        lim = max(stable) if stable else 0.0
        contiguous = 0.0
        for g, C in worst:
            if g > 1e-12:
                break
            contiguous = C
        print(f"{name:44s} max_theta |G| - 1 over C: {g_max:+.2e} (at C = {C_at:.2f}); stable for C <= {contiguous:.2f}"
              f"; stable C values: {len(stable)}/100")


if __name__ == "__main__":
    main()

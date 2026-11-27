"""docs/notes/102 (V6-1, V6-2): the full viscous stress tensor in 2-D with the tensor product of the 1-D
minimum-viscous-energy reconstructions (the user's proposal), single phase, periodic: Fourier symbols.

1-D reconstruction of the averages q_i = e^{i theta i} (rho = mu = 1, cell [0, h], s = x/h): face stresses
tau = H^{-1} D q, H = h tridiag(1/6, 2/3, 1/6); in cell 0

    X(s) = 1 + tau_{-1} (w_l - h/3) + tau_0 (w_r - h/6),  w_l = h (s - s^2/2), w_r = h s^2/2,  X' = (1 - s) tau_{-1} + s tau_0

with tau_0 = T, tau_{-1} = T e^{-i theta}, T = (e^{i theta} - 1) / (h (2/3 + cos(theta)/3)). Symbols per cell:
b = int |X|^2, a = int |X'|^2, c = int conj(X) X'. For u = alpha X(x) Y(y), v = beta X Y the energy
int mu (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2) gives the 2 x 2 Hermitian

    K = mu [[2 a_x b_y + b_x a_y, c_x conj(c_y)], [conj(c_x) c_y, a_x b_y + 2 b_x a_y]],   G = b_x b_y I   (method C)

and method D replaces G by the tensor-product PCCD Gram g_A(theta_x) g_A(theta_y) h^2. The discrete decay rates are
the eigenvalues of G^{-1} K; the exact ones mu |k|^2 (divergence-free) and 2 mu |k|^2 (curl-free; grad div adds mu).

    python -m cases.viscous_2d
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, symbols


def symbols_1d(theta, h):
    T = (np.exp(1j * theta) - 1.0) / (h * (2 / 3 + math.cos(theta) / 3))
    tl, tr = T * np.exp(-1j * theta), T
    s = XG
    X = 1 + tl * (h * (s - s * s / 2) - h / 3) + tr * (h * s * s / 2 - h / 6)
    dX = (1 - s) * tl + s * tr
    b = h * np.sum(WG * np.abs(X) ** 2)
    a = h * np.sum(WG * np.abs(dX) ** 2)
    c = h * np.sum(WG * np.conj(X) * dX)
    return a, b, c


def rates(tx, ty, h, mu=1.0, method="C"):
    ax, bx, cx = symbols_1d(tx, h)
    ay, by, cy = symbols_1d(ty, h)
    K = mu * np.array([[2 * ax * by + bx * ay, cx * np.conj(cy)], [np.conj(cx) * cy, ax * by + 2 * bx * ay]])
    if method == "C":
        g = bx * by
    else:
        g = h * h * symbols(tx, 0.37)["gram"].real * symbols(ty, 0.37)["gram"].real
    ev = np.linalg.eigvalsh(K / g)
    return np.sort(ev.real)


def scan(N, method="C", mu=1.0, modes=((1, 0), (1, 1), (2, 1), (3, 2), (1, 3))):
    """Relative errors of the two rates for the wavevectors k = 2 pi (m, n) on [0, 1]^2 with N x N cells."""
    h = 1.0 / N
    out = []
    for m, n in modes:
        tx, ty = 2 * math.pi * m * h, 2 * math.pi * n * h
        k2 = (2 * math.pi) ** 2 * (m * m + n * n)
        r = rates(tx, ty, h, mu, method)
        out.append({"mode": [m, n], "div_free": float(abs(r[0] - mu * k2) / (mu * k2)),
                    "curl_free": float(abs(r[1] - 2 * mu * k2) / (2 * mu * k2))})
    return out


def isotropy(N, kh=0.5, method="C", angles=12):
    """Relative errors of both rates for |k| h = kh at angles 0 .. 90 degrees (isotropy of the discrete operator)."""
    h = 1.0 / N
    out = []
    for t in np.linspace(0, math.pi / 2, angles):
        tx, ty = kh * math.cos(t), kh * math.sin(t)
        k2 = (kh / h) ** 2
        r = rates(tx, ty, h, 1.0, method)
        out.append({"angle_deg": float(math.degrees(t)), "div_free": float((r[0] - k2) / k2),
                    "curl_free": float((r[1] - 2 * k2) / (2 * k2))})
    return out


def stiffness(N, method="C"):
    """max over the grid wavevectors of the largest rate x h^2 / mu."""
    h = 1.0 / N
    th = np.linspace(-math.pi, math.pi, 65)
    return float(max(rates(a, b, h, 1.0, method)[1] for a in th for b in th if (a, b) != (0.0, 0.0)) * h * h)


def main() -> None:
    res = {"scan": {}, "isotropy": {}, "stiffness": {}}
    for method in ("C", "D"):
        res["scan"][method] = {N: scan(N, method) for N in (16, 32, 64, 128)}
        res["isotropy"][method] = isotropy(32, 0.5, method)
        res["stiffness"][method] = stiffness(32, method)
    json.dump(res, open("docs/figures/viscous_2d.json", "w"), indent=1)
    for method in ("C", "D"):
        for N, rows in res["scan"][method].items():
            print(method, N, " ".join(f"{r['mode']}: {r['div_free']:.2e}/{r['curl_free']:.2e}" for r in rows))
        iso = res["isotropy"][method]
        print(method, "isotropy |k|h = 0.5: div-free", f"{min(r['div_free'] for r in iso):+.2e} .. {max(r['div_free'] for r in iso):+.2e}",
              "curl-free", f"{min(r['curl_free'] for r in iso):+.2e} .. {max(r['curl_free'] for r in iso):+.2e}",
              "stiffness", f"{res['stiffness'][method]:.2f}")


if __name__ == "__main__":
    main()

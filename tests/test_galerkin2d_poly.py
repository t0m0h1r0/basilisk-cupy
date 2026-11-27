"""docs/notes/91: the polynomial-approximated Galerkin map keeps energy, free stream and momentum exactly."""

import numpy as np

from cases.galerkin2d import Reconstruction2D, assemble
from cases.galerkin2d_circle import RK, Circle, poly_transport


def test_polynomial_galerkin_structure():
    N = 16
    U, dt = (1.0, 0.5), 0.37 / 16
    g0 = Circle(cx=0.5313, cy=0.5157)
    r0 = Reconstruction2D(N, g0, hinge="exact", workers=4, **RK)
    r1 = Reconstruction2D(N, g0.moved(U, dt), hinge="exact", workers=4, **RK)
    G0, G1, C, _ = assemble(N, r0, r1, U, dt, 4)
    m = r1.mass
    s = 1 / np.sqrt(m)
    assert np.linalg.eigvalsh(G1 * s[:, None] * s[None, :]).min() > 1 - 1e-10     # G1 >= M (shift)
    one = np.ones(N * N)
    g = np.linalg.eigvalsh(G0).max()
    for k in (2, 4, 6):
        T = poly_transport(G1, C, m, k)
        D = T.T @ G1 @ T - G0
        assert np.linalg.eigvalsh(0.5 * (D + D.T)).max() < 1e-14 * g          # energy cannot grow
        assert abs(T @ one - 1).max() < 1e-10                                    # free stream
        assert abs(one @ G1 @ T - one @ C).max() < 1e-12 * m.max()               # total momentum
    # the single root at 1 alone: r = (1 - lam)^2 exceeds 2 exactly when lam > 1 + sqrt 2, and then the energy grows
    lmax = np.linalg.eigvalsh(G1 * s[:, None] * s[None, :]).max()
    T = poly_transport(G1, C, m, 1)
    D = T.T @ G1 @ T - G0
    grows = np.linalg.eigvalsh(0.5 * (D + D.T)).max() > 1e-12 * g
    assert grows == (lmax > 1 + np.sqrt(2))

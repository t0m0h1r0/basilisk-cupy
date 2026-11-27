"""docs/notes/89 (P1-2): a circular interface, the exact signed-distance hinge against the tangent-line hinge."""

import numpy as np

from cases.galerkin2d_circle import Circle, exactness


def test_circle_quadrature_is_exact_for_the_area():
    """The cut-cell quadrature (cosine-mapped x Gauss, y Gauss) reproduces the disc area and the cell areas."""
    g = Circle()
    N = 24
    h = 1.0 / N
    heavy, total = 0.0, 0.0
    for iy in range(N):
        for ix in range(N):
            for X, w, hv in g.pieces(ix * h, iy * h, (ix + 1) * h, (iy + 1) * h):
                total += w.sum()
                heavy += w.sum() if hv else 0.0
    assert abs(total - 1.0) < 1e-12
    assert abs(heavy - np.pi * g.R0 ** 2) < 1e-12


def test_exact_distance_hinge_is_exact_and_tangent_hinge_is_not():
    """P0 + max(d/h, 0) P1 with d the exact signed distance: reproduced by the exact hinge; the osculating parabola
    misses it by O(h^2) in cell units, the tangent line by O(h)."""
    assert exactness(24, "exact", workers=4) < 1e-12
    q, t = exactness(24, "quadratic", workers=4), exactness(24, "tangent", workers=4)
    assert 1e-4 < q < t / 4

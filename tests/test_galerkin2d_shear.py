"""docs/notes/90 (P1-3): a disc deformed by an unsteady shear with an exact flow map."""

import numpy as np

from cases.galerkin2d import Reconstruction2D
from cases.galerkin2d_circle import RK
from cases.galerkin2d_shear import ShearedDisc, assemble, exactness


def test_sheared_quadrature_is_exact_for_the_areas():
    """y outside (u^2 substitution at the tangencies), x inside: the disc area is preserved by the map."""
    for S, cy in ((0.2417, 0.4813), (0.1, 0.53)):
        g = ShearedDisc(S=S, cy=cy)
        N = 24
        h = 1.0 / N
        tot = hv = 0.0
        for iy in range(N):
            for ix in range(N):
                for _, w, v in g.pieces(ix * h, iy * h, (ix + 1) * h, (iy + 1) * h):
                    tot += w.sum()
                    hv += w.sum() if v else 0.0
        assert abs(tot - 1.0) < 1e-12
        assert abs(hv - np.pi * g.R0 ** 2) < 1e-12


def test_structure_on_the_curved_common_partition():
    """Moved old cells with curved faces: free stream, G 1 = m, positive semi-definite block, identity at dS = 0."""
    N = 16
    dS = 0.9 * 0.37 / N
    g0 = ShearedDisc(S=0.05)
    g1 = g0.at(0.05 + dS)
    r0 = Reconstruction2D(N, g0, hinge="exact", workers=4, **RK)
    r1 = Reconstruction2D(N, g1, hinge="exact", workers=4, **RK)
    G0, G1, C, Cg = assemble(N, r0, r1, dS, workers=4)
    one = np.ones(N * N)
    assert abs(np.linalg.solve(G1, C) @ one - 1).max() < 1e-10
    assert abs(Cg.sum(1) / r1.mass - 1).max() < 1e-12
    assert abs(G0 @ one - r0.mass).max() < 1e-12 * r0.mass.max()
    ev = np.linalg.eigvalsh(np.block([[G0, C.T], [C, G1]]))
    assert ev.min() > -1e-14 * ev.max()
    G00, _, C0, _ = assemble(N, r0, r0, 0.0, workers=4)
    assert abs(np.linalg.solve(G00, C0) - np.eye(N * N)).max() < 1e-10


def test_defining_function_hinge_is_exact():
    """P0 + max(phi/h, 0) P1, phi = |Phi^{-1}(x) - c| - R0 (not a distance): reproduced by the phi hinge."""
    assert exactness(24, "exact", S=0.15, workers=4) < 1e-12
    assert exactness(24, "tangent", S=0.15, workers=4) > 1e-3

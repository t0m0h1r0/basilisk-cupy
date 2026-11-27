"""docs/notes/87: the 2-D geometry-consistent Galerkin transport (oblique periodic band, exact geometry)."""

from cases.galerkin2d import exactness, free_stream


def test_structure_on_the_common_partition():
    """Free stream, total momentum, G1 1 = m1 and the positive semi-definite block [[G0, C^T], [C, G1]]."""
    r = free_stream(16)
    assert r["free_stream"] < 1e-9 and r["remap_free_stream"] < 1e-9
    assert r["momentum"] < 1e-12 and r["mass_G1"] < 1e-10
    assert r["block_min_eig"] > -1e-14


def test_piecewise_polynomials_are_exact():
    """P0 + max(xi, 0) P1 (continuous, normal gradient jump) is reproduced near the interface."""
    assert exactness(24) < 1e-9


def test_interface_window_is_rank_safe_parallel_to_the_grid():
    """docs/notes/88: an interface parallel to the grid on a cell boundary: the 7 x 7 cell-centred window cannot
    reproduce the piecewise quartics, the interface-centred window of radius 5.5 can."""
    import cases.galerkin2d as g

    G = g.Geometry
    try:
        g.Geometry = lambda a=1, b=1, zA=5 / 24, zB=0.647, ratio=1e3, d=0.0: G(a, b, zA, zB, ratio, d)
        assert exactness(24, a=0, b=1) > 1e-3
        assert exactness(24, a=0, b=1, window="interface", switch="angle", diff=True, R=5.5) < 1e-10
    finally:
        g.Geometry = G

"""docs/notes/90: the velocity-difference shift is the unconstrained least squares plus the constant (note 85), not the
least squares constrained to the cell mean (notes 88-89)."""

from cases.galerkin2d_shiftfix import amplification, equivalence


def test_split_fit_equals_the_plain_shift():
    r = equivalence(N=16)
    assert r["shift_vs_plain"] < 1e-8
    assert r["constrained_vs_lsq"] < 1e-5          # the old form is the constrained least squares


def test_no_amplification_at_density_ratio_one_million():
    """Lambda = max sum_j |phi_ij| next to a thin heavy slice (zeta = 100): ~26 with the shift, ~400 constrained."""
    s = amplification(24, 0, 100.0, 1e6, "shift", workers=4)["Lambda"]
    c = amplification(24, 0, 100.0, 1e6, "constrained", workers=4)["Lambda"]
    assert s < 40 and c > 300

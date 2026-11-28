"""docs/notes/115 (R0-P2): the total momentum with the pressure included and the hydrostatic balance of the
cell-centred solver as it is. The failing combinations are kept here as such."""

import pytest

from cases.r0p2_pressure_momentum import hydrostatic, momentum, one_d


def test_one_d_identity_arithmetic_mean_only():
    o = one_d()
    assert o["arithmetic"]["dP"] < 1e-12
    for k in ("harmonic", "gfm, theta = 0.3"):
        assert o[k]["dP"] > 1.0 and abs(o[k]["dP"] - o[k]["formula"]) < 1e-9 * o[k]["dP"]


@pytest.mark.parametrize("kind", ["smooth", "mean"])
def test_total_momentum_with_pressure_is_conserved_with_the_arithmetic_cell_coefficient(kind):
    o = momentum(kind, True, 32, t_end=0.25, samples=(0.25,))
    assert o["failed"] is None and o["momentum_change"] < 1e-9


def test_total_momentum_is_not_conserved_with_the_gfm_cell_coefficient_or_the_plain_transport():
    assert momentum("gfm", True, 32, t_end=0.25, samples=(0.25,))["momentum_change"] > 1e-3
    assert momentum("smooth", False, 32, t_end=0.25, samples=(0.25,))["momentum_change"] > 1e-4


@pytest.mark.parametrize("kind", ["smooth", "mean", "gfm"])
def test_oblique_hydrostatic_balance_holds_with_the_reduced_form_only(kind):
    red = hydrostatic(kind, 26.565, "reduced", 32, t_end=0.25, samples=(0.25,))["history"][-1]
    assert red["cells"] < 1e-8 and red["faces"] < 1e-8 and red["df"] < 1e-8
    pla = hydrostatic(kind, 26.565, "plain", 32, t_end=0.25, samples=(0.01, 0.25))["history"]
    assert pla[-1]["cells"] > 1e-4 and pla[-1]["cells"] > 5 * pla[0]["cells"]   # not balanced, and growing


def test_parallel_interface_is_balanced_with_both_forms():
    for grav in ("plain", "reduced"):
        h = hydrostatic("smooth", 0.0, grav, 32, t_end=0.25, samples=(0.25,))["history"][-1]
        assert h["cells"] < 1e-8 and h["faces"] < 1e-8


def test_interface_reaching_a_wall_within_a_cell_of_a_corner_is_not_balanced_on_a_coarse_grid():
    """Recorded as it is: 45 degrees through (0.5, 0.5137) reaches the walls 0.0137 from two corners (h = 1/32)."""
    bad = hydrostatic("smooth", 45.0, "reduced", 32, t_end=0.25, samples=(0.25,))["history"][-1]
    assert 1e-4 < bad["cells"] < 1e-1
    ok = hydrostatic("smooth", 45.0, "reduced", 32, t_end=0.25, samples=(0.25,), offset=0.1171)["history"][-1]
    assert ok["cells"] < 1e-8


def test_momentum_budget_closes_with_the_walls_pressure_force():
    """B's request on PR #157: closed box, the wall force read from the wall-adjacent cells' pressure."""
    from cases.r0p2_pressure_momentum import wall_budget

    o = wall_budget("smooth", 32, t_end=0.25)
    assert o["budget_residual"] < 1e-9 and o["wall_impulse"] > 1e-3          # closes, and is not trivially zero
    assert wall_budget("gfm", 32, t_end=0.25)["budget_residual"] > 1e-3      # recorded: not with the GFM coefficient


def test_total_momentum_is_not_conserved_with_filtered_face_densities():
    # audit on PR #157: FILTERED makes the face density a mean of the smoothed f, not of the cells' densities
    assert momentum("smooth-filtered", True, 32, t_end=0.25, samples=(0.25,))["momentum_change"] > 1e-3

"""docs/notes/120 (A-2 of #145, R0-P2): the unbalanced hydrostatic cases of docs/notes/115 come from the walls'
mirrored ghost cells in the height functions (an implicit 90 degree contact angle)."""

import pytest

from cases.r0p2_corner import FAILING, balance, potential

pytestmark = pytest.mark.xdist_group("r0p2_corner")


@pytest.mark.parametrize("geom", FAILING)
def test_potential_is_wrong_only_with_the_mirrored_ghosts(geom):
    assert potential(*geom)["max_phi"] > 1e-4
    assert potential(*geom, mode="exact")["max_phi"] < 1e-14
    assert potential(*geom, mode="exact corners only")["max_phi"] > 1e-4     # not the corner ghosts themselves


def test_passing_geometry_has_zero_potential():
    assert potential(45.0, 0.0137, 128)["max_phi"] < 1e-14 and potential(26.565, 0.2313, 64)["max_phi"] < 1e-14


def test_solver_at_rest_with_the_planes_ghosts():
    m = balance(45.0, 0.0137, 32, "mirror", t_end=0.1, samples=(0.1,))["history"][-1]["cells"]
    p = balance(45.0, 0.0137, 32, "plane", t_end=0.1, samples=(0.1,))["history"][-1]["cells"]
    assert m > 1e-4 and p < 1e-12


def test_round_off_slivers_move_the_heights_away_from_walls():
    from cases.r0p2_corner import sliver_periodic

    quiet = sliver_periodic((1, 2), 0.0137, 32, steps=10, noise=0.0)
    noisy = sliver_periodic((1, 2), 0.0137, 32, steps=10)
    assert quiet["devmax_cells"] < 1e-12 and quiet["slivers_max"] == 0
    assert noisy["slivers_max"] > 0 and noisy["devmax_cells"] > 1.0

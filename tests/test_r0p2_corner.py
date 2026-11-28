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


def test_no_slivers_away_from_walls_with_a_step_bound():
    """#193 (correction 2 of docs/notes/120): with the step bound DT = 0.5 / N no sliver appears in a periodic band and
    the heights stay on the lines; the first version's jumps came from a run without a bound (dt grew to ~4e13)."""
    from cases.r0p2_corner import sliver_periodic

    bounded = sliver_periodic((1, 2), 0.0137, 32, steps=200)
    assert bounded["slivers_max"] == 0 and bounded["devmax_cells"] < 1e-12
    unbounded = sliver_periodic((1, 2), 0.0137, 32, steps=10, dt_max=None)        # the artefact, kept as such
    assert unbounded["dt_last"] > 1e10 and unbounded["slivers_max"] > 0

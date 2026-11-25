"""One velocity field with a consistent projection (docs/notes/48)."""

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from cases.single_velocity import interp_div, projection

pytestmark = pytest.mark.xdist_group("single_velocity")


def test_cell_projection_is_an_orthogonal_projector():
    grid = CartesianGrid((12, 12), length=(2.0, 2.0), origin=(0.0, -1.0))
    I, D, _ = interp_div(grid)
    Dc = D @ I
    P, S = projection(Dc, -Dc.T)
    assert np.abs(P @ P - P).max() < 1e-12 and np.abs(P - P.T).max() < 1e-12
    assert np.abs(Dc @ P).max() < 1e-10
    ev = np.sort(np.linalg.eigvalsh(-S))
    assert ev[0] < 1e-10 * ev[-1] < ev[1]  # one null mode (the constant) with walls


def test_cn_on_the_exact_cell_projection_is_second_order():
    from cases import capillary_wave as cw
    from cases.capwave_normal_mode import K, mode
    from cases.single_velocity import operators, semi_discrete, setup, step_eigen

    cm = setup(16, 4.0, "fd2")
    ops = operators(cm, "fd2")
    s, _ = mode(K, 1.0, 1.0, cw.MU * 4.0)
    J, _, _ = semi_discrete(ops, cw.MU * 4.0, "cell")
    e = [step_eigen(J, s, cw.STEP / d, "cn").real for d in (1, 2, 4)]
    r = (e[2] - e[1]) / (e[1] - e[0])
    assert 0.2 < r < 0.3

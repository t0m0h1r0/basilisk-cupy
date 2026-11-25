"""The layered CCD viscosity with walls: non-symmetric, defective zero eigenvalue, energy growth (docs/notes/37)."""

import numpy as np
import pytest

from cases.layered_energy import analyse, one_d, stress_matrix

pytestmark = pytest.mark.xdist_group("layered_energy")


def test_one_d_closures():
    r = one_d(32)
    assert r["periodic D1 D1"]["norm"] == pytest.approx(1.0, abs=1e-12)
    assert r["walls D1_ext D1_dir (layered)"]["sym_max"] > 1.0 and r["walls D1_ext D1_dir (layered)"]["norm"] > 1.3
    assert r["walls D2_dir (split)"]["norm"] < 1.0 and r["walls -D1_dir^T D1_dir"]["sym_max"] < 0.0


def test_two_d_layered_walls_grow_periodic_do_not():
    walls = analyse(stress_matrix(8, "layered", False))
    per = analyse(stress_matrix(8, "layered", True))
    split = analyse(stress_matrix(8, "split", False))
    assert per["asym"] < 1e-12 and per["norm"] == pytest.approx(1.0, abs=1e-10)
    assert walls["sym_max"] > 1.0 and walls["null"][1] > walls["null"][0]  # Jordan block at 0
    assert walls["energy"][-1] > 10.0 > 1.0 > split["energy"][-1]
    assert np.all(np.diff(split["energy"]) < 0)

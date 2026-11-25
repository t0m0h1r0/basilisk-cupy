"""Case C stage solver and its integrators (docs/notes/47)."""

import pytest

from cases import capillary_wave as cw
from cases.case_c import dispersion

pytestmark = pytest.mark.xdist_group("case_c")


def test_face_kept_cn_is_second_order():
    """N = 16, x 4: with u_f^n as the face base, CN's damping error changes at second order in dt."""
    e = [dispersion(16, 4.0, cw.STEP / d, method="cn", face_base="faces")["damping_err"] for d in (1, 2, 4)]
    r = (e[2] - e[1]) / (e[1] - e[0])
    assert 0.15 < r < 0.35


def test_rebuilding_the_faces_masks_the_order():
    """With the faces rebuilt from the cells each step (CenteredNS), CN and SDIRK2 follow the same curve
    as each other within a few tenths of a point."""
    cn = dispersion(16, 4.0, cw.STEP, method="cn")
    sd = dispersion(16, 4.0, cw.STEP, method="sdirk2")
    assert abs(cn["damping_err"] - sd["damping_err"]) < 0.01

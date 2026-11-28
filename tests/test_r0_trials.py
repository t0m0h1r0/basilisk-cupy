"""docs/notes/123 (#166's bounded trials). Kept as found: E3's band form grows at kappa_L = 10; E1's coefficient
handling does not stop the jump at nearly coincident events, which comes from the face-value pressure source."""

import pytest

from cases.r0_trials import capwave, events, identity, kappa_drop

pytestmark = pytest.mark.xdist_group("r0_trials")


def test_e3_band_form_is_note_113_in_single_phase_and_grows_at_large_kappa():
    assert max(r["difference"] for r in identity("E3band1", flows=("tg",), grids=(32,))["rows"]) == 0.0
    assert kappa_drop("E3band1", 3.0, steps=200)["max_err"] < 1e-8
    assert kappa_drop("E3band1", 10.0, steps=600)["max_err"] > 0.1


def test_event_jump_comes_from_the_pressure_source_not_e1():
    quiet = events("noE1", "tg", None)["err_max"]
    jump = events("noE1", "tg", 1e-7)["err_max"]
    assert jump > 3 * quiet                                       # the source term alone (no E1) jumps
    assert events("base-lib", "tg", 1e-7)["err_max"] < 1.01 * events("base-lib", "tg", None)["err_max"]


def test_e1_midpoint_force_removes_most_of_the_extra_damping():
    e1, mid, base = (capwave(c, 32)["damping"] for c in ("E1", "E1mid", "base-lib"))
    assert e1 - base > 0.1 and 0 < mid - base < 0.01

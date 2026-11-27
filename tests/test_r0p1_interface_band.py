"""docs/notes/117: the half-step source of the face value switched off near the interface (option 4). B's five pass
conditions on small cases."""

import pytest

from cases.r0p1_interface_band import identical
from cases.r0p1_source_feedback import growth, vortex


@pytest.mark.parametrize("mode", ["face-band1", "face-band2"])
def test_band_switch_is_stable_at_high_density_ratios(mode):
    for ratio, shape in ((1e3, "drop"), (1e6, "band")):
        o = growth(ratio, mode, steps=200, shape=shape)
        ref = growth(ratio, "none", steps=200, shape=shape)
        assert not o["unstable"] and o["max"] < 10 * ref["max"] + 1e-12
    assert growth(1e3, "face", steps=120)["unstable"]           # the unswitched source (docs/notes/114)


def test_band_switch_conserves_the_total_momentum():
    o = vortex("face-band1", 32)
    assert o["failed"] is None and o["momentum_change"] < 1e-9 and o["umax"] < 3.0


def test_band_switch_is_the_unswitched_scheme_without_an_interface(tmp_path):
    o = identical(32, "face-band1", str(tmp_path))
    assert o["difference"] == 0.0 and o["error"] < 1e-2

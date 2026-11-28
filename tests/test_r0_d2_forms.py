"""docs/notes/119 (A-1 of #145, audit #162): the forms of the D2 fix. The limited form keeps the free stream and is
identical to the base where it never cuts; the full form and the centroid form lose accuracy (kept as such)."""

import numpy as np
import pytest

from cases.r0_aligned_free_stream import free_stream, identical, kick_d2
from cases.r0_d2_forms import accuracy, cut_share

pytestmark = pytest.mark.xdist_group("r0_d2_forms")


@pytest.mark.parametrize("name,ratio,kw", [("column", 1e3, {}), ("square", 1e6, {}), ("column", 1e3, {"dt": 0.4999 / 32}),
                                           ("column", 1e6, {"U": (1.0, 0.37)})])
def test_limited_and_centroid_forms_keep_the_free_stream(name, ratio, kw):
    for s in ("d1+d2lim", "d1+geo"):
        assert free_stream(name, s, ratio, **kw)["max_err"] < 1e-8


def test_limited_form_never_cuts_at_density_ratio_one_and_matches_the_base():
    for so in (True, False):
        a, b = accuracy("base", 48, second_order=so), accuracy("d1+d2lim", 48, second_order=so)
        assert all(abs(a[k] - b[k]) <= 1e-12 * a[k] for k in ("max", "L2", "L1"))


def test_full_and_centroid_forms_lose_accuracy():
    a = accuracy("base", 48)
    assert accuracy("d1+d2", 48)["L2"] > 2.5 * a["L2"]
    assert accuracy("d1+geo", 48)["L2"] > 1.2 * a["L2"]


def test_limited_form_cuts_only_in_the_aligned_band():
    c = cut_share()
    assert c["vortex"]["dropped"] == 0 and c["vortex"]["partial"] > 1000
    assert 0 < c["aligned band"]["share"] < 0.1


def test_limited_form_one_sweep_kills_the_d2_kick_and_no_interface_is_identical():
    rows = kick_d2(cfl=(0.4999,), c=(0.5,))
    assert abs(rows[0]["gain_base"]) > 100
    assert abs(rows[0]["measured"]["d1+d2lim"]) < 1e-15                 # cut: the kick is gone
    assert abs(rows[0]["measured"]["d1+geo"]) < 2e-6                     # bounded: O(g), not ratio x g
    ref = identical("base", steps=10)
    for s in ("d1+d2lim", "d1+geo"):
        assert np.array_equal(identical(s, steps=10), ref)

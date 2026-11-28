"""docs/notes/122 (A-3 stage 1, #166): every R0-P modification at once (hooks). Kept as found: E3 (Crank-Nicolson,
note 113) breaks the two-phase free stream and the total momentum; without it the combination holds."""

import numpy as np
import pytest

from cases.r0_aligned_free_stream import free_stream
from cases.r0_combined import combined, full
from cases.r0_d2_forms import vortex_budget

pytestmark = pytest.mark.xdist_group("r0_combined")
MU = (1e-2, 1e-2 / 55.0)


def test_single_phase_is_note_113s_corrected_scheme():
    import tempfile

    from cases.r0p1_unsteady_viscous import CONFIGS, NU, run

    with tempfile.TemporaryDirectory() as d:
        run(32, 0.5, flow="asym", nu=NU["asym"], save=d + "/a.npy", t_ends=(0.125,), **CONFIGS["cons/ab+visc+cn"])
        with full():
            run(32, 0.5, flow="asym", nu=NU["asym"], save=d + "/b.npy", t_ends=(0.125,), **CONFIGS["cons"])
        assert np.array_equal(np.load(d + "/a.npy"), np.load(d + "/b.npy"))


def test_e3_breaks_the_two_phase_free_stream_and_momentum_and_the_rest_holds():
    with combined():
        assert free_stream("drop", "d1+d2lim", 1e3, U=(1.0, 0.37), mu=MU, steps=200)["max_err"] > 1.0
    with combined(e3=False):
        assert free_stream("drop", "d1+d2lim", 1e3, U=(1.0, 0.37), mu=MU, steps=200)["max_err"] < 1e-8
    with combined():
        assert vortex_budget("d1+d2lim", n=32, mu=MU, t_end=0.1)["momentum_change"] > 1e-4
    with combined(e3=False):
        assert vortex_budget("d1+d2lim", n=32, mu=MU, t_end=0.1)["momentum_change"] < 1e-9
    with combined(e3_form="momentum"):                       # the explicit half as a momentum: conserving again
        assert vortex_budget("d1+d2lim", n=32, mu=MU, t_end=0.1)["momentum_change"] < 1e-9


def test_inviscid_free_stream_with_everything():
    for name, kw in (("column", {}), ("column", {"dt": 0.4999 / 32}), ("square", {"U": (1.0, 0.37)})):
        with combined():
            assert free_stream(name, "d1+d2lim", 1e3, **kw)["max_err"] < 1e-8

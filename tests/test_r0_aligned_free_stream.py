"""docs/notes/118 (audit #162): a free stream carrying an interface on grid faces in the conserving momentum
transport. The two defects (D1, D2) and their fixes as hooks (the library is unchanged); the growing base scheme is
kept here as such."""

import numpy as np
import pytest

from cases.r0_aligned_free_stream import SCHEMES, free_stream, identical, kick_d2, scheme

pytestmark = pytest.mark.xdist_group("r0_aligned_free_stream")


def test_hooks_apply_to_the_library_and_change_nothing_when_off():
    from basilisk_cupy.interface import two_phase, vof

    saved = (vof._sweep, two_phase.TwoPhase._vof, two_phase._MomentumField.fill_ghosts)
    with scheme(True, True):
        assert vof._sweep is not saved[0]
    assert (vof._sweep, two_phase.TwoPhase._vof, two_phase._MomentumField.fill_ghosts) == saved
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    def plain(steps=40, n=32):                       # the library itself, no hook
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        i = np.arange(n)[:, None] * np.ones((1, n))
        two_phase.TwoPhase(ns, CellField.from_interior(grid, ((i >= 8) & (i < 16)).astype(float),
                                                       BoundaryConditions.periodic(2)),
                           rho1=1.0, rho2=1e-3, conserving=True)
        ns.cfl = 0.5
        ns.u[0].interior = 1.0 + 0.0 * i
        ns.u[1].interior = 0.0 * i
        ns.start()
        m = 0.0
        for _ in range(steps):
            ns.step()
            m = max(m, float(np.abs(ns.u[0].interior - 1).max()), float(np.abs(ns.u[1].interior).max()))
        return m

    assert free_stream("column", "base", 1e3, steps=40)["max_err"] == plain()


def test_d2_one_sweep_formula_is_exact_and_the_fix_removes_it():
    rows = kick_d2(cfl=(0.45, 0.4999), c=(0.5, 0.51))
    for r in rows:
        assert abs(r["measured"]["base"] - r["predicted"]) < 1e-8 * abs(r["predicted"])
        assert abs(r["measured"]["d2"]) < 1e-15 and abs(r["measured"]["d1+d2"]) < 1e-15
    worst = min(rows, key=lambda r: r["delta"])
    assert worst["cfl"] == 0.4999 and abs(worst["gain_base"]) > 100      # ~ ratio / 8 as delta -> 0


def test_no_interface_all_schemes_identical():
    ref = identical("base", steps=10)
    for s in SCHEMES:
        assert np.array_equal(identical(s, steps=10), ref)


def test_aligned_column_grows_in_the_base_and_stays_at_round_off_with_both_fixes():
    out = {s: free_stream("column", s, 1e3)["max_err"] for s in ("base", "d1", "d2", "d1+d2")}
    assert out["base"] > 1e-2                        # audit #162 (C's 7.8e-2)
    assert out["d1"] > 1e-2                          # D1 alone does not cure the growth at the CFL bound
    assert 1e-6 < out["d2"] < 1e-3                   # D2 alone: the D1 level remains
    assert out["d1+d2"] < 1e-9


def test_below_the_cfl_bound_d1_is_the_cause():
    out = {s: free_stream("column", s, 1e3, cfl=0.45)["max_err"] for s in ("base", "d1", "d1-absent", "d1-lin")}
    assert out["base"] > 1e-6 and out["d1"] < 1e-9
    assert out["d1-absent"] > 1e-7 and out["d1-lin"] > 1e-7     # both parts are needed


@pytest.mark.parametrize("name,ratio,U", [("square", 1e6, (1.0, 0.0)), ("column", 1e6, (1.0, 0.37)),
                                          ("drop", 1e3, (1.0, 0.37))])
def test_free_stream_with_both_fixes(name, ratio, U):
    assert free_stream(name, "d1+d2", ratio, U=U, steps=300)["max_err"] < 1e-8

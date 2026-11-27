"""docs/notes/95: density-robust two-phase PCCD (diagnosis, local CCD correction, phase split, Floquet)."""

import math

import numpy as np

from cases.pccd_gram import stats
from cases.pccd_jump import A_, B_, exact
from cases.pccd_local import Local, Split, end_closure, oracle_local
from cases.pccd_twophase import averages


def test_end_closure_is_the_users_five_cell_formula():
    cv, cg = end_closure()
    assert np.allclose(cv, [137 / 60, -163 / 60, 137 / 60, -21 / 20, 1 / 5])
    assert np.allclose(cg, [-15 / 4, 109 / 12, -35 / 4, 17 / 4, -5 / 6])


def _err(R, N):
    x = (np.arange(N * 41) + 0.5) / (N * 41)
    return abs(R(x) - exact(x)).max()


def test_local_correction_and_phase_split_are_fifth_order_at_ratio_one_million():
    for make in (lambda N, ub: oracle_local(N, 1e6, A_, B_, ub), lambda N, ub: Local(N, 1e6, A_, B_, ub),
                 lambda N, ub: Split(N, 1e6, A_, B_, ub)):
        e = [_err(make(N, averages(N, A_, B_, 1e6, exact)), N) for N in (64, 128)]
        assert math.log2(e[0] / e[1]) > 4.5


def test_the_light_to_heavy_leak_drives_the_gram_matrix():
    """lambda_max >= r (1/h) int_H |phi_j|^2 for the light cells; the phase split removes the growth with r."""
    for mode in ("none", "joint"):
        s = stats(32, 1e6, mode)
        assert s["lmax"] >= s["bound_r_leak"] > 1e3 and s["top_mode_light_share"] > 0.99
    a, b = stats(32, 1e3, "split")["lmax"], stats(32, 1e6, "split")["lmax"]
    assert abs(a / b - 1) < 0.01 and b < 100


def test_floquet_galerkin_is_bounded_and_split_remap_is_stable():
    from cases.pccd_floquet import floquet

    g = floquet(32, 1e6, 0.05, "split", "galerkin")
    assert g["sv_G"] < 1 + 1e-9
    r = floquet(32, 1e6, 0.05, "split", "remap")
    assert r["spectral_radius"] < 1 + 1e-9 and r["sv_mass"] < 6
    j = floquet(32, 1e3, 0.5, "joint", "remap")
    assert j["spectral_radius"] > 1.1                   # the cut-off jump estimate makes the remap unstable

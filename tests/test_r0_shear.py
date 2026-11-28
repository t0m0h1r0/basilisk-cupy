"""A-7 (#167): the oblique two-layer shear flow + Galilean boost. The closed-form reference (checked against the
audit's independent C-3 in docs/figures/r0_shear.json) and the library's values before the R0 changes."""

import math

import numpy as np
import pytest

from cases.r0_shear import Shear, run

pytestmark = pytest.mark.xdist_group("r0_shear")


@pytest.mark.parametrize("n", [(0, 1), (1, 1), (1, 2)])
def test_reference_modes_cells_and_faces(n):
    ex = Shear(n=n)
    assert all(m["closure"] < 1e-12 for m in ex.modes)
    lam_ratio = ex.lam[0] / (ex.k2 * 0.40031355226532683)                   # C-3's lambda_1 for n = (0, 1), scaled
    assert abs(lam_ratio - 1.0) < 1e-9                                       # the rate scales with |n|^2
    N, t = 24, 0.3
    w = ex.w_cells(N, t)
    # the cell averages against a fine midpoint rule
    k = 40
    s = (np.arange(k) + 0.5) / (k * N)
    i, j = 5, 7
    X, Y = np.meshgrid(i / N + s, j / N + s, indexing="ij")
    z = n[0] * (X - ex.U[0] * t) + n[1] * (Y - ex.U[1] * t)
    assert abs(ex.w_at(z, t).mean() - w[i, j]) < 2e-5
    fx, fy = ex.face_fluxes(N, t)
    assert np.abs(fx[1:, :] - fx[:-1, :] + fy[:, 1:] - fy[:, :-1]).max() < 1e-12
    f = ex.fraction_cells(N, t)
    assert abs(f.mean() - (ex.beta - ex.alpha)) < 1e-12


def test_library_run_conserves_with_exact_initial_faces_and_not_with_its_own_start():
    a = run(n=(1, 2), boost=False, N=32, samples=(0.5,))
    b = run(n=(1, 2), boost=False, N=32, samples=(0.5,), exact_uf=False)
    assert a["failed"] is None and a["rows"][-1]["mass"] < 1e-11
    assert b["rows"][-1]["mass"] > 1e-5                                      # the unprojected first face velocity
    assert abs(a["rows"][-1]["all"]["L2"] - b["rows"][-1]["all"]["L2"]) < 0.01 * a["rows"][-1]["all"]["L2"]


def test_aligned_band_with_boost_runs():
    r = run(n=(0, 1), boost=True, N=32, samples=(0.25,))
    assert r["failed"] is None and r["rows"][-1]["all"]["L2"] < 0.05 and r["rows"][-1]["momentum"] < 1e-11

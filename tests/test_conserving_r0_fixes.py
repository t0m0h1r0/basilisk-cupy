"""#166 (docs/notes/118, 119, 121, 123): the library options of the R0 fixes, off by default (conserving.h).

``TwoPhase(conserving_consistent=True)`` (D1), ``conserving_slope_limit=3.0`` (D2 limited), ``CenteredNS.project_start``
(a divergence-free first face velocity) and the warning when rho_lin reaches its floor."""

import warnings

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
from basilisk_cupy.interface.two_phase import TwoPhase
from basilisk_cupy.time.basilisk_centered import CenteredNS

pytestmark = pytest.mark.xdist_group("conserving_r0_fixes")
R0 = dict(conserving_consistent=True, conserving_slope_limit=3.0)


def _band(n):
    i = np.arange(n)[:, None] * np.ones((1, n))
    return ((i >= n // 4) & (i < n // 2)).astype(float)


def _free_stream(f0, ratio, U, steps, n=32, **kw):
    grid = CartesianGrid((n, n))
    ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
    tp = TwoPhase(ns, CellField.from_interior(grid, f0, BoundaryConditions.periodic(2)), rho1=1.0, rho2=1.0 / ratio,
                  conserving=True, **kw)
    ns.cfl = 0.5
    for k in range(2):
        ns.u[k].interior = U[k] + 0.0 * ns.u[k].interior
    ns.start()
    worst = 0.0
    for _ in range(steps):
        ns.step()
        worst = max(worst, max(float(np.abs(ns.u[k].interior - U[k]).max()) for k in range(2)))
    return worst, tp


@pytest.mark.parametrize("ratio,U", [(1e3, (1.0, 0.0)), (1e3, (1.0, 0.37)), (1e6, (1.0, 0.0))])
def test_aligned_band_free_stream_with_the_r0_options(ratio, U):
    assert _free_stream(_band(32), ratio, U, 600, **R0)[0] < 1e-8


def test_aligned_band_grows_with_the_defaults():
    # audit #162: conserving.h's transport (the defaults) lets the aligned band grow at the CFL bound
    assert _free_stream(_band(32), 1e3, (1.0, 0.0), 600)[0] > 1e-3


def test_no_interface_bit_identical():
    n = 32
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    out = []
    for kw in ({}, R0):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        TwoPhase(ns, CellField.from_interior(grid, np.zeros((n, n)), BoundaryConditions.periodic(2)), rho1=1.0,
                 rho2=1e-3, conserving=True, **kw)
        ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        ns.start()
        for _ in range(20):
            ns.step()
        out.append(np.stack([ns.u[0].interior, ns.u[1].interior]))
    assert np.array_equal(out[0], out[1])


def test_rho_floor_is_counted_and_warned():
    f0 = _band(16)
    f0[0, 0] = -0.01                                          # far below 0: rho_lin < min(rho) / 2 at ratio 1000
    with pytest.warns(RuntimeWarning, match="floor"):
        _, tp = _free_stream(f0, 1e3, (1.0, 0.0), 1, n=16, **R0)
    assert tp.rho_floor_hits > 0


def test_project_start_gives_a_divergence_free_first_face_velocity():
    n = 32
    rng = np.random.default_rng(0)
    for proj in (False, True):
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        ns.project_start = proj
        ns.u[0].interior = rng.standard_normal((n, n))
        ns.u[1].interior = rng.standard_normal((n, n))
        ns.start()
        ux, uy = ns.uf.components
        div = np.abs(ux[1:, :] - ux[:-1, :] + uy[:, 1:] - uy[:, :-1]).max()
        if proj:
            assert div < 1e-8
        else:
            assert div > 1e-2

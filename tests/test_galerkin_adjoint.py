"""docs/notes/84: the adjoint-corrected Galerkin transport T_beta = T [I + beta (I - T^dagger T)]."""

import math

import numpy as np

from cases.galerkin_cls import positions, step10
from cases.galerkin_fourier import dissipation


def test_structure_kept_by_the_correction():
    """Free stream, total momentum and the energy-norm bound survive beta = 1/2 at density ratio 1e3."""
    st = step10(48, 4, positions(48, 3)[1], 0.37 / 48, 0.75, 1e3, 0.5)
    assert st["free_stream"] < 1e-11 and st["momentum"] < 1e-11
    assert st["amp"] <= 1.0 + 1e-9


def test_small_cfl_loss_becomes_second_order():
    """Density ratio 1: the worst one-step loss is O(CFL) without the correction and O(CFL^2) with it."""
    s0 = positions(48, 3)[1]
    loss = {b: [step10(48, 4, s0, c / 48, 0.75, 1.0, b)["loss"] for c in (1e-3, 1e-2)] for b in (0.0, 0.5)}
    assert 8 < loss[0.0][1] / loss[0.0][0] < 12
    assert 80 < loss[0.5][1] / loss[0.5][0] < 120


def test_constant_density_symbol():
    """Bloch: the corrected cell-wise quartic's dissipation per unit time vanishes with the CFL number."""
    d = [dissipation("poly", c, math.pi / 2, 0.5)["d"] for c in (1e-3, 1e-2)]
    assert abs(d[1] / d[0] - 10) < 0.5 and d[0] < 1e-6
    assert np.isclose(dissipation("poly", 1e-3, math.pi / 2, 0.5)["d"], 7.03e-7, rtol=1e-2)


def test_face_continuous_correction():
    """P0-10c/d at constant density: the face-continuous correction (face value = mean) removes the small-CFL
    dissipation (6.92e-5 at CFL 0.001, wavelength 4h) and keeps the cell average; with the adjoint correction 5.94e-5
    at CFL 0.37 (the user's values)."""
    from cases.galerkin_fourier import phi

    assert np.isclose(dissipation("c0avg", 1e-3, math.pi / 2)["d"], 6.92e-5, rtol=1e-2)
    assert np.isclose(dissipation("c0avg", 0.37, math.pi / 2, 0.5)["d"], 5.94e-5, rtol=1e-2)
    xg, wg = np.polynomial.legendre.leggauss(10)
    avg = 0.5 * np.sum(wg * phi("c0avg", 0.7, 0.5 * xg))  # cell 0 = [-1/2, 1/2]
    assert abs(avg - 1.0) < 1e-13

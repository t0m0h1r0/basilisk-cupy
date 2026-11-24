"""Conservative level set (TwoPhaseFlow's CLS): maps, mass closure, transport, re-initialisation."""

import math

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.interface.cls import CLSInterface, mass_correction, phi_from_psi, psi_from_phi
from basilisk_cupy.interface.fractions import fraction


def test_psi_phi_maps_are_inverse():
    phi = np.linspace(-0.05, 0.05, 101)
    eps = 0.01
    psi = psi_from_phi(phi, eps)
    np.testing.assert_allclose(psi, 1.0 / (1.0 + np.exp(-phi / eps)), atol=1e-15)  # H_eps(-phi_paper)
    np.testing.assert_allclose(phi_from_psi(psi, eps), phi, atol=1e-12)
    assert abs(phi_from_psi(np.array(1.0), eps)) < 2 * eps * math.atanh(1 - 2e-6) + 1e-12  # limited


def test_mass_correction_restores_the_total_and_only_touches_the_interface():
    rng = np.random.default_rng(0)
    psi = np.clip(rng.random((16, 16)) * 1.4 - 0.2, 0, 1)
    out = mass_correction(psi, 0.01, float(psi.sum() * 0.01) + 0.003)
    assert float(out.sum() * 0.01) == pytest.approx(float(psi.sum() * 0.01) + 0.003, abs=1e-12)
    assert np.all(out[(psi == 0) | (psi == 1)] == psi[(psi == 0) | (psi == 1)])


def _circle(grid):
    xp = grid.xp
    x, y = grid.meshgrid()
    return 0.2 - xp.sqrt((x + 0.2) ** 2 + (y + 0.236338) ** 2)


def test_reinitialisation_keeps_a_circle(backend):
    """100 re-initialisations (4 pseudo-steps each) do not move the psi = 1/2 contour of a circle;
    the total is kept to round-off. Measured (sharp-indicator L1 error): 4.997e-3 before and after, N = 64."""
    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    cls = CLSInterface.from_level_set(grid, _circle(grid), BoundaryConditions())
    f0 = fraction(grid, lambda x, y: 0.04 - (x + 0.2) ** 2 - (y + 0.236338) ** 2)
    sharp0 = float(abs((cls.psi.interior > 0.5) - f0).sum())
    m0 = cls.mass()
    for _ in range(100):
        cls.reinitialize(4)
    sharp1 = float(abs((cls.psi.interior > 0.5) - f0).sum())
    assert sharp1 == pytest.approx(sharp0, rel=0.05) and abs(cls.mass() / m0 - 1) < 1e-13
    p = to_numpy(cls.psi.interior)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_translation_conserves_mass_and_stays_bounded(backend):
    """Periodic translation by (1, 0.5) over t = 1, N = 64, CFL 0.25: mass to round-off, psi in [0, 1];
    psi L1 error 5.4e-3 (VOF on the same velocity: 3.4e-4)."""
    n = 64
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    bc = BoundaryConditions.periodic(2)
    cls = CLSInterface.from_level_set(grid, 0.2 - xp.sqrt(x * x + y * y), bc)
    m0 = cls.mass()
    uf = FaceField(grid, [xp.ones(grid.face_shape(0)), 0.5 * xp.ones(grid.face_shape(1))])
    for _ in range(4 * n):
        cls.advect(uf, 1.0 / (4 * n))
    ye = ((y + 1.0) % 1.0) - 0.5
    exact = CLSInterface.from_level_set(grid, 0.2 - xp.sqrt(x * x + ye * ye), bc).psi.interior
    err = float(abs(cls.psi.interior - exact).sum()) / (n * n)
    assert abs(cls.mass() / m0 - 1) < 1e-13 and err < 7e-3
    assert float(cls.psi.interior.min()) >= 0.0 and float(cls.psi.interior.max()) <= 1.0


def test_reversed_vortex_against_vof(backend):
    """Basilisk's reversed.c velocity, N = 32, T = 15: CLS sharp-indicator error 0.0707 (VOF: 0.0606);
    total psi to round-off."""
    from cases import reversed_cls

    r = reversed_cls.run(32, backend)
    assert r["sharp"][0] == pytest.approx(0.0707, rel=0.02)
    assert abs(r["mass_change"]) < 1e-13
    assert r["psi_range"][0] >= 0.0 and r["psi_range"][1] <= 1.0


def test_two_dimensions_uniform_only():
    grid = CartesianGrid((8, 8, 8))
    with pytest.raises(NotImplementedError):
        CLSInterface(CellField(grid))

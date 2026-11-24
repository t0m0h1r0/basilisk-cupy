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
        cls.reinitialize(4, method="compression")
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


def _exact_circle(n, backend, bc=None):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    return grid, CLSInterface.from_level_set(grid, _circle(grid), bc or BoundaryConditions())


@pytest.mark.parametrize("distance,bounds", [("points", (0.035, 0.05)), ("segments", (0.0, 0.005))])
def test_xi_sdf_on_an_exact_profile(backend, distance, bounds):
    """One xi-SDF re-initialisation of the exact H_eps profile of a circle, N = 32 / 64 / 128:
    the paper's distance to the nearest crossing point changes psi by 4.5e-2 / 4.5e-2 / 4.2e-2 (an
    oblique interface makes the axis-aligned crossing up to sqrt(2) farther than the interface);
    the distance to the polyline through the crossings by 3.6e-3 / 1.8e-3 / 9.0e-4."""
    changes = []
    for n in (32, 64, 128):
        grid, cls = _exact_circle(n, backend)
        p0 = cls.psi.interior.copy()
        m0 = cls.mass()
        cls.reinitialize(distance=distance)
        changes.append(float(abs(cls.psi.interior - p0).max()))
        assert abs(cls.mass() / m0 - 1) < 1e-12
    assert all(bounds[0] <= c <= bounds[1] for c in changes)
    if distance == "segments":
        assert changes[0] / changes[2] > 3.5  # first order


def test_xi_sdf_keeps_the_phase_of_every_cell_and_the_mass(backend):
    grid, cls = _exact_circle(64, backend)
    inside = to_numpy(cls.psi.interior > 0.5)
    m0 = cls.mass()
    for _ in range(20):
        cls.reinitialize()
    assert (to_numpy(cls.psi.interior > 0.5) == inside).mean() > 0.995 and abs(cls.mass() / m0 - 1) < 1e-12


def test_xi_sdf_across_a_periodic_boundary(backend):
    """A circle cut by the periodic boundaries: distances wrap (minimum image)."""
    grid = CartesianGrid((32, 32), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    dx, dy = x - 0.45, y - 0.45
    dx, dy = dx - xp.round(dx), dy - xp.round(dy)
    cls = CLSInterface.from_level_set(grid, 0.2 - xp.sqrt(dx * dx + dy * dy), BoundaryConditions.periodic(2))
    p0 = cls.psi.interior.copy()
    cls.reinitialize(distance="segments")
    assert float(abs(cls.psi.interior - p0).max()) < 5e-3


def test_reversed_vortex_with_xi_sdf_reinitialisation(backend):
    """Every 10 steps, N = 32: sharp-indicator error 0.1092 (no re-initialisation: 0.0707; VOF 0.0606)."""
    from cases import reversed_cls

    r = reversed_cls.run(32, backend, reinit_every=10, method="xi_sdf")
    assert r["sharp"][0] == pytest.approx(0.1092, rel=0.02) and abs(r["mass_change"]) < 1e-12


def test_reversed_vortex_with_compression_every_step(backend):
    """Compression--diffusion after every step thins the filaments: N = 32 / 64 / 128 sharp-indicator
    error 0.171 / 0.137 / 0.0687 (none: 0.0707 / 0.0306 / 0.0139; every 10 steps: 0.0707 / 0.0306 / 0.0080)."""
    from cases import reversed_cls

    r = reversed_cls.run(32, backend, reinit_every=1, method="compression")
    assert r["sharp"][0] == pytest.approx(0.17099, rel=0.01) and abs(r["mass_change"]) < 1e-12

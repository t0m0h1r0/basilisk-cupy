"""The energy-consistent implicit momentum transport on small matrices (docs/notes/78, P0-3 / P0-4)."""

import math

import pytest

from cases.energy_transport import accuracy, identities


@pytest.mark.parametrize("structure", ["pair2", "pair4"])
@pytest.mark.parametrize("lam_scale", [0.0, 1.0])
def test_identities(structure, lam_scale):
    """Random masses (1000:1), fluxes and velocities: momentum, the energy balance -1/2 sum |M| lambda |dw|^2 and a
    free stream to round-off; (A + A^T)/2 positive definite; mass-weighted amplification at most 1."""
    for seed in range(5):
        r = identities(seed, structure=structure, lam_scale=lam_scale)
        assert r["momentum"] < 1e-14 and r["energy_balance"] < 1e-14 and r["free_stream"] < 1e-14
        assert r["sym_min_eig"] > 0.0 and r["amplification"] <= 1.0 + 1e-12
        if lam_scale == 0.0:
            assert abs(r["dE"]) < 1e-14
        else:
            assert r["dE"] < 0.0


def test_pair4_order():
    """At a small CFL number the pair structure of the fourth-order difference is close to fourth order in space."""
    e = [accuracy("pair4", n, 0.05) for n in (16, 32)]
    assert math.log2(e[0] / e[1]) > 3.5
    assert accuracy("pair2", 32, 0.05) > 50 * e[1]


def test_pair4_variable_density_order():
    """P0-4a: the space error alone (semi-discrete) of pair4 with a density wave (the face mass flux varies in space
    and time) is fourth order."""
    from cases.energy_transport import semi_discrete

    e = [semi_discrete("pair4", n, "density wave") for n in (16, 32)]
    assert math.log2(e[0] / e[1]) > 3.7


def test_energy_orders_deform_return():
    """P0-4b: the exact cell averages' E_2 changes by O(h^2) under the deformation, E_4 by O(h^4)."""
    from cases.energy_transport import exact_energies

    d = []
    for n in (32, 64):
        e0, e1 = exact_energies(n, 0.0), exact_energies(n, 0.5)
        d.append((abs(e1[0] / e0[0] - 1), abs(e1[1] / e0[1] - 1)))
    assert 1.9 < math.log2(d[0][0] / d[1][0]) < 2.1
    assert 3.8 < math.log2(d[0][1] / d[1][1]) < 4.2


def test_pair4_variable_velocity_second_order():
    """docs/notes/79: with a varying velocity (steady variable density, constant mass flux) pair4 is second order at an
    intermediate time; its fourth order at t = 1 is an end-of-period cancellation."""
    from cases.energy_transport import semi_discrete

    e = [semi_discrete("pair4", n, "steady variable velocity", times=[0.25, 1.0]) for n in (64, 128)]
    assert 1.8 < math.log2(e[0][0.25] / e[1][0.25]) < 2.3
    assert math.log2(e[0][1.0] / e[1][1.0]) > 3.8


def test_similarity_transform_deform_return():
    """docs/notes/79 (the user's R^-1 A R): fourth order on the deform-and-return shear with point velocities, E_R
    conserved; second order with the rows' mean velocities."""
    from cases.energy_transport import reconstructed_shear

    c = [reconstructed_shear("C", n) for n in (32, 64)]
    b = [reconstructed_shear("B", n) for n in (32, 64)]
    assert math.log2(c[0][0] / c[1][0]) > 3.9 and max(c[0][1], c[1][1]) < 1e-14
    assert math.log2(b[0][0] / b[1][0]) < 2.1


def test_reconstruction_variable_density():
    """docs/notes/79: rho* = R rhobar, (rho q)* = R (rho q)bar and the split form: fourth order at t = 1/4 with a
    varying velocity, momentum to round-off for a steady mass flux."""
    from cases.energy_transport import reconstructed_semi

    r = [reconstructed_semi(n, "steady variable velocity", times=(0.25,)) for n in (32, 64)]
    assert math.log2(r[0][0.25] / r[1][0.25]) > 3.8
    assert r[1][("momentum", 0.25)] < 1e-14


def test_point_space_density_wave():
    """docs/notes/79: density and momentum moved in point space by the same skew operator: fourth order at t = 1/4
    with a mass flux varying in space and time, mass and momentum to round-off."""
    from cases.energy_transport import point_space_semi

    r = [point_space_semi(n, "density wave", times=(0.25,)) for n in (32, 64)]
    assert math.log2(r[0][0.25] / r[1][0.25]) > 3.8
    assert r[1][("mass", 0.25)] < 1e-14 and r[1][("momentum", 0.25)] < 1e-14

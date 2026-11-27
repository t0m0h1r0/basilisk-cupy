"""docs/notes/105 (Gate 1-A2): the divergence-consistent MAC-PCCD (the user's proposal), single phase, periodic."""

import math

from cases.mac_pccd import identities, pressure_gradient, projection_accuracy, structure_2d, symbols_table


def test_face_reconstruction_identities():
    r = identities(24)
    assert r["face_left"] < 1e-13 and r["face_right"] < 1e-13 and r["dRF_eq_RC_D"] < 1e-12 and r["GF_one"] < 1e-12
    assert abs(r["GF_eigs"][0] - 0.50116) < 1e-4 and abs(r["GF_eigs"][1] - 1.0) < 1e-12


def test_schur_symbols_match_the_users_values():
    s = symbols_table(1025)
    assert abs(s["mac_pccd_1d"] - 0.656386) < 1e-5 and abs(s["mac_pccd_2d"] - 0.428506) < 2e-4
    assert abs(s["g_F"][0] - 0.501160) < 1e-5 and abs(s["mac_2d"] - 4 / math.pi ** 2) < 1e-6


def test_projection_is_orthogonal_and_pointwise_divergence_free():
    r = structure_2d(10)
    assert r["null_L"] == 1 and r["idempotent"] < 1e-12 and r["B_Pi"] < 1e-12 and r["energy_max"] < 1e-12
    assert r["momentum"] < 1e-12 and r["pointwise_div"] < 1e-11


def test_divergence_free_waves_are_kept_and_gradients_removed():
    for row in projection_accuracy(thetas=(0.8, 0.4)):
        assert row["div_free_err"] < 1e-14
    a, b = projection_accuracy(thetas=(1.6, 0.8))
    assert b["grad_err"] < a["grad_err"] / 1000


def test_pressure_gradient_beats_classical_mac():
    for N in (16, 32):
        r = pressure_gradient(N)
        assert r["mac_pccd"] < 1e-6 and r["mac"] > 1e-2

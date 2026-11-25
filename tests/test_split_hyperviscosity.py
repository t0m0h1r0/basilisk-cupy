"""Spectrally split damping gamma Q^T M Q (docs/notes/42): what it does and what it does not."""

import numpy as np
import pytest

from cases.split_hyperviscosity import filters, laplacian_1d

pytestmark = pytest.mark.xdist_group("split_hyperviscosity")


def test_filters_are_complementary_and_selective():
    P, Q, _ = filters(16, 16, 10.0, 4)
    assert np.allclose(P + Q, np.eye(256))
    i = np.arange(16)
    smooth = np.outer(np.cos(np.pi * (i + 0.5) / 16), np.ones(16)).ravel()  # lowest wall mode
    check = np.outer((-1.0) ** i, (-1.0) ** i).ravel()                     # checkerboard
    assert np.linalg.norm(Q @ smooth) < 1e-5 * np.linalg.norm(smooth)
    assert np.linalg.norm(Q @ check) > 0.8 * np.linalg.norm(check)


def test_backward_euler_substep_dissipates_with_walls():
    """The CCD Laplacian with walls is not symmetric (0.1), but M + dt gamma Q^T M Q is SPD: energy decays."""
    L = laplacian_1d(16, "neumann")
    assert np.linalg.norm(L - L.T) / np.linalg.norm(L) > 0.05
    _, Q, _ = filters(16, 16, 10.0, 4)
    rng = np.random.default_rng(0)
    rho = np.where(rng.random(256) > 0.5, 1.0, 0.01)
    A = np.diag(rho) + 10.0 * Q.T @ (rho[:, None] * Q)
    u = rng.standard_normal(256)
    for _ in range(5):
        new = np.linalg.solve(A, rho * u)
        assert new @ (rho * new) < u @ (rho * u)
        u = new


def test_damping_term_lowers_the_capillary_damping():
    from cases import capillary_wave as cw
    from cases.capwave_jacobian import dispersion

    base = dispersion(16, 4.0)
    damped = dispersion(16, 4.0, hyper={"gamma": 10.0 / cw.STEP, "alpha": 10.0, "m": 4})
    assert damped["damping_err"] < base["damping_err"]  # the layer carries the dissipation (note 41)

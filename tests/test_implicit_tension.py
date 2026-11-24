"""Semi-implicit surface tension on the transport velocity (TwoPhaseCLS(..., implicit_tension=theta))."""

import numpy as np
import pytest

from basilisk_cupy.interface.cls_two_phase import _cg_face, _face_operator


def _edge(b, axis, periodic):
    e = 0.5 * (b + np.roll(b, -1, axis))
    if not periodic:
        idx = [slice(None)] * 2
        idx[axis] = -1
        e[tuple(idx)] = 0.0
    return e


@pytest.mark.parametrize("ax", [0, 1])
def test_face_solve_periodic_fourier_mode(ax):
    """u / alpha - beta lap u = f / alpha for a Fourier mode: exact to round-off."""
    n0, n1 = 16, 12
    h = (1 / n0, 1 / n1)
    X, Y = np.meshgrid((np.arange(n0) + 0.5) * h[0], (np.arange(n1) + 0.5) * h[1], indexing="ij")
    f = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    a, b = 2.0, 0.003
    kd = 4 / h[0] ** 2 * np.sin(np.pi * h[0]) ** 2 + 4 / h[1] ** 2 * np.sin(np.pi * h[1]) ** 2
    u, _ = _cg_face(np, f.copy(), np.full(f.shape, b), np.full(f.shape, a), ax, (True, True), h)
    np.testing.assert_allclose(u, f / (1 + a * b * kd), atol=1e-12)


@pytest.mark.parametrize("ax", [0, 1])
@pytest.mark.parametrize("per", [(False, False), (False, True), (True, False)])
def test_face_operator_is_symmetric_positive_definite(ax, per):
    rng = np.random.default_rng(1)
    shape = (5, 4)
    B, A = rng.random(shape), 0.5 + rng.random(shape)
    op = _face_operator(np, B, A, ax, per, (0.1, 0.2), _edge(B, ax, per[ax]), _edge(B, 1 - ax, per[1 - ax]))
    M = np.stack([op(e.reshape(shape)).ravel() for e in np.eye(B.size)], axis=1)
    assert np.abs(M - M.T).max() < 1e-12 and np.linalg.eigvalsh(M).min() > 0.5


@pytest.mark.parametrize("kw,stable", [({}, False), ({"implicit_tension": 1.0}, True),
                                       ({"surface_viscosity": 1.0}, False)])
def test_static_drop_at_the_tension_bound(backend, kw, stable):
    """GFM drop, density ratio 100, mu = 0.1, N = 64, tension.h's step (factor 1, limit 0.95-1.0 without):
    stable with theta = 1 (max |u| 1.0e-10 after 800 steps), not with the surface viscosity added to the
    predictor's viscous solve. N = 128: theta = 1 / 4 / 16 stable up to 1.0 / 1.5 / 2.0."""
    from cases.capillary_dt_scan import growth

    r = growth(64, 0.01, 1.0, steps=800, backend=backend, **kw)
    assert r["stable"] == stable
    if stable:
        assert r["umax"] < 1e-9


def test_capillary_wave_cost(backend):
    """16 points per wavelength (GFM, corrected divergence, 6th-order faces): 0.0660 with theta = 1
    (0.0639 without; theta = 4: 0.0718)."""
    from cases import capillary_wave as cw

    r = cw.run(32, backend, interface="cls", pressure_jump=True, divergence="fccd", face_order=6,
               implicit_tension=1.0)
    assert r["error"] == pytest.approx(0.06599, rel=2e-3)

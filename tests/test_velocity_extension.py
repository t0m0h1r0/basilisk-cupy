"""Band correction of the CLS kinematics (docs/notes/44)."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from cases.velocity_extension import attach, corrected_face_velocity

pytestmark = pytest.mark.xdist_group("velocity_extension")


def _circle(n):
    g = CartesianGrid((n, n), origin=(0.0, 0.0))
    x, y = g.meshgrid()
    return g, CLSInterface.from_level_set(g, 0.15 - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2), BoundaryConditions.periodic(2),
                                          ops=FCCDFaceComplex(order=4))


def test_uniform_velocity_is_left_unchanged():
    g, cls = _circle(24)
    uf = FaceField(g, [np.full(g.face_shape(0), 0.7), np.full(g.face_shape(1), -0.3)], "uf")
    c = corrected_face_velocity(cls, uf)
    assert max(float(np.abs(c[a] - uf[a]).max()) for a in range(2)) < 1e-12


def test_translation_matches_the_transport_and_keeps_the_mass():
    g, a = _circle(24)
    _, b = _circle(24)
    attach(b)
    uf = FaceField(g, [np.ones(g.face_shape(0)), np.ones(g.face_shape(1))], "uf")
    m0 = b.mass()
    for _ in range(10):
        a.advect(uf, 0.01)
        b.advect(uf, 0.01)
    assert np.abs(a.psi.interior - b.psi.interior).max() < 1e-12
    assert abs(b.mass() / m0 - 1) < 1e-12


def test_capillary_wave_damping_is_restored():
    from cases.capwave_normal_mode import run

    t = run(16, 4.0, periods=1.0)
    e = run(16, 4.0, periods=1.0, kinematics="extension")
    assert t["damping_err"] < -0.10 and abs(e["damping_err"]) < 0.5 * abs(t["damping_err"])
    assert abs(e["frequency_err"]) < abs(t["frequency_err"])

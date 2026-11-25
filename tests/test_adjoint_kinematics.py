"""Adjoint of the corrected CLS kinematics (docs/notes/45)."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from cases.adjoint_kinematics import adjoint_force, crossing_matrix, transport_matrix

pytestmark = pytest.mark.xdist_group("adjoint_kinematics")


def _wave(n=16):
    g = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    x, y = g.meshgrid()
    return g, CLSInterface.from_level_set(g, y - 0.02 * np.cos(2 * np.pi * x), BoundaryConditions(), divergence="fccd",
                                          ops=FCCDFaceComplex(order=6))


@pytest.mark.parametrize("kin", ["transport", "extension"])
def test_semi_discrete_energy_identity(kin):
    g, cls = _wave()
    T, faces = transport_matrix(cls, kin)
    F, grad = adjoint_force(cls, T)
    u = np.random.default_rng(3).standard_normal(len(faces))
    dEdt = -grad @ (T @ u)
    work = u @ F * float(g.h[0] * g.h[1])
    assert abs(dEdt + work) < 1e-12 * abs(dEdt)


def test_crossing_matrix_is_the_derivative_of_the_crossing_heights():
    from cases.capwave_jacobian import crossing_heights

    g, cls = _wave()

    class TP:
        pass

    tp = TP()
    tp.f, tp.cls = cls.psi, cls
    C = crossing_matrix(cls)
    d = np.random.default_rng(4).standard_normal(cls.psi.interior.shape) * 1e-7
    q0 = cls.psi.interior.copy()
    e0 = crossing_heights(tp)
    cls.psi.interior = q0 + d
    e1 = crossing_heights(tp)
    cls.psi.interior = q0
    assert np.allclose(e1 - e0, C @ d.ravel(), rtol=1e-4, atol=1e-12)

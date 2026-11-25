"""Adjoint surface tension (docs/notes/43): exact energy exchange, but a diffuse force."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from cases.adjoint_tension import area_and_gradient, face_force

pytestmark = pytest.mark.xdist_group("adjoint_tension")


def test_work_equals_minus_the_surface_energy_change():
    n = 24
    g = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    x, y = g.meshgrid()
    cls = CLSInterface.from_level_set(g, y - 0.05 * np.cos(2 * np.pi * x), BoundaryConditions(), ops=FCCDFaceComplex(order=6))
    F, _ = face_force(cls, 1.0)
    rng = np.random.default_rng(1)
    ufx, ufy = rng.standard_normal((n + 1, n)), rng.standard_normal((n, n + 1))
    ufx[[0, n], :] = 0.0
    ufy[:, [0, n]] = 0.0
    psidot = cls._rhs(cls.psi.interior, FaceField(g, [ufx, ufy], "uf"))
    A, dA = area_and_gradient(cls.psi)
    dEdt = float(np.sum(dA * psidot))
    work = float(np.sum(ufx * F[0]) + np.sum(ufy * F[1])) * float(g.h[0] * g.h[1])
    assert abs(dEdt + work) < 1e-12 * abs(dEdt)
    assert A == pytest.approx(2.0 * (1 + 0.25 * (0.05 * 2 * np.pi) ** 2), rel=2e-3)


def test_spreading_the_jump_with_the_kinematic_weights_does_not_help():
    from cases.capwave_jacobian import dispersion

    sharp = dispersion(16, 4.0, curvature="exact", kinematics="weights")
    pair = dispersion(16, 4.0, curvature="exact", kinematics="weights", force_weights=True)
    assert pair["damping_err"] < sharp["damping_err"] < 0

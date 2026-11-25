"""Attribution of the remaining over-damping (docs/notes/46)."""

import pytest

from cases import capillary_wave as cw
from cases.capwave_jacobian import dispersion

pytestmark = pytest.mark.xdist_group("residual_attribution")

KW = {"curvature": "exact", "cls_kinematics": "extension", "tolerance": 1e-14}


def test_splitting_does_not_matter_and_the_ccd_viscosity_does():
    from basilisk_cupy.terms.viscous import CCDViscosity

    A = dispersion(16, 4.0, cw.STEP, **KW)
    C = dispersion(16, 4.0, cw.STEP, viscous=CCDViscosity(), **KW)
    D = dispersion(16, 4.0, cw.STEP, ns_kw={"coupled_iterations": 40, "coupled_tolerance": 1e-13,
                                            "coupled_on_failure": "ignore"}, **KW)
    assert abs(D["damping_err"] - A["damping_err"]) < 0.005
    assert C["damping_err"] < A["damping_err"] - 0.01


def test_smaller_steps_lower_the_over_damping():
    """N = 32, x 1: +7.7 % at dt, +5.2 % at dt/4 (at N = 16, 8 points per wavelength, the sign differs)."""
    a = dispersion(32, 1.0, cw.STEP, **KW)
    b = dispersion(32, 1.0, cw.STEP / 4, **KW)
    assert a["damping_err"] == pytest.approx(0.0772, abs=0.002)
    assert b["damping_err"] == pytest.approx(0.0519, abs=0.002)

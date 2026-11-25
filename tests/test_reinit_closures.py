"""The paper's two re-initialisation closures (lambda_Gamma, eps_eff), docs/notes/54."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface


def test_two_closures_reinit():
    """eps_eff of one re-initialisation of an exact profile: well posed for a drop, not for a nearly flat wave."""
    out = {}
    for n in (32, 64):
        g = CartesianGrid((n, n), origin=(-0.5, -0.5))
        x, y = g.meshgrid()
        for name, phi in (("drop", 0.25 - np.hypot(x, y)), ("wave", 0.01 * np.cos(2 * np.pi * x) - y)):
            c = CLSInterface.from_level_set(g, phi, BoundaryConditions())
            c.reinitialize(method="xi_sdf_vw")
            info = c.reinit_info
            assert abs(info["mass_residual"]) < 1e-12 and abs(info["volume_residual"]) < 1e-10
            out[(name, n)] = info["eps_eff"]
    assert out[("drop", 32)] == pytest.approx(1.0217, abs=1e-3) and out[("drop", 64)] == pytest.approx(1.0006, abs=1e-3)
    assert out[("wave", 32)] > 3.5 and out[("wave", 64)] == pytest.approx(0.25)

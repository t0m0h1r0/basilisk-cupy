"""#194: the parts of R0's tests run on small grids (no verdict: the values are the library's before the R0 changes)."""

import numpy as np
import pytest

from cases import r0_parts as P

pytestmark = pytest.mark.xdist_group("r0_parts")


def test_layers_exact_profiles():
    for kind in ("couette", "poiseuille"):
        f = P._layers_exact(kind, 0.4, (1.0, 0.1), U=1.0, F=1.0)
        y = np.linspace(0, 1, 2001)
        u = f(y)
        assert abs(u[0]) < 1e-14 and abs(u[-1] - (1.0 if kind == "couette" else 0.0)) < 1e-12
        k = np.searchsorted(y, 0.4)
        assert abs(f(np.array([0.4 - 1e-9]))[0] - f(np.array([0.4 + 1e-9]))[0]) < 1e-7          # continuity of u
    av = P._cell_averages_1d(lambda y: y * y, 4, 0.37)
    assert np.allclose(av, [((j + 1) ** 3 - j ** 3) / 3 / 16 for j in range(4)])


def test_couette_runs_and_reports():
    r = P.couette_poiseuille("couette", 16, offset=0.3, t_end=0.5)
    assert r["error"]["all"]["L2"] < 0.1 and r["divergence"] < 1e-8 and r["error"]["band"]["cells"] > 0


def test_translation_static_drop_shear_layer_report():
    d = P.drop_translation(16, periods=0.25)
    assert d["mass"] < 1e-10 and d["velocity_error"] < 1e-8
    s = P.static_drop(16, offset=(0.3, 0.1), t_end=0.02)
    assert s["failed"] is None and "band" in s["speed_end"]
    sh = P.shear_layer(16, t_end=0.1)
    assert sh["failed"] is None and sh["momentum_max"] < 1e-6


def test_configuration_switches():
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.vof_jump import TwoPhaseVOFJump
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    g = CartesianGrid((8, 8))
    ns = CenteredNS(g, periodic=(0, 1))
    tp = P.two_phase(ns, CellField.from_interior(g, np.zeros((8, 8)), BoundaryConditions.periodic(2)), (1.0, 1e-3),
                     (0.0, 0.0), {"face_density": "mean", "fixes": True, "project_start": True})
    assert isinstance(tp, TwoPhaseVOFJump) and tp.conserving_consistent and tp.conserving_slope_limit == 3.0
    assert ns.project_start

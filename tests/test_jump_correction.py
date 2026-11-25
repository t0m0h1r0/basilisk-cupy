"""Jump-corrected viscosity (docs/notes/40): what it fixes and what it does not; curvature and damping."""

import numpy as np
import pytest

from cases.capwave_normal_mode import K, mode, run

pytestmark = pytest.mark.xdist_group("jump_correction")


def test_correction_removes_the_kink_error_of_the_operator():
    """Exact mode with its kink moved onto the CLS interface: fd2's error in the tangential component
    near the interface drops from 0.31 to 0.07 (|mu lap u| = 0.66)."""
    import warnings

    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.discretization.ccd import FCCDFaceComplex
    from basilisk_cupy.interface.cls import CLSInterface
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.terms.viscous import FD2Stress
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from cases import capillary_wave as cw
    from cases.jump_viscous import correction

    warnings.filterwarnings("ignore")
    n, nu = 32, cw.MU
    _, F = mode(K, 1.0, 1.0, nu)
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")})
    x, y = grid.meshgrid()
    eta = 0.01 * np.cos(K * x)
    u, v, _, lu, _ = F(x, y - eta)
    cls = CLSInterface.from_level_set(grid, y - eta, BoundaryConditions(), ops=FCCDFaceComplex(order=6))
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=nu, mu2=nu, pressure_jump=True)
    tp.add_surface_tension(1.0)
    ns.u[0].interior = u.real
    ns.u[1].interior = v.real
    ns.u.fill_ghosts()
    st = FD2Stress()
    Lh = st(ns.u, nu)
    S = correction(ns, tp, st)
    near = np.abs(y) < 3 * grid.h[1]
    e0 = np.abs((Lh[0] - nu * lu.real)[near]).max()
    e1 = np.abs((Lh[0] + S[0] - nu * lu.real)[near]).max()
    assert e0 == pytest.approx(0.307, rel=0.05) and e1 < 0.3 * e0


def test_correction_is_unstable_with_the_moving_interface():
    r = run(32, 1.0, periods=1.0, cls_kw={"curvature_kw": {"method": "heights"}}, jump_correction={})
    assert not np.isfinite(r["damping_err"]) or r["damping_err"] < -1.0  # measured -246 % at N = 64, 1.5 periods


def test_height_curvature_reduces_the_damping_error():
    psi = run(32, 1.0, periods=1.5)
    hts = run(32, 1.0, periods=1.5, cls_kw={"curvature_kw": {"method": "heights"}})
    assert psi["damping_err"] == pytest.approx(-0.1635, abs=0.003)
    assert hts["damping_err"] == pytest.approx(-0.1136, abs=0.003)

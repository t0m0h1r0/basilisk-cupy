"""Linear stability of one step and the outer iterations at high density ratio (docs/notes/35)."""

import math

import numpy as np
import pytest

from cases.capillary_dt_scan import growth
from cases.operator_commutator import measure
from cases.step_jacobian import StepMap, arnoldi


def test_arnoldi_matches_the_dense_jacobian():
    """N = 16, ratio 1, K = 0, 1.6 x tension.h's step: the dense Jacobian's leading eigenvalues are a cluster
    -1.7786, -1.7753, -1.7749, -1.7733; 40 Arnoldi steps land in it (1.7718)."""
    sm = StepMap(16, 1.0, 0.1, 1.6, 0, settle=200)
    lam, _ = arnoldi(sm.jvp, sm.q0.size, m=40)
    assert 1.765 < abs(lam[0]) < 1.780
    assert abs(abs(np.angle(lam[0])) - math.pi) < 1e-3


@pytest.mark.parametrize("K,stable", [(1, False), (2, True)])
def test_one_outer_iteration_is_not_enough_at_ratio_100(K, stable):
    """Static drop, N = 32, ratio 100, 1.6 x: K = 1 has the oscillatory modes |lambda| = 1.189,
    arg = 0.22 pi; K = 2 is stable (|lambda| 0.999)."""
    r = growth(32, 0.01, 1.6, steps=400, mu=0.1, ns_kw={"coupled_iterations": K})
    assert r["stable"] == stable


def test_coupled_tolerance_stops_the_iterations():
    """Ratio 100, 1.6 x, K <= 16 until dt |g_{k+1} - g_k| < 1e-10."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.interface.cls import CLSInterface
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    n = 32
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-14,
                    coupled_iterations=16, coupled_tolerance=1e-10)
    cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=0.01, mu1=0.1, mu2=0.1, pressure_jump=True, capillary_dt_factor=1.6)
    tp.add_surface_tension(1.0)
    ns.start()
    its = []
    for _ in range(300):
        ns.step()
        its.append(len(ns.coupled_history))
    # the start-up transient takes all 16; then a few iterations reach 1e-10
    assert min(its) >= 2 and np.mean(its[-100:]) < 6
    assert ns.coupled_history[-1] * ns.clock.dt < 1e-10


def test_lagging_pressure_residue_scales_with_the_density_ratio():
    """|P (B - I) dt alpha grad q| / |dt grad q|, N = 32: 0.039 (ratio 1), 8.06 (ratio 100, mu equal),
    2.76 (ratio 100, nu equal), 0.0026 (ratio 1, mu ratio 100)."""
    a = measure(32, 1.0, 0.1)["lag"]
    b = measure(32, 0.01, 0.1)["lag"]
    c = measure(32, 0.01, 0.001)["lag"]
    d = measure(32, 1.0, 0.001)["lag"]
    assert a == pytest.approx(3.902e-2, rel=0.02) and b == pytest.approx(8.062, rel=0.02)
    assert c == pytest.approx(2.764, rel=0.02) and d == pytest.approx(2.641e-3, rel=0.05)

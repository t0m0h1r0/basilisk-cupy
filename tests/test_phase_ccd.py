"""Phase-wise CCD face gradient for a density ratio (docs/notes/55, 61)."""

import math

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.discretization.phase_ccd import _extend_outside_band, hermite_at_crossings
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX


def test_hermite_at_crossings_is_high_order():
    """Smooth periodic j, crossings at theta = 0.3 on every face: error ~ h^6."""
    errs = []
    for n in (16, 32):
        h = 1.0 / n
        x = (np.arange(n) + 0.5) * h
        j = np.sin(2 * math.pi * x)[:, None] * np.ones((1, 3))
        theta = np.full((n + 1, 3), 0.3)
        v = hermite_at_crossings(j, theta, 0, h, True)
        xf = (np.arange(n + 1) - 0.5 + 0.3) * h
        errs.append(float(np.abs(v[:, 0] - np.sin(2 * math.pi * xf)).max()))
    assert errs[1] < 1e-6 and math.log2(errs[0] / errs[1]) > 5.0


def test_extend_outside_band():
    v = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).T * np.ones((1, 2))
    band = np.zeros((5, 2), bool)
    band[1:3, 0] = True  # second column has no band cell: unchanged
    out = _extend_outside_band(v, band, 0)
    assert out[:, 0].tolist() == [2.0, 2.0, 3.0, 3.0, 3.0] and out[:, 1].tolist() == v[:, 1].tolist()


@pytest.mark.parametrize("ratio,u_res", [(1.0, 1.523647e-02), (1000.0, 1.207246e-01)])
def test_manufactured_density_ratio(ratio, u_res):
    """p_k = rho_k s + j0 H, N = 32: the library operator (FCCD divergence) as the prototype; pressure converges at
    about 3.8th order for any density ratio (note 55: 2.50 / 0.179 / 0.0118 at ratio 1000, N = 32 / 64 / 128)."""
    from cases.split_ppe_check import solve

    assert solve(32, ratio, "lib")["u_res"] == pytest.approx(u_res, rel=1e-4)


def test_static_drop_density_ratio_1000():
    """Walls, R = 0.25, sigma = 1, mu = 0.1, N = 32, 40 steps: jump 4.0024 (fd2 GFM 4.0098), max |u| 8.9e-5 (2.0e-4)."""
    g = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    x, y = g.meshgrid()
    ns = TwoPhaseIMEX(g, 0.25 - np.hypot(x, y), rho1=1.0, rho2=0.001, mu1=0.1, mu2=0.1, sigma=1.0,
                      faces="phase-ccd", tolerance=1e-8)
    ns.start()
    for _ in range(40):
        ns.step()
    p = ns.pressure()
    r = np.hypot(x, y)
    assert float(p[r < 0.1].mean() - p[r > 0.4].mean()) == pytest.approx(4.00238, abs=2e-5)
    assert ns.max_velocity() == pytest.approx(8.88e-5, rel=0.03)


@pytest.mark.parametrize("periodic", [True, False])
def test_batched_equals_line_by_line(periodic):
    from basilisk_cupy import CellField
    from basilisk_cupy.boundary import BoundaryConditions, Neumann

    n = 48
    g = CartesianGrid((n, n))
    x, y = g.meshgrid()
    ns = TwoPhaseIMEX(g, 0.25 - np.hypot(x - 0.5, y - 0.5), rho1=1.0, rho2=0.001,
                      periodic=(0, 1) if periodic else (), faces="phase-ccd")
    ns.start()
    bc = BoundaryConditions.periodic(2) if periodic else BoundaryConditions.uniform(2, Neumann(0.0))
    p = CellField(g, bc=bc)
    p.interior = np.cos(2 * math.pi * x) * np.sin(4 * math.pi * y) + 0.3 * x * x
    p.fill_ghosts()
    for ax in range(2):
        assert np.abs(ns.ops.grad_f(p, ax) - ns.ops._grad_f_loop(p, ax)).max() < 1e-10

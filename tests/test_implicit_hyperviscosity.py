"""Implicit UCCD6 hyperviscosity (``UCCD6Convection(implicit=True)``): solved with the viscous step."""

import math
import warnings

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.boundary import Periodic
from basilisk_cupy.fields import VectorField
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.terms.convection.uccd6 import UCCD6Convection, _hyperviscosity_eigen
from basilisk_cupy.terms.viscous import CCDViscosity
from basilisk_cupy.time.imex_rk import IMEXRKNS


@pytest.mark.parametrize("kind", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize("n", [16, 64])
def test_one_dimensional_hyperviscosity_has_a_nearly_non_negative_real_spectrum(kind, n):
    """Real spectrum in [-1.7, 8493.5] (units of h^-8). The compact extrapolating closure of the outer
    three D2 leaves one wall eigenvalue at -1.66 / -1.61 (Dirichlet, n = 16 / 64; Neumann -0.020 / -0.019),
    i.e. a growth rate 1.7 sigma |u| / h, 2e-4 of the Nyquist damping and below the Re(-D1 h) = 0.096 |u| / h
    that the previous wide Dirichlet closure gave the advection itself (compact: 1e-15)."""
    lam, V, Vi = _hyperviscosity_eigen(n, 1.0 / n, kind, kind)
    lam = lam / n ** 8  # symbol omega_2^8
    assert lam.real.min() > -1.7 and lam.real.max() < 8493.47 + 1e-6
    assert abs(lam.imag).max() < 0.1
    np.testing.assert_allclose(V @ Vi, np.eye(n), atol=1e-8)


def _channel_field(backend, n=12):
    grid = CartesianGrid((n, n), backend=backend)
    bc = BoundaryConditions().set("left", Periodic()).set("right", Periodic()) \
        .set("bottom", Dirichlet(0.0)).set("top", Dirichlet(0.0))
    rng = np.random.default_rng(0)
    u = VectorField([CellField.from_interior(grid, grid.xp.asarray(rng.random(grid.n)), bc) for _ in range(2)])
    return grid, u, bc


def test_fast_diagonalization_solves_the_implicit_stage(backend):
    grid, u, bc = _channel_field(backend)
    hv = UCCD6Convection(implicit=True).implicit_part(u)
    hv.coeffs = [1e-3 * grid.h[0] ** 7 * 1.3, 1e-3 * grid.h[1] ** 7 * 0.7]
    tau = 0.05
    rng = np.random.default_rng(1)
    r = [grid.xp.asarray(rng.random(grid.n)) for _ in range(2)]
    z = hv.precondition(r, tau, [bc, bc])
    w = VectorField([CellField.from_interior(grid, zi, bc) for zi in z])
    az = [zi + tau * hi for zi, hi in zip(z, hv.apply(w))]
    assert max(float(abs(to_numpy(a - b)).max()) for a, b in zip(az, r)) < 1e-12


def test_defect_correction_with_viscosity_and_hyperviscosity(backend):
    grid, u, bc = _channel_field(backend, 32)
    hv = UCCD6Convection(implicit=True).implicit_part(u)
    hv.coeffs = [1e-3 * grid.h[k] ** 7 * 1.0 for k in range(2)]
    dt = 2.0 * grid.h[0]  # CFL 2: tau H ~ 34 at the Nyquist mode
    stats = CCDViscosity().solve(u, FaceField.constant(grid, 1e-3), 1.0, dt, tolerance=1e-10, extra=hv)
    assert stats.converged and stats.i <= 25  # measured 17 (white-noise rhs, 32^2, channel)


def _taylor_green(n, cfl, t_end, backend, *, implicit, face_native=False, mu=0.01, noise=0.0):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "uccd6", "implicit": implicit, "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, periodic=(0, 1), mu=mu, ops=scheme.face, convection=scheme.convection,
                  viscous=scheme.viscous, fixed_dt=t_end / round(t_end * n / cfl), tolerance=1e-10,
                  face_native=face_native)
    ex = [-xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y), xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)]
    rng = np.random.default_rng(0)
    for a in range(2):
        ns.u[a].interior = ex[a] + noise * xp.asarray(rng.standard_normal(grid.n))
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(t_end)
    d = math.exp(-8 * math.pi ** 2 * mu * t_end)
    return ns, max(float(xp.max(xp.abs(ns.u[a].interior - d * ex[a]))) for a in range(2))


def test_same_solution_as_the_explicit_hyperviscosity_at_a_small_step(backend):
    _, explicit = _taylor_green(32, 0.1, 0.1, backend, implicit=False)
    _, implicit = _taylor_green(32, 0.1, 0.1, backend, implicit=True)
    assert implicit == pytest.approx(explicit, rel=0.02)


def test_inviscid_large_steps(backend):
    """Inviscid noisy Taylor--Green, N = 32, t = 1: explicit hyperviscosity blows up at CFL 0.3 (bound
    ~0.16); implicit + face state keeps the CFL-0.1 accuracy at CFL 2."""
    _, explicit = _taylor_green(32, 0.3, 1.0, backend, implicit=False, mu=0.0, noise=1e-6)
    _, implicit = _taylor_green(32, 2.0, 1.0, backend, implicit=True, face_native=True, mu=0.0, noise=1e-6)
    assert not explicit < 0.5
    assert implicit < 1e-4


def test_spectrum_and_step_bound_drop_the_hyperviscosity(backend):
    grid = CartesianGrid((16, 16), backend=backend)
    u = VectorField([CellField.from_interior(grid, grid.xp.ones(grid.n), BoundaryConditions.periodic(2))
                     for _ in range(2)])
    imag, real = UCCD6Convection(implicit=True).spectrum(u)
    assert real == 0.0 and imag > 0.0
    assert UCCD6Convection(implicit=True).max_stable_dt(u) == float("inf")


def test_lid_cavity_with_walls(backend):
    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: Dirichlet(0.0) for w in walls}
    tang["top"] = Dirichlet(1.0)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "uccd6", "implicit": True, "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                  convection=scheme.convection, viscous=scheme.viscous, fixed_dt=2.0 / 64, face_native=True)
    ns.start(events=[1.1])
    ns.run(1.1)
    # explicit hyperviscosity at CFL 0.07 (1006 steps): 0.009221
    assert ns.kinetic_energy() == pytest.approx(0.009221, rel=0.01)
    assert ns.max_divergence() < 1e-5

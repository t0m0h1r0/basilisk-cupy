"""Step 4e: high-order implicit viscosity (CCD full stress, defect correction with Basilisk's multigrid)."""

import math
import warnings

import numpy as np
import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.fields import VectorField
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.terms.viscous import BasiliskViscosity, CCDStress, CCDViscosity, cell_viscosity
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS

from .conftest import order

X, Y = sp.symbols("x y")
XY = (X, Y)
U = [sp.sin(X + 2 * Y) + sp.Rational(3, 10) * sp.cos(2 * X - Y), sp.cos(X + Y) * sp.sin(2 * X) + sp.Rational(1, 5)]


def _fn(expr, xp):
    f = sp.lambdify(XY, expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


def _stress_errors(form, periodic, mu_expr, backend):
    target = [sum(sp.diff(mu_expr * (sp.diff(U[a], XY[b]) + sp.diff(U[b], XY[a])), XY[b]) for b in range(2))
              for a in range(2)]
    errs, ns = [], [16, 32, 64, 128]
    for n in ns:
        grid = CartesianGrid((n, n), length=(2 * math.pi,) * 2, origin=(0.3, -0.2), backend=backend)
        exact = [_fn(e, grid.xp) for e in U]
        bcs = [BoundaryConditions.periodic(2) if periodic else BoundaryConditions.uniform(2, Dirichlet(exact[j]))
               for j in range(2)]
        u = VectorField([CellField.from_interior(grid, exact[j](*grid.meshgrid()), bcs[j]) for j in range(2)])
        u.fill_ghosts()
        x, y = grid.meshgrid()
        mu = _fn(mu_expr, grid.xp)(x, y) if mu_expr.free_symbols else float(mu_expr)
        out = CCDStress(form=form)(u, mu)
        errs.append(max(float(abs(to_numpy(out[a] - _fn(target[a], grid.xp)(x, y))).max()) for a in range(2)))
    return errs, ns


@pytest.mark.parametrize("form", ["split", "layered"])
# walls: the stress differentiates derivatives (extrapolating closure): 3rd order at the wall cells (measured 2.79-3.28)
@pytest.mark.parametrize("periodic,min_order", [(True, 5.9), (False, 2.7)])
@pytest.mark.parametrize("variable_mu", [False, True])
def test_full_stress_orders(backend, form, periodic, min_order, variable_mu):
    mu = sp.Rational(7, 10) + (sp.Rational(1, 5) * sp.sin(X - Y) if variable_mu else 0)
    errs, ns = _stress_errors(form, periodic, mu, backend)
    assert min(order(errs, ns)) > min_order


def test_split_is_more_accurate_than_layered_at_walls(backend):
    mu = sp.Rational(7, 10) + sp.Rational(1, 5) * sp.sin(X - Y)
    split, _ = _stress_errors("split", False, mu, backend)
    layered, _ = _stress_errors("layered", False, mu, backend)
    assert split[-1] < layered[-1] / 10


def _solve(form, n, kappa, backend, variable_mu=False, walls=True, maxiter=300):
    grid = CartesianGrid((n, n), backend=backend)
    xp = grid.xp
    bc = BoundaryConditions.uniform(2, Dirichlet(0.0)) if walls else BoundaryConditions.periodic(2)
    rng = np.random.default_rng(1)  # white noise: every mode, the grid-scale one included
    u = VectorField([CellField.from_interior(grid, xp.asarray(rng.random(grid.n)), bc) for _ in range(2)])
    if variable_mu:
        mu = FaceField(grid, [1.0 + 9.0 * (0.5 + 0.5 * xp.sin(2 * math.pi * sum(grid.face_meshgrid(a))))
                              for a in range(2)])
    else:
        mu = FaceField.constant(grid, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CCDViscosity(form=form, maxiter=maxiter).solve(u, mu, 1.0, kappa * grid.h[0] ** 2, tolerance=1e-10)


@pytest.mark.parametrize("kappa", [0.1, 10.0, 1000.0])
@pytest.mark.parametrize("variable_mu", [False, True])
def test_defect_correction_converges_for_any_kappa(backend, kappa, variable_mu):
    """kappa = nu dt / h^2 from explicit-like to Stokes-like: ~20-30 corrections, also for mu in [1, 10]."""
    stats = _solve("split", 32, kappa, backend, variable_mu)
    assert stats.converged and stats.i <= 32


def test_iterations_independent_of_resolution(backend):
    its = [_solve("split", n, 10.0, backend, variable_mu=True).i for n in (32, 64)]
    assert max(its) - min(its) <= 3


def test_layered_form_stalls_for_large_kappa(backend):
    """The CCD first derivative has no grid-scale mode, the fd2 correction operator damps it most."""
    assert _solve("layered", 32, 0.1, backend).converged
    assert not _solve("layered", 32, 10.0, backend, maxiter=100).converged


def test_cell_viscosity_of_a_constant_face_field(backend):
    grid = CartesianGrid((8, 8), backend=backend)
    assert float(abs(to_numpy(cell_viscosity(FaceField.constant(grid, 0.3)) - 0.3)).max()) < 1e-15


def _taylor_green(backend, n, viscous, mu=0.01, t_end=0.25):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"}, "terms": {"convection": "central",
                                                                                "viscosity": viscous}})
    ns = IMEXBDF2NS(grid, periodic=(0, 1), mu=mu, ops=scheme.face, tolerance=1e-11,
                    fixed_dt=t_end / round(t_end * n / 0.05), convection=scheme.convection, viscous=scheme.viscous)
    ns.u[0].interior = -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y)
    ns.u[1].interior = xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    ns.run(t_end)
    decay = math.exp(-8 * math.pi ** 2 * mu * t_end)
    ex = [-xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y) * decay, xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y) * decay]
    return max(float(xp.max(xp.abs(ns.u[a].interior - ex[a]))) for a in range(2))


def test_viscous_taylor_green_high_order_with_ccd_viscosity(backend):
    """mu = 0.01: the fd2 viscosity caps the error at 2nd order; CCD viscosity lifts it to ~5th."""
    ccd = [_taylor_green(backend, n, "ccd") for n in (16, 32)]
    assert math.log2(ccd[0] / ccd[1]) > 4.5
    basilisk = _taylor_green(backend, 32, "basilisk")
    assert ccd[1] < basilisk / 50


def test_lid_cavity_closer_to_the_fine_grid_value(backend):
    """E(t = 1.1), N = 64; reference: fd2 IMEX-BDF2 on 256^2 (0.009694, measured), lid.ref (N = 64) is 0.008904."""
    from basilisk_cupy import Dirichlet as D

    fine = 0.009694
    out = {}
    for viscous in ("basilisk", "ccd"):
        grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
        walls = ("left", "right", "bottom", "top")
        tang = {w: D(0.0) for w in walls}
        tang["top"] = D(1.0)
        scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                               "terms": {"convection": "central", "viscosity": viscous}})
        ns = IMEXBDF2NS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                        convection=scheme.convection, viscous=scheme.viscous, fixed_dt=0.25 / 64)
        ns.start(events=[1.1])
        ns.run(1.1)
        out[viscous] = ns.kinetic_energy()
        assert ns.max_divergence() < 1e-5
    assert abs(out["ccd"] - fine) < 0.75 * abs(out["basilisk"] - fine)


def test_bdf2_stokes_still_second_order_in_time(backend):
    from .test_imex_bdf2 import self_convergence

    ks = [10, 20, 40]
    # CCDViscosity through the public constructor argument
    import basilisk_cupy.time.imex_bdf2 as imex

    class CCDStepper(imex.IMEXBDF2NS):
        def __init__(self, grid, **kw):
            super().__init__(grid, viscous=CCDViscosity(), **kw)

    errs = self_convergence(CCDStepper, 16, 0.5, ks, 320, stokes=True, backend=backend)
    assert min(order(errs, ks)) > 1.95


def test_scheme_registry_builds_viscous_solvers():
    assert isinstance(build_scheme({}).viscous, BasiliskViscosity)
    scheme = build_scheme({"terms": {"viscosity": "ccd"}})
    assert isinstance(scheme.viscous, CCDViscosity) and scheme.viscous.stress.form == "split"
    assert build_scheme({"terms": {"viscosity": "ccd_layered"}}).viscous.stress.form == "layered"
    with pytest.raises(ValueError):
        build_scheme({"terms": {"viscosity": "spectral"}})

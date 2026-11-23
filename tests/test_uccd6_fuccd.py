"""Step 4d: UCCD6 and FUCCD momentum convection (TwoPhaseFlow paper §4.5, §4.7)."""

import math
import warnings

import numpy as np
import pytest
import sympy as sp

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.discretization import FCCDFaceComplex
from basilisk_cupy.fields import VectorField
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.terms.convection.central import CentralFluxConvection
from basilisk_cupy.terms.convection.fuccd import FUCCDConvection
from basilisk_cupy.terms.convection.uccd6 import OMEGA2_SQ_NYQUIST, UCCD6Convection
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS

from .conftest import order

X, Y = sp.symbols("x y")
XY = (X, Y)
# smooth, not divergence free, no symmetry
U = [sp.sin(X + 2 * Y) + sp.Rational(3, 10) * sp.cos(2 * X - Y), sp.cos(X + Y) * sp.sin(2 * X) + sp.Rational(1, 5)]
SKEW = [-sp.Rational(1, 2) * sum(U[k] * sp.diff(U[j], XY[k]) + sp.diff(U[k] * U[j], XY[k]) for k in range(2))
        for j in range(2)]
CONS = [-sum(sp.diff(U[k] * U[j], XY[k]) for k in range(2)) for j in range(2)]


def _fn(expr, xp):
    f = sp.lambdify(XY, expr, modules=[{"sin": xp.sin, "cos": xp.cos}])
    return lambda x, y: f(x, y) + 0 * x


def _velocity(n, periodic, backend):
    grid = CartesianGrid((n, n), length=(2 * math.pi, 2 * math.pi), origin=(0.3, -0.2), backend=backend)
    exact = [_fn(e, grid.xp) for e in U]
    bcs = [BoundaryConditions.periodic(2) if periodic else BoundaryConditions.uniform(2, Dirichlet(exact[j]))
           for j in range(2)]
    u = VectorField([CellField.from_interior(grid, exact[j](*grid.meshgrid()), bcs[j]) for j in range(2)])
    uf = FaceField(grid, [exact[k](*grid.face_meshgrid(k)) for k in range(2)])
    return grid, u, uf


def _tendency_errors(term, target, periodic, backend, ns=(16, 32, 64, 128)):
    errs = []
    for n in ns:
        grid, u, uf = _velocity(n, periodic, backend)
        out = term.tendency(u, uf)
        x, y = grid.meshgrid()
        errs.append(max(float(abs(to_numpy(out[j] - _fn(target[j], grid.xp)(x, y))).max()) for j in range(2)))
    return errs, list(ns)


@pytest.mark.parametrize("periodic,min_order", [(True, 6.2), (False, 5.4)])
def test_uccd6_sixth_order(backend, periodic, min_order):
    """Skew CCD form with the hyperviscosity: 6th order, walls included (product closure)."""
    errs, ns = _tendency_errors(UCCD6Convection(), SKEW, periodic, backend)
    assert min(order(errs, ns)) > min_order


@pytest.mark.parametrize("periodic", [True, False])
def test_fuccd_face_flux_fourth_order_divergence_second(backend, periodic):
    """FUCCD4 face fluxes are 4th order; the conservative face difference D_f makes the
    point-value tendency 2nd order (same structure as TwoPhaseFlow's face_divergence)."""
    term = FUCCDConvection()
    flux_errs = []
    for n in (16, 32, 64):
        grid, u, uf = _velocity(n, periodic, backend)
        u.fill_ghosts()
        for k in range(2):
            a = uf[k]
            flux = a * term.fccd.interp_f(u[0], k) + abs(a) * (term.sigma * grid.h[k] ** 7
                                                                * term.dissipative_gradient(u[0], k))
            exact = _fn(U[k] * U[0], grid.xp)(*grid.face_meshgrid(k))
            if not periodic:  # interior faces (the dissipative wall faces are zero by construction)
                sl = [slice(None)] * 2
                sl[k] = slice(1, -1)
                flux, exact = flux[tuple(sl)], exact[tuple(sl)]
            if k == 0:
                flux_errs.append(0.0)
            flux_errs[-1] = max(flux_errs[-1], float(abs(to_numpy(flux - exact)).max()))
    assert min(order(flux_errs, [16, 32, 64])) > 3.8
    errs, ns = _tendency_errors(term, CONS, periodic, backend, ns=(16, 32, 64))
    assert 1.95 < min(order(errs, ns)) and max(order(errs, ns)) < 2.2


def test_fuccd_differs_from_central_fccd_by_an_h7_dissipation(backend):
    """FUCCD4 = central FCCD flux + a dissipation acting as -sigma |a| h^7 d^8/dx^8."""
    diffs = []
    for n in (32, 64, 128):
        grid, u, uf = _velocity(n, True, backend)
        fu = FUCCDConvection().tendency(u, uf)
        ce = CentralFluxConvection(FCCDFaceComplex()).tendency(u, uf)
        diffs.append(max(float(abs(to_numpy(a - b)).max()) for a, b in zip(fu, ce)))
    assert all(o > 6.5 for o in order(diffs, [32, 64, 128]))


def test_fuccd1_is_first_order_upwind(backend):
    errs, ns = _tendency_errors(FUCCDConvection(r=1), CONS, True, backend, ns=(16, 32, 64))
    assert all(0.9 < o < 1.1 for o in order(errs, ns))


def test_uccd6_energy_budget_periodic(backend):
    """Periodic grid: the skew part conserves kinetic energy to round-off, the hyperviscosity removes it."""
    grid, u, uf = _velocity(32, True, backend)
    term = UCCD6Convection()
    skew_only = UCCD6Convection(sigma=1e-300)
    work = [sum(float((u[j].interior * t[j]).sum()) for j in range(2)) for t in (skew_only.tendency(u), term.tendency(u))]
    scale = sum(float(abs(u[j].interior * skew_only.tendency(u)[j]).sum()) for j in range(2))
    assert abs(work[0]) < 1e-12 * scale
    assert work[1] < 0.0


def test_fuccd_dissipation_symbol_matches_operator():
    """Largest decay rate of the r = 4 dissipative operator = sigma |a| mu_4(pi) / h (paper eq. fuccd_periodic_symbol)."""
    n = 32
    grid = CartesianGrid((n,), length=(1.0,))
    term = FUCCDConvection(sigma=1.0)
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        f = CellField.from_interior(grid, e, BoundaryConditions.periodic(1))
        g = term.dissipative_gradient(f, 0) * grid.h[0] ** 7
        cols.append(-term.fccd.div_f(FaceField(grid, [g])))
    ev = np.linalg.eigvals(np.array(cols).T)
    assert abs(ev.imag).max() < 1e-9
    assert abs(ev.real).max() * grid.h[0] == pytest.approx(term.nyquist_symbol(), rel=1e-12)
    assert term.nyquist_symbol() == pytest.approx(4 * (1 + 9.6 / 24) * 9.6 ** 3)


def _noisy_taylor_green(backend, term, n=32):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    ops = FCCDFaceComplex()
    ns = IMEXBDF2NS(grid, periodic=(0, 1), mu=0.0, ops=ops, tolerance=1e-9, convection=term)
    rng = np.random.default_rng(0)
    ns.u[0].interior = -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y) + 1e-8 * xp.asarray(rng.standard_normal(grid.n))
    ns.u[1].interior = xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y) + 1e-8 * xp.asarray(rng.standard_normal(grid.n))
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    return ns


@pytest.mark.parametrize("fraction,stable", [(0.9, True), (1.1, False)])
def test_uccd6_explicit_stability_bound(backend, fraction, stable):
    """EXT2/BDF2 bound sigma sum_k |u_k|/h_k omega_2(pi)^8 dt <= 4/3 is sharp for UCCD6."""
    term = UCCD6Convection()
    ns = _noisy_taylor_green(backend, term)
    dt = fraction * term.max_stable_dt(ns.u, ns.uf)
    speeds = sum(float(abs(to_numpy(c.interior)).max()) for c in ns.u)
    assert dt == pytest.approx(fraction * (4 / 3) / (1e-3 * OMEGA2_SQ_NYQUIST ** 4 * speeds * ns.grid.n[0]), rel=1e-12)
    ns.fixed_dt = dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.start(events=[300 * dt])
        ns.run(300 * dt)
    energy = ns.kinetic_energy()
    if stable:
        assert energy == pytest.approx(0.25, rel=1e-3)
    else:
        assert not energy < 1.0  # grows without bound (or NaN)


def test_adaptive_step_respects_the_bound_and_fixed_step_warns(backend):
    term = UCCD6Convection()
    ns = _noisy_taylor_green(backend, term)
    ns.start(events=[0.01])
    ns.step()
    assert ns.clock.dt <= term.max_stable_dt(ns.u, ns.uf) * 1.05
    ns2 = _noisy_taylor_green(backend, UCCD6Convection())
    ns2.fixed_dt = 0.01
    with pytest.warns(RuntimeWarning, match="stability bound"):
        ns2.start(events=[0.01])
        ns2.step()


@pytest.mark.parametrize("name", ["central", "uccd6", "fuccd4"])
def test_inviscid_taylor_green_fifth_order_with_fccd(backend, name):
    """Euler steady state: with FCCD operators the IMEX-BDF2 solution converges at ~5th order in space."""
    errs = []
    for n in (16, 32):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"}, "terms": {"convection": name}})
        ns = IMEXBDF2NS(grid, periodic=(0, 1), mu=0.0, ops=scheme.face, tolerance=1e-11, fixed_dt=0.25 / round(5 * n),
                        convection=scheme.convection)
        ex = [-xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y), xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)]
        for a in range(2):
            ns.u[a].interior = ex[a]
        ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
        ns.p.fill_ghosts()
        ns.centered_gradient()
        ns.start(events=[0.25])
        ns.run(0.25)
        errs.append(max(float(xp.max(xp.abs(ns.u[a].interior - ex[a]))) for a in range(2)))
    assert math.log2(errs[0] / errs[1]) > 4.5


@pytest.mark.parametrize("name,rel", [("uccd6", 0.015), ("fuccd4", 0.01)])
def test_lid_cavity_with_walls(backend, name, rel):
    from basilisk_cupy import Dirichlet as D
    from cases import lid_cavity

    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: D(0.0) for w in walls}
    tang["top"] = D(1.0)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"}, "terms": {"convection": name}})
    ns = IMEXBDF2NS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                    convection=scheme.convection)
    ns.start(events=[round(0.1 * k, 10) for k in range(1, 12)])
    ns.run(1.1)
    assert ns.kinetic_energy() == pytest.approx(lid_cavity.REFERENCE[-1][1], rel=rel)
    assert ns.max_divergence() < 1e-5


@pytest.mark.slow
def test_double_shear_layer_needs_dissipation(backend):
    """Under-resolved inviscid shear layer (N = 64): the central FCCD flux gains energy, UCCD6/FUCCD4 do not."""
    out = {}
    for name in ("central", "uccd6", "fuccd4"):
        grid = CartesianGrid((64, 64), backend=backend)
        xp = grid.xp
        x, y = grid.meshgrid()
        scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"}, "terms": {"convection": name}})
        ns = IMEXBDF2NS(grid, periodic=(0, 1), mu=0.0, ops=scheme.face, convection=scheme.convection, cfl=0.4)
        ns.u[0].interior = xp.where(y <= 0.5, xp.tanh(80 * (y - 0.25)), xp.tanh(80 * (0.75 - y)))
        ns.u[1].interior = 0.05 * xp.sin(2 * math.pi * x)
        times = [0.25 * k for k in range(1, 7)]
        energies, peaks = [], []
        ns.start(events=times)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in times:
                ns.run(t)
                energies.append(ns.kinetic_energy())
                peaks.append(float(xp.max(xp.abs(ns.u[0].interior))))
        out[name] = (energies, max(peaks))
    # inviscid: the energy may only decrease (sampled every 0.25 up to t = 1.5)
    central_e, central_peak = out["central"]
    assert max(central_e) > central_e[0] * 1.02 and central_peak > 2.5
    for name in ("uccd6", "fuccd4"):
        e, peak = out[name]
        assert all(b <= a for a, b in zip(e, e[1:])) and peak < 2.0


def test_scheme_registry_builds_convection_terms():
    for name, cls in (("central", CentralFluxConvection), ("uccd6", UCCD6Convection), ("fuccd4", FUCCDConvection),
                      ("fuccd1", FUCCDConvection)):
        scheme = build_scheme({"grid": {"cells": [16, 16]}, "operators": {"face": "fccd", "cell": "ccd"},
                               "terms": {"convection": name}})
        assert isinstance(scheme.convection, cls)
    assert build_scheme({}).convection is None
    assert build_scheme({"terms": {"convection": "uccd6", "sigma": 2e-3}}).convection.sigma == 2e-3
    with pytest.raises(ValueError):
        build_scheme({"terms": {"convection": "weno"}})

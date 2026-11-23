"""Face-native state (``face_native=True``): the projected face velocity is the state, cells are ``R_h uf``."""

import math
import warnings

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid, Dirichlet
from basilisk_cupy.backend import to_numpy
from basilisk_cupy.scheme import build_scheme
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS
from basilisk_cupy.time.imex_rk import IMEXRKNS

from .conftest import order


def _taylor_green(cls, n, cfl, t_end, *, backend, face_native, face="fccd", mu=0.01, noise=0.0):
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    x, y = grid.meshgrid()
    scheme = build_scheme({"operators": {"face": face, "cell": "ccd" if face == "fccd" else "fd2"},
                           "terms": {"convection": "central", "viscosity": "ccd" if face == "fccd" else "basilisk"}})
    ns = cls(grid, periodic=(0, 1), mu=mu, ops=scheme.face, convection=scheme.convection, viscous=scheme.viscous,
             fixed_dt=t_end / max(1, round(t_end * n / cfl)), tolerance=1e-11, face_native=face_native)
    ex = [lambda x, y: -xp.cos(2 * math.pi * x) * xp.sin(2 * math.pi * y),
          lambda x, y: xp.sin(2 * math.pi * x) * xp.cos(2 * math.pi * y)]
    rng = np.random.default_rng(0)
    for a in range(2):
        ns.u[a].interior = ex[a](x, y) + noise * xp.asarray(rng.standard_normal(grid.n))
    ns.p.interior = -(xp.cos(4 * math.pi * x) + xp.cos(4 * math.pi * y)) / 4
    ns.p.fill_ghosts()
    ns.centered_gradient()
    ns.start(events=[t_end])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns.run(t_end)
    d = math.exp(-8 * math.pi ** 2 * mu * t_end)
    cells = max(float(xp.max(xp.abs(ns.u[a].interior - d * ex[a](x, y)))) for a in range(2))
    faces = max(float(xp.max(xp.abs(ns.uf[a] - d * ex[a](*grid.face_meshgrid(a))))) for a in range(2))
    return ns, cells, faces


def test_error_no_longer_depends_on_the_step(backend):
    """IMEX-RK + FCCD, N = 32: the O(dt h^p) term of the approximate projection is gone."""
    _, fn_small, _ = _taylor_green(IMEXRKNS, 32, 0.1, 0.25, backend=backend, face_native=True)
    _, fn_large, _ = _taylor_green(IMEXRKNS, 32, 1.5, 0.25, backend=backend, face_native=True)
    _, cs_small, _ = _taylor_green(IMEXRKNS, 32, 0.1, 0.25, backend=backend, face_native=False)
    _, cs_large, _ = _taylor_green(IMEXRKNS, 32, 1.5, 0.25, backend=backend, face_native=False)
    assert fn_large == pytest.approx(fn_small, rel=0.05)
    assert cs_large > 5 * cs_small


def test_fourth_order_in_space_at_cfl_one(backend):
    cells, faces = [], []
    for n in (16, 32):
        _, c, f = _taylor_green(IMEXRKNS, n, 1.0, 0.25, backend=backend, face_native=True)
        cells.append(c)
        faces.append(f)
    assert math.log2(cells[0] / cells[1]) > 3.8 and math.log2(faces[0] / faces[1]) > 3.8


def test_bdf2_second_order_in_time_with_fd2(backend):
    """With fd2 the cell-state IMEX-BDF2 is 1st order in dt (cross term O(dt h^2)); the face state restores 2nd."""
    t_end, ks = 0.25, [5, 10, 20]
    out = {}
    for fn in (False, True):
        ref, _, _ = _taylor_green(IMEXBDF2NS, 32, 32 * t_end / 160, t_end, backend=backend, face_native=fn, face="fd2")
        errs = []
        for k in ks:
            ns, _, _ = _taylor_green(IMEXBDF2NS, 32, 32 * t_end / k, t_end, backend=backend, face_native=fn, face="fd2")
            errs.append(max(float(abs(to_numpy(ns.u[a].interior - ref.u[a].interior)).max()) for a in range(2)))
        out[fn] = order(errs, ks)
    assert max(out[False]) < 1.2 and min(out[True]) > 1.8


def test_state_is_divergence_free_and_cells_are_its_reconstruction(backend):
    ns, _, _ = _taylor_green(IMEXRKNS, 16, 1.0, 0.1, backend=backend, face_native=True)
    assert ns.max_divergence() < 1e-9
    for a, v in enumerate(ns._cells_from_faces(ns.uf)):
        np.testing.assert_array_equal(to_numpy(ns.u[a].interior), to_numpy(v))


def test_inviscid_large_steps(backend):
    """Inviscid, noisy, N = 32, t = 1, CFL 2: face state 5.5e-5 (as at CFL 0.1), cell state 5.9e-4."""
    _, cells, _ = _taylor_green(IMEXRKNS, 32, 2.0, 1.0, backend=backend, face_native=True, mu=0.0, noise=1e-6)
    assert cells < 1e-4


def test_lid_cavity_with_walls(backend):
    grid = CartesianGrid((64, 64), origin=(-0.5, -0.5), backend=backend)
    walls = ("left", "right", "bottom", "top")
    tang = {w: Dirichlet(0.0) for w in walls}
    tang["top"] = Dirichlet(1.0)
    scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                           "terms": {"convection": "central", "viscosity": "ccd"}})
    ns = IMEXRKNS(grid, tangential=tang, uf_normal={w: 0.0 for w in walls}, mu=1e-3, ops=scheme.face,
                  convection=scheme.convection, viscous=scheme.viscous, fixed_dt=1.0 / 64, face_native=True)
    ns.start(events=[1.1])
    ns.run(1.1)
    assert ns.kinetic_energy() == pytest.approx(0.009131, rel=0.02)  # IMEX-BDF2, cell state, CFL 0.25
    assert ns.max_divergence() < 1e-5

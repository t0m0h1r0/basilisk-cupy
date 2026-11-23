"""Fused GPU kernels (basilisk_cupy/kernels) give bitwise the result of the array path."""

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, FaceField, Neumann
from basilisk_cupy.backend import cupy_available
from basilisk_cupy.boundary import LOWER, UPPER, Periodic
from basilisk_cupy.fields import VectorField
from basilisk_cupy.linear.multigrid import prolong_bilinear
from basilisk_cupy.linear.poisson import HelmholtzProblem
from basilisk_cupy.linear.viscosity import ViscosityProblem

pytestmark = pytest.mark.skipif(not cupy_available(), reason="cupy not installed")


@pytest.fixture
def cp():
    import cupy

    return cupy


@pytest.mark.parametrize("n,lam", [((32,), 1.0), ((16, 12), None), ((16, 12), -2.0), ((8, 6, 10), -2.0),
                                   ((8, 6, 10), None)])
def test_poisson_relaxation_kernel(cp, n, lam):
    out = []
    for fused in (False, True):
        rng = np.random.default_rng(1)
        grid = CartesianGrid(n, length=(1.0, 2.0, 0.5)[:len(n)], backend="cupy")
        alpha = FaceField(grid, [cp.asarray(1 + rng.random(grid.face_shape(a))) for a in range(len(n))])
        problem = HelmholtzProblem(CellField(grid), alpha, lam)
        assert problem.fused
        problem.fused = fused
        da = CellField(problem.levels[-1])
        da.data[...] = cp.asarray(rng.random(da.data.shape))
        problem.relax([da], [cp.asarray(rng.random(grid.n))], len(problem.levels) - 1)
        out.append(cp.asnumpy(da.data))
    np.testing.assert_array_equal(out[0], out[1])


@pytest.mark.parametrize("n,rho_array", [((32,), False), ((16, 12), False), ((16, 16), True), ((8, 6, 10), True)])
def test_viscosity_relaxation_kernel(cp, n, rho_array):
    out = []
    for fused in (False, True):
        rng = np.random.default_rng(1)
        grid = CartesianGrid(n, length=tuple(v / 16 for v in n), backend="cupy")
        mu = FaceField(grid, [cp.asarray(1 + rng.random(grid.face_shape(a))) for a in range(len(n))])
        rho = cp.asarray(1 + rng.random(grid.n)) if rho_array else 1.3
        problem = ViscosityProblem(VectorField.zeros(grid, [BoundaryConditions() for _ in n]), mu, rho, 0.01)
        assert problem.fused
        problem.fused = fused
        da = [CellField(grid) for _ in n]
        for c in da:
            c.data[...] = cp.asarray(rng.random(c.data.shape))
        problem.relax(da, [cp.asarray(rng.random(grid.n)) for _ in n], len(problem.levels) - 1)
        out.append([cp.asnumpy(c.data) for c in da])
    for a, b in zip(*out):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n,ghosts", [((7,), 1), ((6, 5), 1), ((6, 5), 2), ((4, 5, 6), 1), ((5, 4, 6), 3)])
def test_ghost_kernel(cp, n, ghosts):
    rng = np.random.default_rng(0)
    conds = [Dirichlet(1.5), Neumann(-0.7), Dirichlet(0.0), Neumann(0)]
    grid = CartesianGrid(n, length=tuple(0.3 * (a + 1) for a in range(len(n))), ghosts=ghosts, backend="cupy")
    for _ in range(20):
        bc = BoundaryConditions()
        for ax in range(len(n)):
            if rng.random() < 0.3:
                bc.set((ax, LOWER), Periodic()).set((ax, UPPER), Periodic())
            else:
                for side in (LOWER, UPPER):
                    bc.set((ax, side), conds[rng.integers(len(conds))])
        assert bc._kernel_sides(len(n)) is not None
        fused = cp.asarray(rng.random(grid.padded_shape))
        reference = fused.copy()
        bc.apply(grid, fused)
        for ax in range(len(n)):
            for side in (LOWER, UPPER):
                bc.get(ax, side).fill(grid, reference, ax, side)
        np.testing.assert_array_equal(cp.asnumpy(fused), cp.asnumpy(reference))


def test_ghost_kernel_falls_back_for_callable_values():
    bc = BoundaryConditions.uniform(2, Dirichlet(lambda x, y: x + y))
    assert bc._kernel_sides(2) is None


@pytest.mark.parametrize("n,ghosts", [((5,), 1), ((4, 6), 1), ((4, 6), 2), ((3, 4, 5), 1), ((4, 4, 4), 2)])
def test_prolongation_kernel(cp, n, ghosts):
    data = np.random.default_rng(0).random(tuple(v + 2 * ghosts for v in n))
    host = prolong_bilinear(CellField(CartesianGrid(n, ghosts=ghosts), data=data.copy()))
    dev = prolong_bilinear(CellField(CartesianGrid(n, ghosts=ghosts, backend="cupy"), data=cp.asarray(data)))
    np.testing.assert_array_equal(host, cp.asnumpy(dev))


# -- CUDA graphs of the multigrid cycle (linear/cycle_graph.py) -----------------------------
def _flow_runs(cp):
    from basilisk_cupy.discretization import FCCDFaceComplex
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS
    from cases import lid_cavity

    from .test_imex_bdf2 import taylor_green

    out = []
    for cls, kw in ((CenteredNS, {}), (IMEXBDF2NS, dict(ops=FCCDFaceComplex()))):
        ns = taylor_green(cls, 32, 2e-3, 0.03, backend="cupy", **kw)
        out += [cp.asnumpy(c.interior) for c in ns.u] + [cp.asnumpy(ns.p.interior)]
    ns, _ = lid_cavity.run(32, 0.2, "cupy", log=None)  # adaptive dt, moving lid
    return out + [cp.asnumpy(c.interior) for c in ns.u]


def test_cycle_graphs_are_recorded_and_bitwise_equal(cp, monkeypatch):
    from basilisk_cupy.linear import cycle_graph

    counts = {"record": 0, "replay": 0}
    record, run = cycle_graph.CycleGraphs._record, cycle_graph.CycleGraphs.run

    def counting_record(self, nrelax):
        counts["record"] += 1
        return record(self, nrelax)

    def counting_run(self, nrelax):
        counts["replay"] += nrelax in self.graphs
        return run(self, nrelax)

    monkeypatch.setattr(cycle_graph.CycleGraphs, "_record", counting_record)
    monkeypatch.setattr(cycle_graph.CycleGraphs, "run", counting_run)
    with_graphs = _flow_runs(cp)
    assert counts["record"] > 0 and counts["replay"] > 5 * counts["record"]
    monkeypatch.setenv("BASILISK_CUPY_GRAPHS", "0")
    without = _flow_runs(cp)
    for a, b in zip(with_graphs, without):
        np.testing.assert_array_equal(a, b)


def test_reused_problems_equal_fresh_ones(cp, monkeypatch):
    reused = _flow_runs(cp)
    monkeypatch.setattr(HelmholtzProblem, "reuse", classmethod(
        lambda cls, a, alpha, lam=None, *, ops=None, min_cells=2: cls(a, alpha, lam, ops=ops, min_cells=min_cells)))
    monkeypatch.setattr(ViscosityProblem, "reuse", classmethod(
        lambda cls, u, mu, rho, dt, min_cells=2: cls(u, mu, rho, dt, min_cells=min_cells)))
    fresh = _flow_runs(cp)
    for a, b in zip(reused, fresh):
        np.testing.assert_array_equal(a, b)


def test_reuse_does_not_write_into_the_callers_coefficients(cp):
    from basilisk_cupy import poisson

    grid = CartesianGrid((16, 16), backend="cupy")
    a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
    b = cp.ones(grid.n)
    first = FaceField(grid, [cp.full(grid.face_shape(ax), 2.0) for ax in range(2)])
    poisson(a, b, first, tolerance=1e-8)
    poisson(a, b, FaceField(grid, [cp.full(grid.face_shape(ax), 5.0) for ax in range(2)]), tolerance=1e-8)
    assert all(float(c.min()) == float(c.max()) == 2.0 for c in first.components)


@pytest.mark.parametrize("n", [(8, 8, 8), (16, 8, 16)])
def test_cycle_graphs_in_3d(cp, monkeypatch, n):
    import math

    from basilisk_cupy import poisson

    out = []
    for flag in ("1", "0"):
        monkeypatch.setenv("BASILISK_CUPY_GRAPHS", flag)
        grid = CartesianGrid(n, backend="cupy")
        x, y, z = grid.meshgrid()
        a = CellField(grid, bc=BoundaryConditions.uniform(3, Dirichlet(0.0)))
        poisson(a, cp.sin(math.pi * x) * cp.sin(2 * math.pi * y) * z, tolerance=1e-9)
        out.append(cp.asnumpy(a.interior))
    np.testing.assert_array_equal(out[0], out[1])

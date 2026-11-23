"""Multigrid cycles replayed as CUDA graphs (cupy).

One Basilisk cycle on a 512^2 grid is ~100 small kernels (restriction,
ghost filling, relaxation and prolongation on every level). Launched from
Python each costs ~10-20 us of host time while the GPU work of the coarse
levels is a few microseconds, so the GPU mostly waits. Recording the cycle
once as a CUDA graph and replaying it removes the per-launch host cost.

The recorded cycle does exactly what :func:`multigrid.mg_cycle` does, with
the same kernels in the same order, but writes into buffers allocated
beforehand (residual and correction on every level, the fields' own arrays),
since a graph keeps the device pointers it was recorded with. The memory
allocator is switched off while recording, so an operation that would
allocate raises instead of corrupting memory later.

Per ``nrelax`` the first cycle runs normally (it compiles the kernels), the
second one is recorded, and later ones replay the graph. The recorded cycles
and their buffers are kept on the problem object, so repeated ``mg_solve``
calls for the same problem and unknowns (the preconditioner of the defect
correction) replay them too. The buffers' contents between calls do not
matter: a cycle zeroes the coarsest correction and overwrites the others.

When the problem has a device residual (fd2 Poisson--Helmholtz, viscosity:
kernels/residual_gpu.py) and the unknowns' boundary conditions suit the ghost
kernel, the graph also refills the unknowns' ghosts and computes the residual
and its maximum norm, so a whole ``mg_solve`` iteration is one graph launch and
one 8-byte read. The right-hand side is copied into the graph's own buffer at
the start of each ``mg_solve``; the unknowns' conditions are compared with the
ones recorded (their constant values are baked into the graph).
"""

from __future__ import annotations

import os

from ..boundary import BoundaryConditions
from ..fields import CellField


class _AllocationDuringCapture(RuntimeError):
    pass


def _forbid_allocation(size):  # pragma: no cover - only reached by a bug
    raise _AllocationDuringCapture(f"device allocation of {size} bytes while recording a CUDA graph")


def graphs_enabled() -> bool:
    """CUDA graphs are on unless ``BASILISK_CUPY_GRAPHS=0``."""
    return os.environ.get("BASILISK_CUPY_GRAPHS", "1") != "0"


def supported(a: list[CellField], problem, da_levels) -> bool:
    """Whether the cycle for ``problem`` can be recorded (fused kernels, contiguous buffers,
    correction conditions the ghost kernel handles)."""
    grid = a[0].grid
    if grid.xp.__name__ != "cupy" or not graphs_enabled() or not getattr(problem, "fused", False):
        return False
    for bc in problem.correction_bcs:
        if bc._kernel_sides(grid.ndim) is None:
            return False
    fields = list(a) + [da for das in da_levels for da in das]
    return all(f.data.flags.c_contiguous and f.data.dtype == grid.xp.float64 for f in fields)


def _residual_in_graph(a: list[CellField], problem) -> bool:
    supported_ = getattr(problem, "device_residual_supported", None)
    if supported_ is None or not supported_():
        return False
    return all(s.bc._kernel_sides(s.grid.ndim) is not None for s in a)


def cached(problem, a: list[CellField]):
    """The :class:`CycleGraphs` recorded earlier for ``problem`` and the same unknowns
    (same arrays, same boundary conditions, same residual operator)."""
    graphs = getattr(problem, "_cycle_graphs", None)
    if graphs is None or len(graphs.a) != len(a) or any(s.data is not c.data for s, c in zip(a, graphs.a)):
        return None
    if [s.bc for s in a] != graphs.bcs or _residual_in_graph(a, problem) != graphs.with_residual:
        return None
    return graphs


class CycleGraphs:
    """Recorded multigrid cycles for one ``mg_solve`` call, keyed by ``nrelax``."""

    def __init__(self, a: list[CellField], problem, da_levels, res_levels) -> None:
        import cupy as cp  # noqa: PLC0415

        self.cp = cp
        self.a, self.problem, self.da_levels = a, problem, da_levels
        levels = problem.levels
        # residual buffers on every level (the finest one is mg_solve's)
        for lev in range(len(levels) - 1):
            res_levels[lev] = [cp.zeros(levels[lev].n, dtype=cp.float64) for _ in a]
        self.res_levels = res_levels
        self.seen: set[int] = set()
        self.graphs: dict = {}
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.bcs = [BoundaryConditions(dict(s.bc.sides)) for s in a]  # snapshot: values are baked in
        self.with_residual = _residual_in_graph(a, problem)
        if self.with_residual:
            self.b = [cp.zeros(levels[-1].n, dtype=cp.float64) for _ in a]
            self.norm = cp.zeros(1, dtype=cp.uint64)
        problem._cycle_graphs = self

    def load_rhs(self, b: list) -> None:
        """Copy the right-hand side into the graph's buffer (start of each ``mg_solve``)."""
        if self.with_residual:
            for dst, src in zip(self.b, b):
                dst[...] = src

    # the cycle, allocation free -------------------------------------------------------
    def _cycle(self, nrelax: int) -> None:
        from ..kernels.transfer_gpu import prolong, restrict  # noqa: PLC0415

        problem, res_levels, da_levels = self.problem, self.res_levels, self.da_levels
        nlev = len(problem.levels)
        for lev in range(nlev - 2, -1, -1):
            for fine, coarse in zip(res_levels[lev + 1], res_levels[lev]):
                restrict(fine, coarse)
        for lev in range(nlev):
            das = da_levels[lev]
            if lev == 0:
                for da in das:
                    da.data.fill(0.0)
            else:
                for da, coarse in zip(das, da_levels[lev - 1]):
                    coarse.fill_ghosts()
                    prolong(coarse.data, coarse.grid.n, coarse.grid.ghosts, out=da.data, out_ghosts=da.grid.ghosts)
            for _ in range(nrelax):
                for da in das:
                    da.fill_ghosts()
                problem.relax(das, res_levels[lev], lev)
        for s, ds in zip(self.a, da_levels[-1]):
            target = s.interior
            target += ds.interior
        if self.with_residual:
            for s in self.a:
                s.fill_ghosts()
            self.norm.fill(0)
            problem.device_residual(self.a, self.b, res_levels[-1], self.norm)

    def _record(self, nrelax: int):
        cp = self.cp
        pool = cp.get_default_memory_pool()
        caller = cp.cuda.get_current_stream()
        self.stream.wait_event(caller.record())
        with self.stream:
            cp.cuda.set_allocator(_forbid_allocation)
            try:
                self.stream.begin_capture()
                try:
                    self._cycle(nrelax)
                finally:
                    graph = self.stream.end_capture()
            finally:
                cp.cuda.set_allocator(pool.malloc)
        return graph

    def run(self, nrelax: int) -> float | None:
        """One cycle; returns the residual norm when the graph computes it, else None."""
        graph = self.graphs.get(nrelax)
        if graph is None and nrelax not in self.seen:
            self.seen.add(nrelax)
            self._cycle(nrelax)
        else:
            if graph is None:
                graph = self.graphs[nrelax] = self._record(nrelax)
            graph.launch(stream=self.cp.cuda.get_current_stream())
        if not self.with_residual:
            return None
        return float(self.norm.view(self.cp.float64)[0])

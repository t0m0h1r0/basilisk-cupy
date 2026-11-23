# basilisk-cupy

Python/CuPy port of the [Basilisk](http://basilisk.fr) flow solver, structured so
that every numerical ingredient (spatial operators, linear solvers, equation
terms, pressure projection, interface method, time stepping) can be swapped
for an alternative and validated against the Basilisk-compatible baseline.

Design document: see the project's design doc ("Basilisk → CuPy 移植設計書").

## Status

| Layer | Implemented |
| --- | --- |
| Grid, fields, boundary conditions | uniform cell-centred grid (1D/2D/3D), Dirichlet / Neumann / periodic with automatic homogeneous versions |
| ① Spatial operators | `fd2` (Basilisk-compatible 2nd order); `ccd` (CCD, Chu & Fan: f′ 6th, f″ 5th order with walls) and `fccd` (FCCD face operators, 4th order) on the cell-centred grid with new wall closures (`tools/derive_ccd_closures.py`) |
| ② Linear solvers | Basilisk-style multigrid (`mg_solve`, scalar or vector unknowns, red/black Gauss–Seidel, adaptive `nrelax`) with a separate high-order residual hook; GCR-accelerated defect correction for high-order operators (`poisson(method="dc")`, chosen automatically for non-fd2 `ops`); implicit viscosity (`viscosity.h`) |
| ③ Terms | BCG advection and `prediction()` (`bcg.h`), minmod / superbee / sweby limiters |
| ④ Projection | Basilisk `project()` (approximate projection, variable density) |
| ⑥ Time stepping | `timestep()` / `dtnext()`; `CenteredNS` (`navier-stokes/centered.h`) with Basilisk's event order exposed as hook stages; `IMEXBDF2NS` (TwoPhaseFlow: EXT2 convection, implicit BDF2 viscosity, incremental pressure correction with projection step 2dt/3) |
| ③ Terms | `CentralFluxConvection` (semi-discrete `D_f(P_f(u) u_f)` for multistep integrators) |
| Scheme | `scheme.py` registry and YAML loader |

## Usage

```python
import math
from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField, Dirichlet, poisson

grid = CartesianGrid((128, 128), backend="numpy")   # or "cupy"
x, y = grid.meshgrid()
b = -2 * math.pi**2 * grid.xp.sin(math.pi * x) * grid.xp.sin(math.pi * y)
a = CellField(grid, bc=BoundaryConditions.uniform(2, Dirichlet(0.0)))
stats = poisson(a, b, tolerance=1e-8)
```

## Validation against Basilisk

| Case | Basilisk test | Result |
| --- | --- | --- |
| Taylor–Green vortices, t = 2 | `src/test/taylor-green.c` / `.ref` | avg and rms errors within 0.5 % and 1.5 % of the reference at N = 32…256, third-order convergence |
| Lid-driven cavity, Re = 1000, N = 64 | `src/test/lid.c` / `lid.ref` | all 440 logged energies within 0.07 % (4-digit rounding); steady state reached at t = 44.0 as in Basilisk; centreline profiles within 0.015 of Ghia et al. |

```bash
python -m cases.taylor_green --nmax 256
python -m cases.lid_cavity --out results/lid
```

## Swapping operators: CCD / FCCD

```python
from basilisk_cupy.discretization import FCCDFaceComplex
ns = CenteredNS(grid, periodic=(0, 1), ops=FCCDFaceComplex())   # or operators: {face: fccd} in a scheme file
```

| Check | fd2 | fccd |
| --- | --- | --- |
| Poisson, projection setting (rhs = D_f of face gradients), walls | 2nd order | 4th order (defect correction: FCCD residual, fd2 multigrid) |
| Lid cavity Re = 1000, N = 64, steady state | t = 44.0, max diff. from Ghia 0.0148 | t = 43.9, max diff. from Ghia 0.0138 |
| Defect-correction cost, N = 256, Dirichlet walls | — | 22 GCR steps / 44 MG cycles (plain correction: 400 cycles, not converged) |
| Poisson, pointwise rhs | 2nd order | 2nd order, error halved (D_f is the conservative difference, as in TwoPhaseFlow) |
| Taylor–Green t = 2, N = 32 / 64 / 128, rms error | 3.2e-2 / 4.5e-3 / 5.9e-4 | 9.4e-3 / 1.3e-3 / 2.0e-4 (BCG advection unchanged) |

## Time integration: IMEX-BDF2 + IPC

| Check (Taylor–Green, self-convergence in dt) | Basilisk `CenteredNS` | `IMEXBDF2NS` |
| --- | --- | --- |
| Unsteady Stokes | 1st order (backward-Euler viscosity) | 2nd order (2.02) |
| Navier–Stokes, FCCD, N = 32 | 1st order | ≈ 2nd order (2.3–2.5) |
| Navier–Stokes, fd2 | 1st order | 1st order: cross term O(dt h²) of the approximate projection |
| Lid cavity, t = 1.1 energy | 0.008904 (lid.ref) | 0.00887 |

With the same operators, the incremental (δp) and full-pressure projections are
identical to round-off (Basilisk starts the Poisson iteration from pⁿ), so IPC
only changes results together with BDF2 or variable density.

## GPU (cupy)

On the GPU the multigrid hot loops run as fused `cupy.RawKernel`s
(`basilisk_cupy/kernels/`): red/black relaxation of the Poisson–Helmholtz and
viscosity problems, ghost-layer filling for constant Dirichlet/Neumann/periodic
conditions, restriction and prolongation. They reproduce the array path bitwise
(`--fmad=false`, same operation order; checked in `tests/test_gpu_kernels.py`).
Other conditions (callable values) fall back to the array path.

Each multigrid cycle is then recorded once as a CUDA graph and replayed
(`linear/cycle_graph.py`), which removes the ~10–20 µs host cost of every
launch. The cycle writes only into buffers allocated beforehand (allocation
is switched off while recording), and the Poisson, viscosity and
defect-correction problems are kept with their unknown field and reused with
refreshed coefficients, so the recorded graphs survive from one time step to
the next (dt and a constant ρ are read from device memory). Results are
bitwise those without graphs; `BASILISK_CUPY_GRAPHS=0` turns them off.

Cell restriction sums the 2^d children in C order and divides by 2^d, as
Basilisk's `restriction_average`, on every backend.

Taylor–Green, ms per time step (RTX 3080 Ti vs one core of a Ryzen 9 5950X):

| Solver | N | numpy | cupy | speed-up |
| --- | --- | --- | --- | --- |
| `CenteredNS` (fd2) | 64 | 90.3 | 34.7 | 2.6 |
| `CenteredNS` (fd2) | 256 | 420.3 | 47.4 | 8.9 |
| `CenteredNS` (fd2) | 512 | 1267.9 | 71.7 | 17.7 |
| `IMEXBDF2NS` (FCCD) | 64 | 98.2 | 46.9 | 2.1 |
| `IMEXBDF2NS` (FCCD) | 256 | 678.5 | 58.4 | 11.6 |

3D Poisson (Dirichlet, tolerance 1e-8, 12 cycles), one solve:

| N | numpy | cupy | speed-up |
| --- | --- | --- | --- |
| 32³ | 74.2 ms | 8.9 ms | 8.3 |
| 64³ | 364.4 ms | 18.2 ms | 20.0 |
| 128³ | 4902.0 ms | 101.8 ms | 48.2 |

## Tests

```bash
pip install -e ".[dev]"      # add ",gpu" for cupy
pytest              # fast suite
pytest -m slow      # full lid-cavity run to steady state (~1 min)
```

Tests run on numpy; when cupy is importable every backend-parametrised test
(all solver, Poisson, CCD/FCCD, defect-correction and IMEX-BDF2 tests, and the
slow lid-cavity run) also runs on cupy, together with the kernel tests.

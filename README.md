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

## Tests

```bash
pip install -e ".[dev]"      # add ",gpu" for cupy
pytest              # fast suite
pytest -m slow      # full lid-cavity run to steady state (~1 min)
```

Tests run on numpy; when cupy is importable every backend-parametrised test
also runs on cupy and a numpy/cupy agreement test is enabled.

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
| ③ Terms | semi-discrete convection for multistep integrators (`tendency(u, uf)`): `CentralFluxConvection` (`D_f(P_f(u) u_f)`), `UCCD6Convection` (skew CCD + `σ|u|h⁷(−D₂)⁴`), `FUCCDConvection` (FUCCD₄ / FUCCD₁ conservative upwind face fluxes) |
| Scheme | `scheme.py` registry and YAML loader (`operators: face/cell`, `terms: convection`) |

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

## Convection: UCCD6 / FUCCD

```python
scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"}, "terms": {"convection": "uccd6"}})
ns = IMEXBDF2NS(grid, periodic=(0, 1), ops=scheme.face, convection=scheme.convection)
```

From TwoPhaseFlow paper §4.5 (`04d_uccd6.tex`) and §4.7 (`04e1_fuccd.tex`), with
σ = 1e-3 as there. FUCCD takes the projected face velocity `uf` as the face
speed (the paper's choice; `face_velocity="interpolated"` gives TwoPhaseFlow's
`P_f u` sampling).

| Check | central (FCCD flux) | UCCD6 | FUCCD₄ |
| --- | --- | --- | --- |
| Tendency vs exact, periodic, N = 16…128 | 2nd order (`D_f`) | 6.4–7.1 | 2nd order (`D_f`); face fluxes 4th |
| Tendency vs exact, Dirichlet walls | 2nd order | 5.6–6.3 (product `u_k u_j` with Dirichlet wall data) | 2nd order |
| Explicit (EXT2/BDF2) stability bound, σ = 1e-3 | — | CFL ≤ 0.157 per axis (`σ ω₂(π)⁸ CFL ≤ 4/3`, sharp: 0.9× stable, 1.1× blows up) | CFL ≤ 0.27 (`μ₄(π) = 4954`, matches the operator's eigenvalues) |
| Inviscid Taylor–Green, IMEX-BDF2, FCCD ops, N = 64 | 1.90e-7 (5th order) | 3.55e-7 (5th) | 1.90e-7 (5th) |
| Lid cavity Re = 1000, N = 64, t = 1.1 energy vs lid.ref | 0.21 % | 0.92 % (543 steps) | 0.25 % (315 steps) |
| Inviscid double shear layer, N = 64, t ≤ 1.5 | energy +13 %, max\|u\| 3.8 | energy decreases, max\|u\| 1.67 | energy decreases, max\|u\| 1.63 |

The point-value order of the conservative forms is that of the face difference
`D_f F = (F_{i+1/2} − F_{i−1/2})/h` (2nd order), the same structure as
TwoPhaseFlow's `face_divergence`; the paper's 4th-order claim is for the face
values. The adaptive step of `IMEXBDF2NS` is capped by the term's
`max_stable_dt`; a fixed step above it warns.

With viscosity the Taylor–Green error is limited by the fd2 implicit viscosity
(Basilisk `viscosity()`), whatever the convection and the face operators:
μ = 0.01, N = 64, |u − exact| = 8.1e-5 (fd2 ops) and 6.45e-5 (FCCD ops, all three
convection terms), 2nd order; without viscosity FCCD gives 1.9e-7 against 3.4e-5
for fd2.

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
the next (dt and a constant ρ are read from device memory). For fd2
Poisson–Helmholtz and viscosity problems the graph also contains the residual
and its maximum norm (`kernels/residual_gpu.py`, warp-reduced `atomicMax` on
the bit pattern, exact), so one `mg_solve` iteration is one graph launch and
one 8-byte read; the nrelax heuristic and the stopping test stay on the host,
as in Basilisk. Results are bitwise those without graphs;
`BASILISK_CUPY_GRAPHS=0` turns them off.

Relaxation kernels launch threads for the cells of one colour only and use
32-bit indices when the arrays allow it. On a GeForce card the remaining cost
is double-precision arithmetic (FP64 runs at 1/64 of FP32): the divisions by
h and h² are kept, as in Basilisk, rather than replaced by multiplications
that would change the rounding.

Cell restriction sums the 2^d children in C order and divides by 2^d, as
Basilisk's `restriction_average`, on every backend.

Taylor–Green, ms per time step (RTX 3080 Ti vs one core of a Ryzen 9 5950X):

| Solver | N | numpy | cupy | speed-up |
| --- | --- | --- | --- | --- |
| `CenteredNS` (fd2) | 64 | 89.1 | 29.1 | 3.1 |
| `CenteredNS` (fd2) | 256 | 404.4 | 37.6 | 10.7 |
| `CenteredNS` (fd2) | 512 | 1332.7 | 51.7 | 25.8 |
| `IMEXBDF2NS` (FCCD) | 64 | 98.3 | 41.3 | 2.4 |
| `IMEXBDF2NS` (FCCD) | 256 | 567.4 | 51.4 | 11.0 |

3D Poisson (Dirichlet, tolerance 1e-8, 12 cycles), one solve:

| N | numpy | cupy | speed-up |
| --- | --- | --- | --- |
| 32³ | 75.0 ms | 7.6 ms | 9.9 |
| 64³ | 366.2 ms | 13.8 ms | 26.5 |
| 128³ | 4905.2 ms | 59.3 ms | 82.7 |

## Tests

```bash
pip install -e ".[dev]"      # add ",gpu" for cupy
pytest              # fast suite
pytest -m slow      # full lid-cavity run to steady state (~1 min)
```

Tests run on numpy; when cupy is importable every backend-parametrised test
(all solver, Poisson, CCD/FCCD, defect-correction and IMEX-BDF2 tests, and the
slow lid-cavity run) also runs on cupy, together with the kernel tests.

# basilisk-cupy

Python/CuPy port of the [Basilisk](http://basilisk.fr) flow solver, structured so
that every numerical ingredient (spatial operators, linear solvers, equation
terms, pressure projection, interface method, time stepping) can be swapped
for an alternative and validated against the Basilisk-compatible baseline.

Design document: see the project's design doc ("Basilisk → CuPy 移植設計書").

## Status

| Layer | Implemented |
| --- | --- |
| Grid, fields, boundary conditions | cell-centred grid (1D/2D/3D), uniform or mapped per axis (`grid/mapping.py`: tanh wall clustering, periodic sine stretching, monitor-function equidistribution); Dirichlet / Neumann / periodic with automatic homogeneous versions |
| ① Spatial operators | `fd2` (Basilisk-compatible 2nd order); `ccd` (CCD, Chu & Fan: f′ and f″ 6th order inside; compact wall closures, f′ 4th and f″ 3rd at the wall cells) and `fccd` (FCCD face operators, 4th order) on the cell-centred grid with new wall closures (`tools/derive_ccd_closures.py`) |
| ② Linear solvers | Basilisk-style multigrid (`mg_solve`, scalar or vector unknowns, red/black Gauss–Seidel, adaptive `nrelax`) with a separate high-order residual hook; GCR-accelerated defect correction for high-order operators (`poisson(method="dc")`, chosen automatically for non-fd2 `ops`); implicit viscosity (`viscosity.h`) and its CCD defect-correction version (`CCDViscosity`) |
| ③ Terms | BCG advection and `prediction()` (`bcg.h`), minmod / superbee / sweby limiters |
| ④ Projection | Basilisk `project()` (approximate projection, variable density) |
| ⑥ Time stepping | `timestep()` / `dtnext()`; `CenteredNS` (`navier-stokes/centered.h`) with Basilisk's event order exposed as hook stages; `IMEXBDF2NS` (TwoPhaseFlow: EXT2 convection, implicit BDF2 viscosity, incremental pressure correction with projection step 2dt/3); `IMEXRKNS` (ARS(3,4,3) / ARS(4,4,3) IMEX Runge–Kutta, a projection per stage) |
| ③ Terms | semi-discrete convection for multistep integrators (`tendency(u, uf)`): `CentralFluxConvection` (`D_f(P_f(u) u_f)`), `UCCD6Convection` (skew CCD + `σ|u|h⁷(−D₂)⁴`), `FUCCDConvection` (FUCCD₄ / FUCCD₁ conservative upwind face fluxes) |
| Scheme | `scheme.py` registry and YAML loader (`operators: face/cell`, `terms: convection`) |
| ⑤ Interface (two-phase) | VOF baseline (`vof.h` PLIC advection, `two-phase.h`, `heights.h` / `curvature.h`, `tension.h`, `iforce.h`, `reduced.h`; field-by-field equal to Basilisk's C build on the first steps of `gravity.c` / `capwave.c` and on the curvature tests). TwoPhaseFlow's CLS `psi` (`CLSInterface`: TVD-RK3 transport with FCCD face values and the corrected divergence, mass closure; ξ-SDF / Ridge–Eikonal, compression–diffusion and ACLS re-initialisation; curvature from `psi`, from the logit `phi` or from `psi`'s height functions) |
| Two-phase coupling | `TwoPhaseCLS` on `CenteredNS`: CSF, or the paper's pressure-jump closure as a ghost-fluid method (phase-wise constant densities, GFM coefficient on cut faces; `pressure_jump=True`), with the jump either sharp or in the smooth form `p̃ = p − H j_ext` that FCCD's projection can differentiate (`jump_mode="smooth"`); capillary step bound × √(5/12) for the CCD curvature (stable to density ratio 1000 for a static drop); sharp GFM also on mapped grids |
| Coupled iterations | `CenteredNS(coupled_iterations=K, coupled_rtol=..., coupled_anderson=m)`: outer iterations of viscosity, interfacial acceleration and projection within a step, with change / momentum-residual criteria and Anderson acceleration |
| Paper's two-phase step | `TwoPhaseIMEX` (`time/two_phase_imex.py`): IMEX-BDF2 + CLS + pressure-jump GFM as one combination of the existing parts (§11b1's seven stages), options for the face operators, convection and the interface-transport velocity |
| Not yet | split FCCD PPE with a density ratio; HFE; defect correction of the split PPE; FCCD projection with a density ratio (the smooth form assumes equal densities) |

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
| Lid cavity Re = 1000, N = 64, steady state | t = 44.0, max diff. from Ghia 0.0148 | t = 43.9, max diff. from Ghia 0.0139 |
| Defect-correction cost, N = 256, Dirichlet walls | — | residual 1.8e-10 after 19 GCR steps (round-off level: max\|b\| = 4.8e5); reaching the 1e-10 tolerance then took 128 (plain correction: 400 cycles, not converged) |
| Poisson, pointwise rhs | 2nd order | 2nd order, error halved (D_f is the conservative difference, as in TwoPhaseFlow) |
| Taylor–Green t = 2, N = 32 / 64 / 128, rms error | 3.2e-2 / 4.5e-3 / 5.9e-4 | 9.4e-3 / 1.3e-3 / 2.0e-4 (BCG advection unchanged) |

## Wall closures: compact, accuracy from derivatives

The CCD first-cell rows and the FCCD wall faces read the wall datum and the
first two cells only, the width of the interior stencil. Their accuracy comes
from the implicit derivatives, as in the interior rows, not from a wider
one-sided stencil (`tools/derive_ccd_closures.py`, Taylor matching):

| Closure | implicit terms | explicit terms | order at the wall |
| --- | --- | --- | --- |
| CCD Dirichlet / Neumann, row I (`f′₀`) | `f″₀, f′₁, f″₁` | datum, `f₀, f₁` | `f′` 4th |
| CCD Dirichlet, row II (`f″₀`) | `f′₀, f″₁` | datum, `f₀, f₁` | `f″` 3rd |
| CCD Neumann, row II | `f′₁, f″₁` | datum | `f″` 3rd |
| CCD extrapolate (no datum: derivatives of derivatives) | as Dirichlet | `f₀, f₁` | `f′` 3rd, `f″` 2nd |
| FCCD wall gradient (Dirichlet) | — | datum, `f₀, h f′₀, f₁, h f′₁` | 4th |
| FCCD wall value (Neumann) | — | `h` datum, `f₀, h f′₀, h² f″₀` | 4th |

A wall row one order below the interior (two for a second derivative) keeps
the global order (Gustafsson 1975). A row II with `f′₁` as well would be the
row I relation again (singular system), so 3rd order is the most these
three-term rows give. The earlier wide closures (datum + 4 to 6 cells, `f′`
6th and `f″` 5th at the wall) were measured against them:

| Check | wide (before) | compact |
| --- | --- | --- |
| CCD 1-D Dirichlet, `f′` / `f″`, N = 16 → 256 | 8.1e-7 → 1.1e-11 / 1.1e-5 → 2.1e-9 (N = 256 worse than 128: round-off) | 7.9e-5 → 1.5e-9 / 9.3e-4 → 2.9e-7 (4.0 / 3.0) |
| `Re(−D1 h)` max, Dirichlet (advection eigenvalue; > 0 grows) | +0.096 | +1e-15 |
| FCCD face gradient of a Dirichlet field (max over faces), N = 128 | 2.9e-8 | 1.4e-8 |
| Poisson, projection setting, Dirichlet, N = 16 / 32 / 64 | 1.2e-4 / 4.9e-6 / 1.7e-7 | 7.4e-5 / 2.8e-6 / 1.3e-7 |
| Poisson N = 64: GCR steps / plain defect correction cycles | 17 / 165 | 18 / 461 |
| Taylor–Green in a free-slip box, face state, CCD `R_h`, N = 16 / 32 / 64 | 1.66e-5 / 1.07e-6 / 6.78e-8 | 1.61e-5 / 1.02e-6 / 6.51e-8 |
| Lid cavity E(1.1), N = 64: BDF2 / IMEX-RK cell / face + CCD `R_h` / face + implicit UCCD6 | 0.009131 / 0.009245 / 0.009250 / 0.009246 | 0.009134 / 0.009251 / 0.009251 / 0.009247 |
| Channel decay (CCD viscosity), N = 16 / 32 / 64 | 4.8e-7 / 3.0e-8 / — | 1.1e-6 / 4.0e-8 / 2.0e-9 |
| Boundary layer `exp(−x/0.02)`, N = 32, uniform / tanh | 5.6e-3 / 8.6e-7 | 1.4e-2 / 3.3e-5 |

The operators lose orders at the wall cells (max norm: CCD `f′` 4, `f″` 3,
full stress 3, UCCD6 tendency 4), the Navier–Stokes results do not change.
What is lost: a boundary layer resolved by only a few cells (the tanh case
above), and the plain defect correction converges more slowly (GCR does not).

UCCD6 hyperviscosity: `(D2)^4` with the extrapolating closure in the outer
three factors now has one slightly negative wall eigenvalue, `−1.66 / h⁸`
(Dirichlet; Neumann `−0.020 / h⁸`; the largest is `8493.5 / h⁸`), a growth
rate `1.7 σ|u|/h`. With the wide closures it was ≥ 0, but their Dirichlet
advection had the eigenvalue above, 56 times larger (0.096 against 1.7e-3 at σ = 1e-3). A non-negative factor
needs the field's closure in all four factors (1st order at the wall,
measured) or a symmetrised form (3rd order); neither was adopted.

## Face divergence: 4th order and conservative

FCCD's `div_f` was the plain face difference `D_f` (as fd2 and TwoPhaseFlow's
`face_divergence`). On point values that is `F_x + (h²/24) F_xxx`, 2nd order.
The standard Taylor–Green vortex (`kx = ky`) hides it, because there
`u_xxx + v_yyy = 0`. `ψ = sin(ax) sin(by)` with `a ≠ b` is also an exact
decaying Navier–Stokes solution, and it shows the 2nd order: the projection
forces `D_f u_f = 0` and so moves the exact faces by `O(h²)`.

The fix is the dual of FCCD's face gradient
`(u_R − u_L)/h − (h/24)(q_R − q_L)`. `div_f` differences the flux
`F̂ = F − (h²/24) ∂ₙₙF` (`face_flux`). `∂ₙₙF` comes from the face-line CCD,
the same compact solve as `R_h`. The result is 4th order, the stencil is not
wider, and both terms telescope, so the scheme stays conservative: after a
projection `D_f F̂ = 0` holds exactly in every cell. On a wall, however,
`F̂ = −(h²/24) ∂ₙₙu_n` is not zero although `u_n = 0` (lid cavity: up to
0.04, net zero along the walls), so `F̂` is not yet a flux a VOF advection
can use there. `wall_free_flux(uf, periodic)` (2-D) gives that flux:
`F̂ + curl ψ`, with ψ the running sum of the wall fluxes along the boundary,
Coons-extended inside. It is exactly zero on walls, has the same `D_f`, and
differs from `F̂` by `O(h²)` inside. A tracer next to a wall, 200 upwind
steps: mass change 9.9e-5 with `F̂`, 2.2e-16 with it. On a mapped
axis the difference is divided by `h_ξ/J` at the cell centre. Its sum over
the cells still telescopes; dividing by the cell width would add an `O(h²)`
term.

| Check | plain `D_f` | 4th-order `D_f` |
| --- | --- | --- |
| Divergence of an exact flux, N = 128: walls / tanh × monitor / periodic / sine × sine | 5.1e-5 / 4.0e-4 / 2.0e-4 / 4.2e-4 (2nd) | 1.2e-8 / 3.8e-6 / 1.4e-8 / 1.6e-7 (4th) |
| Projection of the exact `a ≠ b` faces, change, N = 16 / 32 / 64 | 7.7e-3 / 1.9e-3 / 4.8e-4 | 1.7e-4 / 1.1e-5 / 6.8e-7 |
| Central FCCD convection, `a ≠ b`, N = 16 … 128 | 2nd (5.0e-3 at 128) | 4th (9.4e-6) |
| FUCCD4 tendency, periodic / walls | 2nd | 3.9–4.0 / 3.3–3.9 |
| Taylor–Green `a ≠ b` (1, 2), face state, IMEX-RK, CFL 1, N = 16 / 32 / 64 | 1.14e-2 / 3.08e-3 / 7.84e-4 (2nd) | 1.39e-3 / 1.25e-4 / 8.49e-6 (3.9) |
| Same, cell state | 1.13e-2 / 2.86e-3 / 7.81e-4 (2nd) | 7.61e-3 / 3.26e-4 / 1.28e-5 (4.7) |
| Taylor–Green `a = b`, face state, N = 64 | 8.01e-7 | 8.01e-7 |
| Lid cavity E(1.1), IMEX-BDF2, N = 32 / 64 / 128 / 256 | 0.008250 / 0.009134 / 0.009614 / 0.009676 | 0.008293 / 0.009211 / 0.009651 / 0.009688 |
| Lid cavity E(1.1), face state + CCD `R_h`, N = 32 / 64 / 128 / 256 | 0.008318 / 0.009251 / 0.009628 / 0.009745 | 0.008364 / 0.009324 / 0.009665 / 0.009740 |
| max \|λh\| FCCD central flux; FUCCD4 `μ₄(π)` | 1.378; 4954.5 | 1.608; 6936.3 (CFL ≤ 0.19 under EXT2) |

Every "4th order" elsewhere in this README that comes from a Taylor–Green
vortex with `kx = ky` was measured with the plain `D_f`, on a flow that hides
its 2nd-order term. With the 4th-order `D_f` those `a = b` values do not
change (the `a = b` row above).

### 6th-order face value and gradient (`FCCDFaceComplex(order=6)`)

The CCD solve already gives each cell its jet `(f, f′, f″)`. FCCD used only
`f` and `f″`. The quintic Hermite interpolant of the two cell jets, evaluated
at the face, gives the face value `(f_L+f_R)/2 + 5h/32 (f′_L − f′_R) +
h²/64 (f″_L + f″_R)` and the face gradient `15/8 (f_R−f_L)/h −
7/16 (f′_L + f′_R) + h/32 (f″_R − f″_L)`. Both are 6th order on the same
two-cell stencil, with no second solve (`f′′′′ = D2 D2 f` was not used: its
second stage has no wall data). Wall faces keep the 4th-order closures. On a
mapped axis `J_face` comes from the same 6th-order gradient, which keeps
`d x/d x = 1`. Periodic, N = 128: gradient 3.8e-12, value 3.0e-13
(order 4: 2.1e-8, 7.6e-8). Taylor–Green, face state, N = 64: `a = b`
7.90e-9 (order 4: 8.01e-7); `a ≠ b` 4.49e-6 (8.49e-6), where the 4th-order
divergence is the floor. The default stays `order=4`. Design notes:
`docs/notes/`.

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

## Time step: IMEX Runge–Kutta (`IMEXRKNS`)

```python
ns = IMEXRKNS(grid, ..., ops=scheme.face, convection=scheme.convection, viscous=scheme.viscous)  # tableau="ars343"
```

EXT2 (`IMEXBDF2NS`) has no stable interval on the imaginary axis. The
non-dissipative FCCD/CCD convection is therefore stable only through
viscosity, and the high-order operators have larger spectra:
max |λh| = 1.000 (fd2 flux), 1.608 (FCCD flux; 1.378 with the plain face
difference, see *Face divergence* below) and 2.126 (CCD D1). The
explicit part of ARS(3,4,3) has the stability region of RK4 (imaginary
limit 2.83, real limit −2.79). Its implicit part is an L-stable SDIRK with
one γ on every stage. Each stage is projected with incremental pressure
correction. ARS(3,4,3) ends with one extra projection, because its explicit
weights are not its last row. Per step: 4 convection evaluations, 3 viscous
solves, 4 projections. The adaptive step is capped by the convection term's
`spectrum()` and the tableau limits.

Inviscid Taylor–Green with noise (N = 32, t = 1), largest CFL tried that stays stable:

| Convection | Basilisk BCG | `IMEXBDF2NS` | `IMEXRKNS` (ARS(3,4,3)) |
| --- | --- | --- | --- |
| central, fd2 | ≥ 1.5 | 0.7 | ≥ 2.0 (error 2.6e-2) |
| central, FCCD | — | 0.3 (0.5: error 1.6e-2) | ≥ 2.0 (error 5.9e-4) |
| FUCCD4 | — | 0.3 | 0.7 |
| UCCD6 | — | 0.05 | 0.1 (real-axis bound of the explicit hyperviscosity) |

Viscous Taylor–Green (μ = 0.01, t = 0.5), largest dt meeting a target error
over N = 8…64 and CFL = 0.025…2:

| Scheme | 1e-2 | 1e-3 | 1e-4 | 1e-5 |
| --- | --- | --- | --- | --- |
| Basilisk BCG, fd2 | 6.3e-3 | 7.8e-4 | — | — |
| BDF2, central fd2 | 3.1e-2 | 1.6e-2 | — | — |
| BDF2, central FCCD + CCD viscosity | 6.2e-2 | 3.1e-2 | 2.2e-2 | 6.3e-3 |
| RK343, central FCCD + CCD viscosity | 1.2e-1 | 6.2e-2 | 3.1e-2 | 3.1e-2 |
| BDF2, UCCD6 | 1.3e-2 | 6.3e-3 | 3.1e-3 | 1.6e-3 |
| RK343, UCCD6 | 1.3e-2 | 1.3e-2 | 6.3e-3 | 1.6e-3 |

At the same dt RK343 is more accurate (N = 64, CFL 1: 2.6e-6 against
6.2e-5), but a step costs about 4.5× more. With viscosity the wall time to a
given error is about the same: N = 64, 5e-6 takes 3 s (RK, CFL 2) and 4 s
(BDF2, CFL 0.3). RK pays off where stability limits the step, i.e. at low
viscosity. Both steppers keep the O(dt h^p) error of the approximate
projection: the RK error grows linearly with dt (Stokes: 3rd order).

### Face-native state (`face_native=True`)

```python
ns = IMEXRKNS(grid, ..., face_native=True)   # or IMEXBDF2NS(..., face_native=True)
```

The projected face velocity is the state. Cells are its reconstruction
`R_h` (`face_to_cell`: 4th order with FCCD) and are used only to evaluate the
terms (TwoPhaseFlow paper §11b2: `D_f f = 0`, `u = R_h f`). The predictor is
solved on cells, and only its increment is interpolated to the faces. The
non-invertible `R_h`/`P_f` round trip therefore never touches the state.
This removes the `O(dt h^p)` error of the approximate projection.

| Viscous Taylor–Green, IMEX-RK + FCCD, N = 16 / 32 / 64 | cells | faces |
| --- | --- | --- |
| cell state, CFL 0.1 | 2.6e-4 / 8.5e-6 / 2.7e-7 (5.0) | 3.7e-4 / 2.0e-5 / 1.1e-6 (4.2) |
| cell state, CFL 1.0 | 2.7e-3 / 8.6e-5 / 2.7e-6 (5.0) | 1.5e-3 / 5.5e-5 / 2.2e-6 (4.7) |
| face state, CFL 0.1 and 1.0 (identical) | 5.4e-4 / 3.5e-5 / 2.2e-6 (4.0) | 1.1e-4 / 7.0e-6 / 4.5e-7 (4.0) |

* Inviscid, noisy Taylor–Green, N = 32, t = 1, IMEX-RK + FCCD: the error is
  5.5e-5 for every CFL from 0.1 to 2.0, against 3.1e-5 … 5.9e-4 with the
  cell state. CFL 2.5 fails (measured again with the 4th-order `D_f`: 5.5e-5
  up to CFL 2.0; with the plain face difference CFL 3 failed).
* IMEX-BDF2 + fd2: self-convergence in dt goes from 1st order back to 2nd.
* Viscous Taylor–Green, N = 64, t = 0.5, wall time on the GPU:
  - IMEX-BDF2, cell state, CFL 0.2: 2.1e-6 in 160 steps, 6.9 s;
  - IMEX-RK, face state, CFL 2: 1.5e-6 in 16 steps, 3.3 s.
* What it costs: the cell values carry the `R_h` error. With FCCD's 4th-order
  `face_to_cell` the cell state is more accurate on cells at a small step
  (5th against 4th order).
* `reconstruction="ccd"` is a CCD-type `R_h`: no wider stencil, derivatives
  instead. The faces along an axis form a nodal line with the walls on its end
  nodes. A CCD solve on that line gives the face jet `(f, f′, f″)`. Its
  interior rows are Chu & Fan's; its wall rows stay on nodes 0–2, the interior
  stencil width (`tools/derive_face_ccd.py`): `f″` is 3rd order there and,
  coupled through an implicit `f″₀`, `f′` 4th order. The cell value is the
  quintic Hermite midpoint of its two faces,
  `(f_L + f_R)/2 + 5h/32 (f′_L − f′_R) + h²/64 (f″_L + f″_R)`. That is 6th order
  inside and 5th at walls. On its own at N = 128 it is 3.0e-13 periodic
  (6-point Lagrange stencil, `reconstruction=6`: 6.8e-11) and 2.5e-10 with
  walls.
* Wall rows through node 3 would make `R_h` 6th order at walls too, but they
  changed no Navier–Stokes result. Taylor–Green in a free-slip box, face
  state, N = 16 / 32 / 64, cell error: 1.8e-5 / 1.1e-6 / 6.8e-8 (wide) and
  1.7e-5 / 1.1e-6 / 6.8e-8 (compact); FCCD's `face_to_cell` gives 1.9e-7 at
  N = 64 (all with the earlier wide cell closures; with the compact ones of
  *Wall closures* below the compact `R_h` gives 1.6e-5 / 1.0e-6 / 6.5e-8). Wider one-sided rows have larger coefficients (largest row sum 4.6
  against 3.3), so they were dropped: wall rows keep the interior width and
  give up an order.
* In the face-state (periodic) Taylor–Green (N = 64, CFL 1) both `R_h` take
  the cells from 2.2e-6 to 8.0e-7, the level of the faces. The floor is now
  the 4th-order face operators (FCCD). The faces themselves go from 4.5e-7 to
  8.0e-7, because the cell values feed back into the terms; the largest error
  still falls by 2.75×.
* Lid cavity Re = 1000, N = 64, CFL 1: E(1.1) = 0.009257 (IMEX-RK, face
  state); every variant lies within 0.00913–0.00926.

### Implicit UCCD6 hyperviscosity (`implicit=True`)

```python
scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                       "terms": {"convection": "uccd6", "implicit": True, "viscosity": "ccd"}})
```

The explicit hyperviscosity caps UCCD6 at the real-axis bound,
`σ ω₂(π)⁸ CFL ≤ 4/3` (EXT2) or `≤ 2.79` (RK). The paper also says UCCD6 needs
an implicit treatment (§4.5), but integrates it explicitly in the NS.

With `implicit=True` the term `H u = Σ_k c_k D2_k⁴ u` is solved with the
viscous step, `(I − τV + τH)u = r`. The coefficients `c_k = σ max|u_k| h⁷`
are frozen at `uⁿ`. Every 1-D factor has a real spectrum; with walls one
eigenvalue is slightly negative (−1.66/h⁸ Dirichlet, see *Wall closures*),
a growth rate 1.7σ|u|/h, so the implicit term is stable up to that. The viscous
defect correction applies `H` exactly in its residual. Its preconditioner is
the viscous multigrid followed by `(I + τH)⁻¹` by fast diagonalization
(per-axis eigenvectors; condition ≈ 5e5 with walls, hence only as a
preconditioner).

| Case (N, t) | explicit H | implicit H, cell state | implicit H, face state |
| --- | --- | --- | --- |
| Inviscid noisy Taylor–Green (32, 1), largest stable CFL tried | 0.1 | 1.5 (2.0: 8e-3) | 2.0 (5.4e-5 at every CFL) |
| Viscous Taylor–Green (64, 0.5): error / steps / GPU wall time | CFL 0.1: 3.2e-7 / 320 / 48 s | CFL 1: 3.2e-6 / 32 / 7.2 s | CFL 2: 1.5e-6 / 16 / 4.0 s |
| Lid cavity Re = 1000 (64, 1.1): E / steps / time | CFL 0.07: 0.009221 / 1006 / 130 s | CFL 1: 0.009249 / 71 / 12 s | CFL 2: 0.009246 / 36 / 7 s |

At the same step (CFL 0.1) the implicit and explicit results coincide
(3.21e-7). A viscous solve then takes 1, 5 or 8 corrections at CFL 0.1, 1
and 2. IMEX-BDF2 with the implicit hyperviscosity stays at CFL 0.3, which is
EXT2's limit for the CCD advective part.

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
| Explicit (EXT2/BDF2) stability bound, σ = 1e-3 | — | CFL ≤ 0.157 per axis (`σ ω₂(π)⁸ CFL ≤ 4/3`, sharp: 0.9× stable, 1.1× blows up) | CFL ≤ 0.19 (`μ₄(π) = 6936.3`, matches the operator's eigenvalues; 0.27 and 4954.5 with the plain face difference) |
| Inviscid Taylor–Green, IMEX-BDF2, FCCD ops, N = 64 | 1.90e-7 (5th order) | 3.55e-7 (5th) | 1.90e-7 (5th) |
| Lid cavity Re = 1000, N = 64, t = 1.1 energy vs lid.ref | 0.21 % | 0.92 % (543 steps) | 0.25 % (315 steps) |
| Inviscid double shear layer, N = 64, t ≤ 1.5 | energy +13 % (t = 1.0), max\|u\| 3.95 | energy decreases, max\|u\| 1.65 | energy decreases, max\|u\| 1.63 |

The point-value order of the conservative forms is that of the face difference
`D_f F = (F_{i+1/2} − F_{i−1/2})/h` (2nd order), the same structure as
TwoPhaseFlow's `face_divergence`; the paper's 4th-order claim is for the face
values. The adaptive step of `IMEXBDF2NS` is capped by the term's
`max_stable_dt`; a fixed step above it warns.

With viscosity the Taylor–Green error is limited by the fd2 implicit viscosity
(Basilisk `viscosity()`), whatever the convection and the face operators:
μ = 0.01, N = 64, |u − exact| = 8.1e-5 (fd2 ops) and 6.45e-5 (FCCD ops, all three
convection terms), 2nd order; without viscosity FCCD gives 1.9e-7 against 3.4e-5
for fd2. `CCDViscosity` (next section) removes that limit.

## Viscosity: CCD full stress by defect correction

```python
scheme = build_scheme({"operators": {"face": "fccd", "cell": "ccd"},
                       "terms": {"convection": "uccd6", "viscosity": "ccd"}})
ns = IMEXBDF2NS(grid, ..., ops=scheme.face, convection=scheme.convection, viscous=scheme.viscous)
```

`CCDViscosity` (TwoPhaseFlow paper §6.3 `06d_viscous_3layer.tex`, §7.4
`07b_defect_viscous.tex`) solves `A_H u = u*` with `A_H = I − dt/ρ L_H`,
`L_H u = ∇·(μ(∇u + ∇uᵀ))` with CCD, correcting with Basilisk's fd2 viscosity
multigrid (`A_L`) under GCR; the stopping test is on the `A_H` residual.

`L_H` comes in two forms. `layered` is the paper's layers A/B/C literally
(CCD velocity gradients, cell stress, CCD divergence of the stress).
`split` (the default) takes the same-axis term as
`∂_b(μ ∂_b u) = μ D₂u + (D₁μ)(D₁u)` and keeps A/B/C for the cross terms.
Both are the full stress, valid for variable μ. The CCD first derivative
has no grid-scale mode (its symbol is 0 at θ = π), so `D₁D₁` leaves that mode
untouched while `A_L` damps it most, and the layered correction stalls:

| Corrections to 1e-10, N = 64, walls, white-noise rhs | κ = νdt/h² = 0.1 | 1 | 10 | 100 | 1000 |
| --- | --- | --- | --- | --- | --- |
| `layered` | 20 | 65 | 213 | > 300 | > 300 |
| `split` | 13 | 18 | 20 | 22 | 23 |
| `split`, μ ∈ [1, 10] variable, N = 32 / 64 / 128 | 18 / 18 / 18 | — | 22 / 22 / 22 | — | 26 / 26 / 25 |

| Check | fd2 (`basilisk`) | CCD (`ccd`, split) |
| --- | --- | --- |
| `L_H u` vs exact, periodic / walls (μ constant or variable) | 2nd | 6.0 / 3.1–3.3 at the wall cells (layered: 6.0 / 2.8–3.0, 13× larger wall error) |
| Taylor–Green μ = 0.01, IMEX-BDF2, FCCD ops, N = 16 / 32 / 64 | 1.03e-3 / 2.55e-4 / 6.45e-5 (2nd) | 1.16e-4 / 3.30e-6 / 1.70e-7 (5.1, 4.3) |
| Corrections per viscous solve in that run | — | 1–2 |
| Lid cavity Re = 1000, E(t = 1.1), N = 64 (fine-grid value ≈ 0.00969) | 0.008872 | 0.009134 |
| Stokes, self-convergence in dt | 2nd | 2nd |

Lid-cavity energies at N = 32 / 64 / 128 / 256 (IMEX-BDF2, central convection,
dt = 0.25/N): fd2 ops + fd2 viscosity 0.007304 / 0.008872 / 0.009486 / 0.009694;
FCCD ops + CCD viscosity 0.008250 / 0.009134 / 0.009614 / 0.009676. The
corner singularity of the lid limits both to about order 1.5; `lid.ref`
(Basilisk, N = 64: 0.008904) is itself under-resolved at t = 1.1.

## Non-uniform (mapped) grids — step 4f-1: operators

```python
from basilisk_cupy.grid.mapping import TanhMapping, MonitorMapping
grid = CartesianGrid((64, 64), mapping=(TanhMapping(2.0), MonitorMapping(lambda x: 1 + 3 * np.exp(-((x - 0.3) / 0.08) ** 2))))
```

An axis is the image `x(ξ)` of the uniform computational axis (TwoPhaseFlow
paper §10). CCD and FCCD work in ξ and convert with `J = dξ/dx`:
`f_x = J f_ξ`, `f_xx = J² f_ξξ + J J_ξ f_ξ` (CCD), `G_x = J_face G_ξ` (FCCD).
The metric comes from the CCD/FCCD derivatives of the coordinates
themselves (Dirichlet closure with the wall coordinates), so that `d x/d x = 1`
holds to round-off. Neumann data are converted with the exact `dx/dξ` of the
map at the wall. FCCD faces stay centred in ξ, so the paper's `O(H³)`
off-centre faces (§10c) do not occur. fd2 is finite volume: the gradient
divides by the physical centre distance and the divergence by the cell width.
Ghost centres are mirrored about walls. `MonitorMapping` integrates the
monitor with the Boole rule and inverts it by Hermite interpolation; the
paper's trapezoidal construction would cap the map at O(Δξ²).

| Check (N = 16…128) | tanh × tanh, Dirichlet | tanh × tanh, Neumann | sine × sine, periodic |
| --- | --- | --- | --- |
| CCD `f′` / `f″` (walls: max over the wall cells) | 4.0–4.4 / 2.8–3.6 | 4.0–4.4 / 2.9–3.1 | 6.1 / 6.0 |
| FCCD face gradient / value / face→cell | 4.2–4.6 / 4.0 / 4.0 | 3.9–4.0 / 4.0 / 4.0 | 4.0 / 4.0 / 4.0 |
| fd2 face gradient / divergence | 2.0 / 2.0 | 2.0 / 2.0 | 2.0 / 2.0 |
| GCL `d x/d x − 1` (CCD / FCCD / fd2) | 9e-14 / 1e-14 / 0 | | |

CCD full stress (`CCDStress`, split) on tanh × monitor: 4.1–4.3 (wall cells).
Boundary layer `exp(−x/0.02)` at N = 32: the CCD derivative error drops from
1.4e-2 (uniform) to 3.3e-5 (tanh, β = 3, first cell 0.001); the earlier wide
wall closures gave 5.6e-3 and 8.6e-7 (see *Wall closures* below).

### Step 4f-2: solvers

* Multigrid on mapped grids: every level is the same map. The residual
  restriction is volume weighted (`r_c = Σ V r / V_c`). The relaxation is
  alternating zebra line Gauss–Seidel along the wall axes, one batched
  Thomas solve per colour. Basilisk's point red/black smoother stalls on
  stretched (anisotropic) cells: 52 cycles on tanh(β = 2) × tanh(β = 2),
  64², against 11 on a uniform grid. Lines bring it back to 10, also at
  β = 3. Uniform grids keep Basilisk's point smoother, its GPU kernels and
  CUDA graphs.
* FCCD Neumann wall faces return the datum exactly. Without that the
  4th-order face metric breaks the compatibility of pure-Neumann problems,
  and the defect correction stalled at 5e-3.
* Viscosity on mapped grids: `CCDViscosity` and the new `FD2Viscosity`
  (Basilisk's stencil structure, i.e. compact same-axis face fluxes and
  centred cross terms through the velocity ghosts, by defect correction).
  On a mapped grid `A_L` is the componentwise scalar fd2 Helmholtz.
  `BasiliskViscosity` stays uniform-only.

| Check (N = 16 / 32 / 64 / 128) | fd2 multigrid | FCCD residual + fd2 multigrid |
| --- | --- | --- |
| Poisson, Dirichlet, tanh × monitor | 2.0 order, 10–13 cycles | 3.6–6.8 (6.5e-3 / 5.6e-5 / 4.5e-6 at N = 16 / 32 / 64), 16–17 |
| Poisson, Neumann, tanh × tanh | 2.0, 13–14 | 3.9–4.0, 10–11 |
| Poisson, periodic, sine × sine | 2.0, 14–19 | 3.9–4.0, 10–12 |

Viscous defect correction on tanh(β = 2)², 32², white-noise rhs: 24–37
(CCD) and 15–21 (fd2) corrections for κ = νdt/h² = 0.1…1000. Decaying
channel mode u = sin(πy) on tanh(β = 2) in y, ν = 1, 20 backward-Euler
viscous steps to t = 0.1, against the time-discrete factor (1 + νπ²dt)^−20
(spatial error only), N = 8/16/32: FD2Viscosity 1.2e-2 / 3.7e-3 / 9.6e-4,
CCDViscosity 6.6e-4 / 1.3e-5 / 2.6e-7.

BCG, UCCD6/FUCCD and the CFL step still refuse mapped grids (step 4f-3).
On the GPU the line relaxation's sequential Thomas sweep is slow (one
launch per row); a cyclic-reduction solver is planned with 4f-3.

## Two-phase flow (M4) — step M4a: VOF advection

```python
from basilisk_cupy.interface.fractions import fraction
from basilisk_cupy.interface.vof import VOFInterface, VOFTracer, vof_advection
f = CellField.from_interior(grid, fraction(grid, lambda x, y: 0.04 - (x * x + y * y)), BoundaryConditions())
vof_advection([VOFInterface(f, [VOFTracer(cf)])], uf, dt, i)
```

Port of Basilisk's `geometry.h`, `myc2d.h`, `fractions.h` and `vof.h` (2-D, uniform grid):
PLIC with the Mixed Youngs-Centred normal, direction-split geometric fluxes, the
Weymouth & Yue `cc = (c > 1/2)` term, sweeps alternating with the iteration index, and
VOF concentrations. The operations follow the C code's order. `cases/reversed_vof.py`
(`src/test/reversed.c`, constant grid) reproduces `reversed.cref` to the printed digits:

| N | volume sum (t = 0 and T) | error avg / rms / max at t = T |
| --- | --- | --- |
| 32 | 0.124638306824 | 0.0605584 / 0.21841 / 1 |
| 64 | 0.125405298502 | 0.00992636 / 0.0733979 / 1 |
| 128 | 0.125600659213 | 0.00143235 / 0.017043 / 0.474599 |

The volume fraction stays in [−1.4e-16, 1 + 4e-16] and the VOF concentration within
8e-13 of 1. N = 128 (3723 steps) takes 28 s with cupy. With FCCD operators the
advecting velocity must be `ops.wall_free_flux(uf, periodic)` (see *Face divergence*).
Plan and notes: `docs/notes/12_two_phase.md`.

### Step M4b: two-phase properties, height functions, reduced gravity

```python
tp = TwoPhase(ns, f, rho1=1.0, rho2=0.1, mu1=mu, mu2=mu, vof_first=True)   # hooks into CenteredNS
tp.add_reduced_gravity(G=(0.0, 50.0))
```

Ports of `two-phase.h` (`alpha = 1/rho(f_face)`, `rho(f)`, `mu(f_face)`), `heights.h`
(height functions, multigrid version; the column propagation runs as one GPU kernel,
bitwise equal to the array path), `position()` from `curvature.h`, `iforce.h` and
`reduced.h`. `cases/gravity_wave.py` (`src/test/gravity.c`), relative RMS error of the
amplitude against Prosperetti's solution:

| points per wavelength | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| lexicographic relaxation (as Basilisk's CPU build) | 0.126711 | 0.031758 | 0.00807759 | 0.00455815 |
| `gravity.ref` (Basilisk, CPU) | 0.126713 | 0.031758 | 0.00807759 | 0.00455815 |
| red/black relaxation (default) | 0.126717 | 0.0317753 | 0.0080898 | 0.00455964 |
| Basilisk built with `-DGAUSS_SEIDEL` (red/black) | 0.126716 | 0.0317752 | 0.00808982 | 0.00455964 |

Matching Basilisk needed three corrections of the port and one constant (found by building
Basilisk from the mirror and comparing the fields after the first step, which now agree to
1e-15 for both relaxations; `tests/data/`): Basilisk's `project()` solves to
`TOLERANCE/dt²` (`project(..., basilisk_tolerance=True)`, used by `CenteredNS`); its red/black
sweep starts with the odd cells (`i + j`); its CPU build relaxes lexicographically
(`linear.lexicographic.set_relaxation("lexicographic")`, fronts `2i + j = const`, exact for
the 9-point viscous stencil too); and its `pi` is 3.14159265358979
(`basilisk_cupy.constants.PI`), which decides whether a cell cut near a vertex gets a
volume fraction of exactly 1. A flat interface under reduced gravity stays at rest to 1.4e-17.
N = 128 (738 steps) takes 37 s with cupy (red/black).

### Step M4c: surface tension

```python
tp = TwoPhase(ns, f, rho1=1.0, rho2=1.0, mu1=mu, mu2=mu, vof_first=True)
tp.add_surface_tension(1.0)          # phi = sigma kappa, dt <= sqrt(rho_m Delta^3 / (pi sigma))
```

Ports of `curvature()` (`curvature.h`: height-function curvature, then a parabola fitted to
mixed heights, the 3x3 average, a parabola fitted to centroids; `parabola.h`) and
`tension.h`. The curvature agrees bitwise with Basilisk's on five shapes that use all four
methods (`tests/data`, printed by Basilisk built from the mirror). With accurate volume
fractions the largest relative error on a circle (R = 0.2) is 3.7e-2 / 4.9e-3 / 1.2e-3 at
N = 32 / 64 / 128, the level of Basilisk's `curvature.ref`; `fraction()` alone is only
O(h²) accurate and leaves an O(1) curvature error. `cases/capillary_wave.py`
(`src/test/capwave.c`), relative RMS error against Prosperetti's solution:

| points per wavelength | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| lexicographic relaxation | 0.138643 | 0.0242378 | — | — |
| `capwave.ref` (Basilisk, CPU) | 0.13865 | 0.0242375 | 0.00639279 | 0.00226914 |
| red/black relaxation (default) | 0.138631 | 0.0242639 | 0.00638451 | 0.00232291 |
| Basilisk `-DGAUSS_SEIDEL` | 0.138638 | 0.0242618 | 0.0063841 | 0.00232294 |

The first step agrees with Basilisk to 1e-15 in every field; the differences above come from
round-off growing over the run (1e-14 after 50 steps). Basilisk's `init` calls `dtnext()`
before any event time is known: the first step keeps that `dt` and the next one is shortened
to reach the event (`CenteredNS.start()` now does the same). A static drop (R = 0.25,
sigma = 1, equal densities, N = 32) has a pressure jump of 4.038 (sigma/R = 4) and parasitic
velocities of 7.4e-3 (7.4e-4 of sigma/mu) after 10 steps.

### TwoPhaseFlow's conservative level set (CLS) against VOF

```python
from basilisk_cupy.interface.cls import CLSInterface
cls = CLSInterface.from_level_set(grid, distance, bc)   # psi = H_eps(distance), eps = 1.5 Delta
cls.advect(uf, dt)          # TVD-RK3, flux (P_f psi) u_f with FCCD face values, clipping, mass closure
cls.reinitialize(4)         # compression--diffusion (provisional; Ridge--Eikonal next)
```

On Basilisk's reversed vortex (T = 15) the sharp indicator `psi > 1/2` has an L1 error of
0.0707 / 0.0306 / 0.0139 at N = 32 / 64 / 128 (re-initialising every 10 steps: 0.0707 /
0.0306 / 0.0080), against 0.0606 / 0.0099 / 0.0014 for VOF on the same velocity. A periodic
translation gives a psi L1 error of 6.0e-3 / 5.4e-3 / 4.3e-3 (VOF 1.5e-3 / 3.4e-4 / 8.7e-5).
Mass is kept to 1e-13.

`cls.reinitialize()` defaults to the paper's standard re-initialisation on uniform grids, the
Ridge--Eikonal distance reconstruction in its xi-SDF form (zero crossings by linear
interpolation, distance to the nearest crossing, `psi = H_eps`, mass closed by a uniform shift
of `phi`). Measured: on an exact profile it moves psi by 4.5e-2 at every resolution (the
distance to crossing points along the axes overestimates an oblique interface's distance by up
to sqrt(2)); the distance to the polyline through the crossings (`distance="segments"`) moves
it by 3.6e-3 / 1.8e-3 / 9.0e-4 (N = 32 / 64 / 128). In passive transport both make the
results worse than no re-initialisation (reversed vortex every 10 steps, N = 64: 0.068 and
0.079 against 0.031). Notes: `docs/notes/13_cls.md`.

`cls.curvature()` is the paper's curvature from the derivatives of `psi` (CCD `psi_x`, `psi_xx`,
`psi_xy`, zero outside `0.01 < psi < 0.99`). Measured on a circle (max over `|phi| < eps` of
`|kappa - 1/r| R`): on the exact profile 6.3e-4 / 1.2e-3 / 2.4e-3 / 4.8e-3 (N = 64 / 128 / 256 /
512; the profile is 3 cells wide at every N, so the error grows like 1/h), 4.9e-7 from
`phi = H_eps^{-1}(psi)` at N = 128; after a periodic translation 0.39 / 0.61 / 0.81 (N = 32 / 64 /
128) without re-initialisation, 0.13 / 0.037 / 0.017 with compression--diffusion every 10 steps.
Notes: `docs/notes/14_psi_curvature.md`.

`TwoPhaseCLS(ns, cls, rho1=..., rho2=..., mu1=..., mu2=...)` couples the CLS interface to
`CenteredNS` exactly as `TwoPhase` couples VOF (properties from `psi` with arithmetic face
averages, CSF surface tension `sigma kappa grad psi` in `iforce.h`'s balanced-force face form).
Static drop (R = 0.25, N = 64 / 128 / 256): pressure jump 4.0230 / 3.9410 / 3.9228 with the
paper's band `psi_min = 0.01` (the gradient outside the band, 2%, carries no curvature) and
4.1299 / 4.0298 / 4.0067 with `psi_min = 1e-4`; parasitic velocities 3.2e-4 / 1.7e-4 / 1.3e-4
(VOF 4.2e-3 / 2.0e-3 at N = 64 / 128). Capillary wave (`cases/capillary_wave.py`,
`interface="cls"`): 0.355 / 0.315 / 0.236 at 8 / 16 / 32 points per wavelength (VOF 0.139 /
0.024 / 0.0064); the wave is slow by 39% / 17% / 7.1% in the half period, set by `k eps` alone
(the CSF force spread over the interface width). Notes: `docs/notes/15_cls_ns.md`.

`TwoPhaseCLS(..., pressure_jump=True)` is the paper's pressure-jump closure in finite-volume form
(ghost-fluid method): the jump `sigma kappa` enters the face gradient `G p - B(j)` at the
`psi = 1/2` crossings, and the face coefficients are those of phase-wise constant densities
(GFM `1/(theta rho_L + (1 - theta) rho_R)` on cut faces). Static drop (N = 32 / 64 / 128): jump
4.0091 / 4.0027 / 4.0010, parasitic velocity 3.3e-4 / 4.9e-5 / 8.6e-6, 1.1e-8 after 400 steps
(N = 64; VOF 6.1e-8). At density ratio 1000 tension's time-step bound is too large (it
diverges after 200 steps; half the step is stable). Capillary wave: 0.201 / 0.0933 / 0.0279 /
0.0130 at 8 / 16 / 32 / 64 points per wavelength (CSF 0.355 / 0.315 / 0.236; VOF 0.139 /
0.024 / 0.0064 / 0.0023). Notes: `docs/notes/16_split_ppe.md`.

The curvature error after transport comes from the transported interface, not from the curvature
formula: without re-initialisation the `psi = 1/2` crossings carry a grid-scale scatter of 2.2e-4
at every N. Compression--diffusion after every step makes the curvature converge (translation,
max error 0.202 / 0.0139 / 0.00305 at N = 32 / 64 / 128) but thins filaments (reversed vortex
0.171 / 0.137 / 0.0687). Height functions of `psi` or a smoothed copy do not help, and the
capillary wave with the GFM jump does not improve with an ideal curvature (0.193 / 0.107 /
0.0448). Notes: `docs/notes/17_curvature_transport.md`.

That interface error is the paper's plain face difference: `D_f` of the FCCD face fluxes is 2nd
order on point values, and with `eps = 1.5 Delta` its error `-(h^2/24) psi_xxx` is -0.93% of
`psi_x` at the crossing at every N, so the interface lags by about 0.9% of the distance
travelled. `CLSInterface(..., divergence="fccd")` subtracts `(h^2/24)(psi_xx u + 2 psi_x u_x)`
(face-line CCD derivatives; without `psi u_xxx`, so a projected velocity keeps `psi = 1`).
Translation, psi L1 error at N = 32 / 64 / 128: 6.0e-3 / 5.4e-3 / 4.3e-3 (paper), 1.15e-3 /
1.06e-3 / 9.8e-4 (corrected), 1.4e-4 / 1.1e-4 / 1.1e-4 (corrected, 6th-order face values).
Capillary wave with the GFM jump: 0.188 / 0.0639 / 0.0109 / 0.00209 at 8 / 16 / 32 / 64 points
per wavelength (VOF 0.139 / 0.0242 / 0.0064 / 0.0023). Notes: `docs/notes/18_cls_transport.md`.

At density ratios 100 and 1000 (equal viscosities, N = 64) the GFM drop diverged with tension.h's
time step: the explicit capillary loop sees CCD's curvature, whose answer to a grid-scale wiggle
is 48/5 per Delta^2 instead of 4 (2nd-order differences; the bound, also the paper's, has a
margin of about 1.6 for those and none for CCD). `TwoPhaseCLS(..., capillary_dt_factor=
CCD_CAPILLARY_DT_FACTOR)` (the default) multiplies the bound by sqrt(5/12); the parasitic velocity then decays
at every density ratio (3.9e-8 at t = 0.4, ratio 1000, N = 64). Notes: `docs/notes/19_capillary_dt.md`.

Checked by bisection on the static drop (`cases/capillary_dt_scan.py`, density ratio 100, mu = 0.1):
the stability limit, as a multiple of tension.h's bound, is 1.05-1.1 / 0.95-1.0 / 0.91-0.95 /
0.85-0.88 with the CCD curvature and 1.4-1.5 / 1.4-1.5 / 1.3-1.4 / 1.2-1.3 with 2nd-order
differences of `psi` (N = 32 / 64 / 128 / 256): their ratio, 1.45-1.49 for N >= 64, matches the
stiffness ratio sqrt(12/5) = 1.55, but the limits fall with N (and with viscosity and density
ratio), which `omega dt <= 2` does not explain; Basilisk's VOF stays at 1.2-1.4 for N = 64 and
256. sqrt(5/12) is below every limit measured (lowest 0.82-0.85). Notes:
`docs/notes/20_capillary_dt_check.md`.

The mode that grows is a grid-scale capillary wave (sign alternating cell by cell along the
interface, velocity confined to the two cells across it). The CLS transport (central FCCD) does
not damp it; making the transport more accurate makes it worse, and only freezing the interface
or compression--diffusion after every step (4 pseudo-steps) removes it: the limits are then >= 1.6
/ 1.6 / 1.6 / 1.4 (N = 32 / 64 / 128 / 256) and no longer fall with N, at the price of parasitic
velocities that no longer decay and of thinned filaments. Notes: `docs/notes/21_capillary_mode.md`.

### Static drop against Basilisk (figures)

`cases/static_drop_compare.py` runs the static drop (R = 0.25, sigma = 1, mu = 0.1, densities 1 / 0.1,
64^2) with VOF, CLS + CSF and CLS + GFM and plots density, velocity and pressure at t = 0.2 next
to Basilisk's own fields (`tools/basilisk/static_drop.c`): `docs/figures/static_drop_compare.png`
and `docs/figures/static_drop_compare_lines.png`. The VOF port differs from Basilisk by 2.5e-10 in
velocity (parasitic velocity 5.3e-7) and 5.1e-9 in pressure; CLS + CSF keeps parasitic velocities
of 4e-4, CLS + GFM decays like Basilisk (7.5e-7) with a pressure jump of 4.0030.
Notes: `docs/notes/22_static_drop_figures.md`.

`cls.reinitialize(n, method="acls2008" | "acls2017")`: ACLS re-initialisation with the normal of a
reconstructed distance (the polyline through the `psi = 1/2` crossings, in place of fast
marching). Chiodi & Desjardins' 2017 form differentiates `phi_map = eps ln(psi/(1 - psi))` and
balances a planar profile to round-off at any angle (the 2008 form leaves 8e-4 to 2e-3); on a
static circle of 20 cells, 1000 re-initialisations change psi by 1.1e-3 (2008: 3.2e-2). It needs a
quarter of the pseudo-time step, and in these tests it does not beat the FCCD compression--
diffusion: translation curvature 0.066 / 0.015 / 0.007 against 0.202 / 0.014 / 0.003 (N = 32 / 64
/ 128), reversed vortex every 10 steps 0.118 / 0.097 / 0.063 against 0.071 / 0.030 / 0.0083.
Notes: `docs/notes/24_acls.md`.

`cls.curvature(method="heights", heights_method="ccd" | "fd2")`: height functions of the `psi = 1/2`
crossings (cubic interpolation of `phi_map` along the columns), differentiated along the interface
by CCD or central differences. On an exact circle (N = 64 / 128 / 256) the rms error at the
crossings is 2.4e-3 / 1.7e-4 / 2.4e-5 with CCD, against 9.6e-4 / 3.5e-4 / 2.3e-4 for the curvature
of `psi`; column sums instead of crossings are off by 1-3% (the diffuse profile is skewed where the
interface is curved). In flows the gain is small: capillary wave 0.183 / 0.0496 / 0.0100 / 0.00478
(psi 0.188 / 0.0639 / 0.0109 / 0.00209), and CCD used up to the ends of a run (walls) diverges;
2nd-order heights let the static drop (density ratio 100) decay to 1e-11 with tension.h's step.
Notes: `docs/notes/25_height_curvature.md`.

`cls.reinitialize(n, method="acls-fccd")`: the 2008 ACLS flux `(eps grad psi . n - psi (1 - psi)) n` with
FCCD face operators (no logarithm, so CCD can be used). On a static circle of 20 cells, 1000
re-initialisations change psi by 7.2e-4 (compression--diffusion 1.1e-3, 2nd-order ACLS 2008 3.2e-2),
explicit pseudo-steps are stable up to twice the usual step; backward Euler by defect correction
(2nd-order `div(eps n_a^2 grad)`, normals frozen) converges only up to four times. With the GFM drop
(density ratio 100) the grid-scale capillary mode of docs/notes/21 is removed by compression--diffusion
(isotropic diffusion), not by ACLS (diffusion along `n` only). Figures: `docs/figures/reinit_*.png`.
Notes: `docs/notes/26_acls_fccd.md`.

`TwoPhaseCLS(..., implicit_tension=theta)`: semi-implicit surface tension (Hysing 2006, isotropic
surface Laplacian) on the transport velocity: after the projection each component of `uf` solves
`u/alpha - div(theta dt^2 sigma |grad psi| grad u) = u*/alpha` (conjugate gradients on the face grid)
and is projected again. The GFM drop at density ratio 100 is then stable at tension.h's step
(factor 1, where the explicit loop grows) for N = 64-256 with theta = 4 (theta = 1: N = 64 and 128);
adding the same term to the predictor's viscous solve, or smoothing `uf` before the projection, does
not work. The capillary wave loses some accuracy (16 points per wavelength: 0.0639 -> 0.0660 / 0.0718
for theta = 1 / 4). Figure: `docs/figures/implicit_tension.png`. Notes: `docs/notes/27_implicit_tension.md`.

Interface width: with `eps = c Delta`, c = 1.5 (the paper's) leaves only three cells for the transition,
so the compact operators stall. At c = 2 / 3 / 4 the ACLS flux on a planar profile drops from 4.1e-4 to
1.3e-4 / 1.8e-5 / 4.6e-6, the curvature of an exact circle (N = 128) from 1.2e-3 to 1.2e-4 / 6.2e-6 /
9.7e-7, the psi error after a translation (N = 128) from 1.1e-4 to 3.9e-5 / 1.0e-5 / 4.2e-6; but where
the drop is only a few widths across (N = 32) the curvature error rises to 0.3-0.7, the capillary wave
does not change (0.064 at 16 points per wavelength for every c) and neither does the grid-scale
capillary instability. Figure: `docs/figures/eps_width.png`. Notes: `docs/notes/28_eps_width.md`.

The capillary wave's remaining error (CLS + GFM 0.0639 / 0.0109 at 16 / 32 points per wavelength against
VOF's 0.0243 / 0.0064) is under-damping: decay per period 0.7247 / 0.6992 against Prosperetti's 0.692,
converging at 2nd order (CLS is better than VOF at 64 points, 0.0021 / 0.0023). The amplitude reading,
mass closure, clipping, time step, jump position and curvature (even an exact one) do not change it.
Inviscid, VOF's transport damps the wave by 2.9 % per period and CLS's by 0.6 %: the interfacial vorticity
layer (sqrt(2 nu / omega) = 0.057, 0.9 / 1.8 cells) is under-resolved, and VOF's numerical dissipation
happens to make up for it. Figure: `docs/figures/capwave_damping.png`. Notes: `docs/notes/29_capwave_damping.md`.

Confirmed by scaling the viscosity (x0.5 / 1 / 2 / 4) at 16 / 32 / 64 points per wavelength against the normal
mode of two identical viscous fluids (`cases/capwave_modes.py`, which matches Prosperetti at the original
viscosity): the CLS damping-rate error depends on delta / h alone (-22 % at 0.65, -5.5 % / -4.8 % at 1.30
from two different grids, within 1 % above 2.5), the period error on the points per wavelength alone.
Figure: `docs/figures/capwave_layer.png`. Notes: `docs/notes/30_vorticity_layer.md`.

`CenteredNS(..., viscous=CCDViscosity())` solves the implicit viscous step with the CCD full stress (defect
correction on Basilisk's viscous multigrid). On a Taylor-Green flow with kx != ky it stays 2nd order with
Basilisk's face operators (the projection and the face interpolation set the order) and becomes 4th order
with FCCD faces as well (8.5e-4 / 5.7e-5 / 3.6e-6 at N = 16 / 32 / 64). On the capillary wave (GFM, fd2
faces) it does not change the damping: FCCD faces would be needed, and they cannot cross the pressure jump
without the paper's jump-corrected gradient and HFE. Figure: `docs/figures/ccd_viscous.png`.
Notes: `docs/notes/31_ccd_viscous_centered.md`.

The paper's jump-corrected face gradient `G_Gamma(p; j) = G p - B(j)` in its smooth-pressure form:
`TwoPhaseCLS(..., jump_mode="smooth")` solves for `p~ = p - H j_ext` (`j_ext` the jump at the nearest
crossing, so `p~` and its normal derivative are continuous for equal densities) with the face acceleration
`alpha_f (B(j) - D_f(H j_ext))`, so that FCCD faces (`CenteredNS(ops=FCCDFaceComplex(), viscous=CCDViscosity())`,
CLS `divergence="full"`) never cross the jump. Static drop (N = 64): parasitic velocity 4.1e-5 against 4.0e-3
with FCCD across the jump (fd2: 4.4e-5). Capillary wave: 0.0419 / 0.0082 at 16 / 32 points per wavelength
(fd2 faces 0.0639 / 0.0109), mostly from the period; the damping error barely moves. Figure:
`docs/figures/fccd_projection.png`. Notes: `docs/notes/32_fccd_projection.md`.

A linear model of CenteredNS's capillary loop (`cases/coupled_stability.py`: the interface moves with the
previous velocity, the predictor damps the previous step's force with the viscosity and removes it, the new
force enters with the projection) gives `a <= min(2(1+b)/(2-b), (1+b)/(1-b))`, `a = (omega dt)^2`,
`b = 1/(1 + d dt)`: the limit falls from omega dt = 2 to 1 as the viscosity grows, as measured (static drop,
N = 64: 1.60-1.65 times tension.h's step at mu = 0, 1.00-1.05 at mu = 1). With the force inside the implicit
viscosity the model gives `a <= 4 + 2 d dt`. Viscous operators alone: projected white noise decays alike with
fd2 and the CCD split form (E/E0 1.2e-3 after 40 steps), grows with the layered D1 D1 form (7.8).
Figure: `docs/figures/coupled_stability.png`. Notes: `docs/notes/33_coupled_stability.md`.

`CenteredNS(..., coupled_iterations=K)` computes the new acceleration first and solves the viscosity on
`u + dt g`, `g` from the new acceleration and the latest pressure, then projects (K outer iterations).
As the model predicts, the viscosity then raises the stable step instead of lowering it (static drop,
equal densities, N = 64, multiples of tension.h's step: 1.44-1.66 / 1.66-1.88 / 2.09-2.31 / 2.53-2.75 at
mu = 0 / 0.01 / 0.1 / 1, against 1.60-1.65 / 1.40-1.45 / 1.15-1.20 / 1.00-1.05), and N = 128 does not lower
it; the accuracy is unchanged and the capillary wave runs at four times the step. Not at density ratio 100.
Figure: `docs/figures/coupled_viscous.png`. Notes: `docs/notes/34_coupled_viscous.md`.

`cases/step_jacobian.py` linearises one whole step (psi, u, uf, p, g; matrix-free central differences,
Arnoldi, the pressure gauge and the total psi deflated). With the solvers at 1e-14, the factor where the
leading |lambda| crosses 1 falls inside the measured stability bracket in all eight cases checked (static drop,
N = 32, density ratios 1 and 100, K = 0, 1, 2 and converged). At density ratio 100, K = 1 fails through a band of
oscillatory modes (|lambda| 1.19, arg 0.22 pi at 1.6 x, on the oblique parts of the interface), not the
capillary sawtooth (arg = pi): a lagging pressure leaves `(B - I) dt alpha grad dp` after the viscous solve, which
the projection does not remove, and it grows with the gas-side alpha (0.039 / 0.70 / 8.1 / 46.5 relative at
density ratios 1 / 10 / 100 / 1000; viscosity ratio alone: 0.0026). `coupled_tolerance=` stops the outer
iterations when `dt |g_{k+1} - g_k|` falls below it; converged, density ratio 100 at N = 64 is stable up to
1.8-2.0 x (K = 1: below 1.0) with 4-5 iterations per step. Figures: `docs/figures/step_jacobian.png`,
`docs/figures/step_jacobian_modes.png`. Notes: `docs/notes/35_step_jacobian.md`.

The outer iterations now report and control their convergence: `coupled_criterion="change"` (`dt |g_{k+1} - g_k|`)
or `"residual"` (the momentum residual, which is exactly `(dt^2/rho) L (g_{k+1} - g_k)` and is 4-28 times the
change at density ratios 100-1000 because of the stiff gas viscosity, while the velocity error stays about half
the change), `coupled_rtol=` (relative to the first iteration), `coupled_on_failure="warn"|"raise"|"ignore"` and
`coupled_failures`. The lagging-pressure residue `P (B - I) w` splits into `[P, B] w + (B - I) P w`: on the
unstable mode 96 % is the second term (the cell-centred projection does not remove a face gradient averaged to
cells), not the commutator. Converged, the static-drop limit stays at 1.7-1.9 x tension.h's step (ratios 100 and
1000) and 2.1-2.3 x (ratio 1) for N = 16-128, while one iteration (K = 1) falls with N (1.75 / 1.28 / 0.93 at
ratio 100). Anderson acceleration of the pressure iterates (`coupled_anderson=m`) cuts the iterations by 35-50 %;
extrapolating the pressure (`coupled_predictor="linear"`) helps only with `TwoPhaseCLS(jump_mode="smooth")`,
where the jump is rebuilt from the moved interface. For a fixed physical time (oscillating drop, N = 32, ratio
100) Basilisk's splitting at 0.95 x is still faster (9.3 s against 22-31 s) and more accurate. Figure:
`docs/figures/coupled_solver.png`. Notes: `docs/notes/36_coupled_solver.md`.

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

Why the layered CCD viscosity (`CCDViscosity(form="layered")`, the paper's literal D1 o D1) increased the energy
of projected white noise 7.8 times in 40 steps (`cases/layered_energy.py`, operators assembled as matrices):
with walls, layer A closes the velocity derivative with the Dirichlet condition and layer C closes the stress
derivative by extrapolation, so `L` is not symmetric. Its symmetric part has a positive eigenvalue (+12.7/h^2 at
N = 16), `||(I - kappa L)^{-1}||_2 = 8.7`, and the zero eigenvalue of the grid-scale modes (where the CCD first
derivative vanishes) is defective (nullity 4 for `L`, 5 for `L^2`), so the energy grows like a power of the step
count although `rho(A) - 1 = 8e-4`. With periodic boundaries the same composition is symmetric and nothing
grows; the defect correction converges and the projection only speeds the growth up (`rho(PA) = 1.048`). The
default split form has a negative symmetric part with walls and decays monotonically. Figure:
`docs/figures/layered_energy.png`. Notes: `docs/notes/37_layered_energy.md`.

Why the CLS capillary wave is under-damped on coarse grids (`cases/capwave_normal_mode.py`, the wave started on
the exact normal mode of two identical viscous fluids, so the amplitude is one damped cosine): the damping
error depends on delta / h alone (-27.8 / -16.4 / -9.1 / -3.8 / -2.0 % at 0.65 / 0.92 / 1.30 / 1.83 / 2.59) and
not on the time step, the CLS width, the order of the viscous and face operators (fd2, CCD, FCCD) or the jump
form; smearing the force (CSF) makes it much worse. The damping ratio equals, within about 1 %, the enstrophy of
the exact mode sampled at the cell centres over its true value: the vorticity has a cusp at the interface
(`[d_nn u_t] = d_t [p] / mu`) that cell values cannot carry, so that part of the dissipation is missing.
`1 / sqrt(1 + c (h / delta)^2)` with c = 0.376 fits all points. Figure: `docs/figures/capwave_layer_diag.png`.
Notes: `docs/notes/38_capwave_layer_diag.md`.

CLS and the sharp GFM jump on mapped grids (`CartesianGrid(..., mapping=(None, MonitorMapping(...)))`): the CLS
face difference divides by the physical cell widths, the mass uses the cell volumes, the GFM jump divides by
the centre distance (as the fd2 face gradient) and tension's step bound uses the smallest physical width
(re-initialisation, the smooth jump form, the `fccd`/`full` transport corrections, implicit tension and surface
viscosity still refuse mapped grids; momentum advection too, so the comparison runs Stokes). An identity map
reproduces the uniform grid to 1e-8. With 32 x 32 cells and the y axis refined alpha = 2 / 4 / 8 times about the
interface, the capillary-wave damping error drops from -20.6 % to -9.5 / -4.5 / -2.2 % (`cases/mapped_capwave_figures.py`),
but the line relaxation of the mapped multigrid makes it slower on the CPU than a uniform grid of the same
accuracy (43-51 s against 16 s for uniform 64^2 at -6.1 % and 58 s for 128^2 at -0.4 %). The static drop needs
eps from the coarsest cells the interface crosses. Figure: `docs/figures/mapped_cls.png`. Notes:
`docs/notes/39_mapped_cls.md`.

A jump-corrected viscous operator was tried for the vortex layer (`cases/jump_viscous.py`, IIM form): the kink
`[d_nn u_t] = d_t j / mu` is written as a divergence-free `J = H perp grad(-A chi d^3 / 6)` and
`mu (H L(J+) - L(H J+))` is added to the implicit viscous step. As an operator it removes most of the kink error
(fd2, tangential component near the interface: 0.31 -> 0.07), but on a frozen interface it helps only at
delta / h = 1.83 (6.5e-4 -> 3.5e-4) and hurts at 0.92, and with the moving interface (capillary wave) it is
unstable at N = 32 and 64 even at dt / 16: the tangential derivative of the curvature feeds grid-scale noise
back into the capillary loop. Not adopted. By-product: the height-function curvature lowers the damping error
much more (N = 64: -3.8 % -> -0.6 %; N = 32: -16.4 % -> -11.4 %), so the delta / h scaling of note 38 holds for
the psi curvature only. Figure: `docs/figures/jump_correction.png`. Notes: `docs/notes/40_jump_correction.md`.

Discrete dispersion relation of the capillary wave (`cases/capwave_jacobian.py`): one step's Jacobian about the
flat interface at rest, restricted to the fields cos kx / sin kx times a y profile (closed to 2e-5), gives the
scheme's eigenvalue directly (N = 32: -16.4 % damping, -3.6 % frequency, as the time integration). Swapping
one part at a time for an exact one: the exact linear curvature gives the same deficit (-16.6 %), and so do
dt / 4, the CCD viscosity and FCCD faces; moving the interface with the face velocity at the interface instead
of the CLS transport removes it (+3.8 %). The CLS transport moves the psi = 1/2 crossing with
`sum w_f v(y_f)`, weights 0.57 at the interface face and 0.23 at +-h (second moment 0.33 h^2), and v has a peak
of width delta at the interface, so the kinematic error is ~ (h / delta)^2. The normal-mode dispersion relation
with this discrete kinematic condition reproduces the deficit (-17.0 % against -16.6 % at N = 32). This corrects
notes 38 and 40: the deficit is kinematic, not the viscous resolution of the layer, and the height-function
curvature only cancels it. Figure: `docs/figures/capwave_theory.png`. Notes: `docs/notes/41_capwave_theory.md`.

A spectrally split CCD damping `gamma Q^T M Q` (`Q = I - (I + alpha L_h^m)^{-1}` from the CCD Laplacian; proposed
together with a high-wavenumber defect correction) was evaluated (`cases/split_hyperviscosity.py`): it is
energy-dissipative even with the non-symmetric wall closures, but it lowers the capillary wave's damping
(-16.4 % -> -18.9 % at gamma dt = 100, N = 32), because it trims the vortex layer that carries the dissipation,
and it does not stabilise the density-ratio-100 mode (|lambda| 1.189 -> 1.157; 7.3 % of that mode's energy lies
in Q's band, white noise 10.4 %). This damping term is not adopted; the defect correction and the structure-preserving parts of the proposal were not evaluated here. Figure: `docs/figures/split_hyperviscosity.png`.
Notes: `docs/notes/42_split_residual_eval.md`.

Surface tension as the adjoint of the CLS transport (`cases/adjoint_tension.py`): with the discrete area
`A = sum |G psi| V` and `mu = sigma dA/dpsi / V`, the face force `F = -psi_f G_f mu` does exactly the work that
the same face velocity removes from the surface energy (to 1e-14). It is far less accurate than the sharp GFM
jump (capillary wave, N = 32: -63.9 % damping, -32.8 % frequency against -20.7 % / -5.0 %; the error scales with
eps k, as the CSF force), and spreading the GFM jump with the CLS transport's own kinematic weights (an adjoint
pair of width h) also makes it worse (-25.9 % against -16.8 %). What matters is how narrow the kinematics and
the force are compared with the layer, not the adjoint pairing itself: sharp kinematics with the sharp jump
gives +3.8 %. Figure: `docs/figures/adjoint_tension.png`. Notes: `docs/notes/43_adjoint_tension.md`.

Sharp interface kinematics for CLS (`cases/velocity_extension.py`, a prototype; the library transport is
unchanged): the conservative transport by `uf` is kept and, in a band about the interface, the face-normal
velocity is replaced by its value at the foot point on the interface (`delta = chi (V - u . n) n`, built on the
faces and advected without compression). A uniform velocity gives `delta = 0` (identical to the transport).
Capillary wave (normal-mode start): -16.4 % -> +3.4 % damping and -3.4 % -> +0.5 % frequency at N = 32
(+3.5 % / +2.0 % at N = 64 / 128); reversed vortex 0.0139 -> 0.0084 (N = 128); mass and the static-drop jump
unchanged. Not yet met: the energy exchange is not exact (the corrected kinematics and the GFM force are not
adjoint), and static-drop spurious currents decay about 10 times more slowly. Figure:
`docs/figures/velocity_extension.png`. Notes: `docs/notes/44_velocity_extension.md`.

Energy consistency of the corrected kinematics with the GFM force (`cases/adjoint_kinematics.py`): for a frozen
psi the transport is linear in the face velocity, `psi_t = -T uf`, and `F_adj = T^T dE/dpsi / V_f` satisfies the
semi-discrete identity to 1e-14. Measured against the sharp-interface energy `sigma L(eta)` (the one the GFM jump
is consistent with), the work mismatch between `F_GFM` and `F_adj` predicts the default transport's damping
deficit (-19.7 % / -5.8 % predicted, -16.6 % / -4.7 % measured at N = 32 / 64); after the band correction it is
only -2 % and does not cause the remaining over-damping (+7.7 %, of which about half is the time step). The band
area `sigma A_h(psi)` is not the GFM's energy and explains neither damping nor frequency. No force correction is
needed. Figure: `docs/figures/adjoint_kinematics.png`. Notes: `docs/notes/45_adjoint_kinematics.md`.

Attribution of the over-damping left after the band correction (`cases/residual_attribution_figures.py`;
eigenvalues, N = 32, exact curvature): of +7.7 %, about 4.5 points are the time discretisation (+3.2 % as
dt -> 0, measured down to dt/64, first order from dt/16 on; the same mode is tracked, eigenvector overlap
>= 0.995), 2.1 points the fd2 viscosity (+1.1 % with the CCD viscosity as dt -> 0), and +1.1 % remains. Solving
viscosity, surface tension and projection together (converged coupled iterations) changes it by 0.1 point or
less, and there is no interaction between the viscosity and the splitting. Figure:
`docs/figures/residual_attribution.png`. Notes: `docs/notes/46_residual_attribution.md`.

Case C prototype (`cases/case_c.py`): viscosity, GFM force, projection and the band-corrected kinematics solved
together in each implicit stage (fixed point, about 3 iterations per stage), with BE, Crank-Nicolson and
L-stable SDIRK2. With the faces rebuilt from the cells every step (as CenteredNS does), the second-order
integrators do not change the time-step dependence (N = 32: CN +8.5 / +6.8 / +5.4 / +4.5 % at dt / 2 / 4 / 8,
CenteredNS +7.7 / +6.4 / +5.2 / +4.4 %): the per-step approximate projection dominates. Keeping the face velocity
gives clean second order but a different limit (+8.4 %), because the cell and face velocities then evolve
separately. Figure: `docs/figures/case_c.png`. Notes: `docs/notes/47_case_c_time.md`.

One velocity field with a consistent projection (`cases/single_velocity.py`, the linear capillary-wave system
as matrices): the cell-centred velocity with the exact projection `Pi_c = 1 - G_c (D_c G_c)^+ D_c`
(`D_c = D_f I`, `G_c = -D_c^T`) is an orthogonal projector (1e-15), its semi-discrete eigenvalue equals the
dt -> 0 limit of CenteredNS (+3.23 % fd2, +1.14 % CCD at N = 32), and Crank-Nicolson / SDIRK2 on it converge at
second order and reach that limit at the default step (+3.20 % / +1.11 %, against CenteredNS's +7.72 % / +4.43 %).
The static-drop balance is kept (div-free share of the GFM force 8.6e-4 against 8.1e-4). Caveat: in the cell
form checkerboard-modulated pressure modes have the same small eigenvalue as the smoothest mode. A MAC
velocity with these operators over-damps (+12.9 %). Figure: `docs/figures/single_velocity.png`. Notes:
`docs/notes/48_single_velocity.md`.

Summary of B, the capillary-wave damping work of notes 38-48 (`docs/notes/49_b_summary.md`,
`docs/figures/b_summary.png`): the damping deficit on coarse grids is set by the CLS interface kinematics (the
crossing moves with a velocity averaged over +-h), not by the order of the viscous or face operators; energy
consistency and physical accuracy are different properties; wavenumber alone cannot separate an unresolved
physical layer from numerical noise; the collocated approximate projection keeps higher-order integrators at
first order. Development stops here: the band velocity correction and the exact cell projection are kept as
experimental prototypes in `cases/`, high-wavenumber damping and the adjoint surface tension are dropped, MAC and a
full Case C are deferred. For damping within 1 %, raising the resolution (N = 128 for this wave: +0.5 %) is the
cheaper and surer route.

Paper review before the benchmarks (`docs/notes/10_paper_errata.md`, last section): chapter 14 does not run the
seven-stage step of §11b1 — the capillary wave transports a geometric cell fraction (swept volume) on an
interface-fitted grid rebuilt every step, and the oscillating drop integrates the linear Rayleigh–Lamb mode on a
closed-curve chart without coupling to Navier–Stokes (the paper's own `force_admissible=0`). The capillary wave's
amplitude ratio after one period, 0.7936, mixes damping with the phase lag of a 16-point wave (corrected in
`docs/notes/58_phase3_capwave.md`: frequency errors of 4-9.5 % alone bring the ratio to about 0.84; it is not a
damping rate 6.2 times the viscous one, as first stated), and its grid resolves the liquid vorticity layer with
delta/h = 0.20. The Rayleigh–Taylor growth rate is asserted, not fitted.

The paper's standard two-phase step as one combination (`docs/notes/50_two_phase_imex.md`,
`docs/figures/two_phase_imex.png`): `TwoPhaseIMEX(grid, phi0, rho1=, rho2=, mu1=, mu2=, sigma=, gravity=,
faces="fd2"|"fccd", convection="central"|"fuccd"|"uccd6", transport="projected"|"ab2")` is `IMEXBDF2NS` with
`TwoPhaseCLS` (sharp GFM) attached. A static drop stays at rest to t = 0.5 for density ratios 1 to 1000
(jump 4.0105 / 4.0030 at N = 32 / 64, mass to 1e-14), as with `CenteredNS`. On the capillary wave the paper's
stage 1 — psi moved by the frozen projected face velocity `u_f^n` — is first order in time: the damping error at
16 points per wavelength goes -18.8 / -15.9 / -14.4 / -13.6 / -13.2 / -13.0 % as the step is halved five times,
and moving psi with the AB2 midpoint velocity `(1 + w/2) u_f^n - (w/2) u_f^{n-1}` removes it (-12.0 ... -12.8 %,
the same limit as `CenteredNS`). The remaining -12.7 % is the CLS kinematics of B. Independent runs of the case
scripts are spread over CPU cores (`cases/parallel.py`).

Swappable time integration (`docs/notes/51_multistep_schemes.md`, `52_two_phase_imex_rk.md`): `IMEXMultistepNS(scheme=...)`
takes any multistep IMEX pair — `sbdf1`...`sbdf4`, `cnab2`, `mcnab2`, `cnlf`, or (`bdf<k>`/`am<k>`/`theta<x>`,
`ext<k>`/`ab<k>`) — with coefficients from Lagrange polynomials on the actual time levels (variable steps exact,
orders lowered at start-up); `sbdf2` reproduces `IMEXBDF2NS` to 1e-12. `TwoPhaseIMEX` adds the interface-transport
velocity (`frozen`, `ab2`, `ab3`, `pc` predictor-corrector) and the psi integrator (`ssprk3`, `ssprk2`, `heun3`,
`rk4`, `euler`); `TwoPhaseIMEXRK(tableau=...)` advances psi inside the stages of ARS(1,1,1), (1,2,1), (2,2,2),
(2,3,3), (3,4,3), (4,4,3). Measured with kx != ky: the approximate projection's cross term drops every scheme to
first order at N = 16 and vanishes by N = 64; the incremental pressure correction caps multistep schemes at second
order (sbdf3's error is 6.5 times sbdf2's), while IMEX-RK with a projection per stage reaches third order for
Stokes flow and stalls near 3e-6 with convection.

Equations (`docs/notes/53_equations.md`): the discrete equations as implemented, in one place, with the error
structure, the options and the correspondence to the paper's seven stages.

Re-initialisation with the paper's two closures (`reinit_method="xi_sdf_vw"`, `docs/notes/54_reinit_closures.md`):
the profile width that closes the diffuse mass is ill-conditioned for a nearly flat interface (eps_eff/eps = 3.96 /
0.25 from an exact profile at N = 32 / 64) and, used every 10 steps on a capillary wave, sits at the search bounds
and puts steps into the amplitude (RMS error 1.1-2.4 times); every re-initialisation also roughens psi
enough for its CCD curvature to scatter (band curvature 3.5...4.6 -> -6.3...18.0), shifting a static drop's jump by
+13 %.

Density-ratio projection with FCCD (`docs/notes/55_density_ratio_fccd.md`, manufactured `p_k = rho_k s + j0 H`): the
smooth form `p - H j_ext` is 4th order at equal densities (residual 1.3e-4 -> 5.3e-7, N = 32 -> 128) but does not
converge from a density ratio of 10; fd2's GFM is 2nd order at any ratio; switching the face gradient to fd2 near
the interface does not repair FCCD. A phase-wise closure of the CCD line systems (or the paper's HFE) is needed.

Phase 2, static drop compared three ways (`docs/notes/56_phase2_static.md`, `cases/phase2_static.py`): Basilisk's VOF
+ tension.h, `CenteredNS` + CLS/GFM and `TwoPhaseIMEX`, density ratios 1 to 1000, N = 32 and 64, to t = 1. All are
stable; the GFM jump error is 3.5-3.9 times smaller than VOF's (0.075 % against 0.29 % at N = 64; the paper's CSF
path reports 1.23 %), the peak parasitic velocity 25-90 times smaller, while VOF's decays further by t = 1; the
two time integrations differ by 1.5e-4 or less in the jump.

Phase 3, the paper's capillary wave (`docs/notes/57_water_air_stability.md`, `docs/notes/58_phase3_capwave.md`):
with water-air densities every CLS stepper grows unstable after 3-5 periods without re-initialisation (also at
density ratio 1 and at a quarter of the step; Basilisk's VOF decays physically), while the paper's xi-SDF every 10
or even 50 steps keeps it stable. Fitted against the two-fluid normal mode (beta = -0.803 1/s, omega = 134.34 1/s)
at 16 points per wavelength (amplitude from the column integral of phase 1): TwoPhaseIMEX with ab2 transport damps
0.5 times the physical rate with -9.3 % frequency, VOF 2.5 times with -4.0 % (1.4 and 1.3 times, -4.2 and -1.3 %,
at 32 points); the paper's own frozen transport amplifies (beta = +1.5). The paper's ratio at t = T falls in the
range of all methods (0.64-0.87) because of the phase lag, and depends on how the amplitude is measured.
The other three benchmarks (`docs/notes/59_phase3_drop_bubble.md`, `docs/figures/phase3_benchmarks.png`): the
oscillating drop's frequency is within 3 % of Rayleigh-Lamb for every method, but with psi's CCD curvature the CLS
amplitude grows even with xi-SDF every 10 steps (re-initialisation roughens the curvature, note 54) and stops with
height-function curvature; the rising bubble runs to 0.06 s with re-initialisation (without it, it blows up at
0.039 s, where the paper reports its unresolved breakdown), CLS and VOF agree to 1 % on the centroid while the paper
rises about 17 % more and its mean rise velocity is inconsistent with its own centroid; the Rayleigh-Taylor linear
growth rate is 1.721-1.731 against 1.754 for the viscous, capillary normal mode (the paper's 1.78 omits both).

Verdict on the paper (`docs/notes/60_paper_verdict.md`): fifteen claims checked against measurements, with the
recommended configuration of this port: SBDF2, ab2 interface transport, xi-SDF re-initialisation every 10 steps,
height-function curvature for closed interfaces, fd2 GFM faces, tension.h's step times sqrt(5/12), and a
volume-change projection tolerance for dimensional problems.

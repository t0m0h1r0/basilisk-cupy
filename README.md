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
| ① Spatial operators | `fd2` (Basilisk-compatible 2nd order): `FaceComplex`, `CellDerivative` |
| ② Linear solvers | Basilisk-style multigrid (`mg_solve`, red/black Gauss–Seidel, adaptive `nrelax`) with a separate high-order residual hook |
| ④ Projection | Basilisk `project()` (approximate projection, variable density) |
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

## Tests

```bash
pip install -e ".[dev]"      # add ",gpu" for cupy
pytest
```

Tests run on numpy; when cupy is importable every backend-parametrised test
also runs on cupy and a numpy/cupy agreement test is enabled.

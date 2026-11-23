"""Python/CuPy port of Basilisk with swappable numerical layers."""

from .boundary import BoundaryConditions, Dirichlet, Neumann, Periodic
from .fields import CellField, FaceField
from .grid import CartesianGrid
from .linear import MGStats, poisson
from .projection import project
from .scheme import Scheme, build_scheme, load_scheme

__all__ = [
    "BoundaryConditions", "CartesianGrid", "CellField", "Dirichlet", "FaceField", "MGStats",
    "Neumann", "Periodic", "Scheme", "build_scheme", "load_scheme", "poisson", "project",
]
__version__ = "0.1.0"

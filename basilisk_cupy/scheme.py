"""Scheme declaration: which implementation fills each layer.

A scheme is described by a nested mapping (usually loaded from YAML) in the
same spirit as TwoPhaseFlow's ``method:`` section::

    grid:
      cells: [64, 64]
      backend: numpy
    operators:
      face: fd2          # FaceComplex shared by projection and face forces
      cell: fd2          # CellDerivative

Only the operator layer has implementations yet. New implementations are
added with :func:`register` and picked by name; :meth:`Scheme.validate`
rejects inconsistent combinations at start-up instead of mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discretization.base import CellDerivative, FaceComplex
from .discretization.fd2 import FD2CellDerivative, FD2FaceComplex
from .grid.cartesian import CartesianGrid

_REGISTRY: dict[str, dict[str, type]] = {
    "face": {"fd2": FD2FaceComplex},
    "cell": {"fd2": FD2CellDerivative},
}


def register(layer: str, name: str, cls: type) -> None:
    """Register an implementation (e.g. ``register("face", "fccd", FCCDFaceComplex)``)."""
    _REGISTRY.setdefault(layer, {})[name] = cls


def available(layer: str) -> list[str]:
    return sorted(_REGISTRY.get(layer, {}))


def _build(layer: str, name: str):
    try:
        return _REGISTRY[layer][name]()
    except KeyError:
        raise ValueError(f"unknown {layer!r} implementation {name!r}; available: {available(layer)}") from None


@dataclass
class Scheme:
    grid: CartesianGrid
    face: FaceComplex
    cell: CellDerivative
    config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        need = max(self.face.stencil_ghosts, self.cell.stencil_ghosts)
        if self.grid.ghosts < need:
            raise ValueError(f"grid has {self.grid.ghosts} ghost layer(s) but the operators need {need}")


def build_scheme(config: dict[str, Any]) -> Scheme:
    gcfg = dict(config.get("grid", {}))
    ops = dict(config.get("operators", {}))
    face = _build("face", ops.get("face", "fd2"))
    cell = _build("cell", ops.get("cell", "fd2"))
    ghosts = max(int(gcfg.get("ghosts", 1)), face.stencil_ghosts, cell.stencil_ghosts)
    grid = CartesianGrid(
        n=tuple(gcfg.get("cells", (64, 64))),
        length=tuple(gcfg["length"]) if "length" in gcfg else None,
        origin=tuple(gcfg["origin"]) if "origin" in gcfg else None,
        ghosts=ghosts,
        backend=gcfg.get("backend", "numpy"),
    )
    scheme = Scheme(grid=grid, face=face, cell=cell, config=config)
    scheme.validate()
    return scheme


def load_scheme(path: str | Path) -> Scheme:
    import yaml  # noqa: PLC0415

    with open(path, encoding="utf-8") as fh:
        return build_scheme(yaml.safe_load(fh) or {})

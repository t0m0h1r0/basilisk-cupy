"""Scheme declaration: which implementation fills each layer.

A scheme is described by a nested mapping (usually loaded from YAML) in the
same spirit as TwoPhaseFlow's ``method:`` section::

    grid:
      cells: [64, 64]
      backend: numpy
    operators:
      face: fd2          # FaceComplex shared by projection and face forces
      cell: fd2          # CellDerivative
    terms:
      convection: bcg    # bcg (CenteredNS's own) | central | uccd6 | fuccd4 | fuccd1
      sigma: 1.0e-3      # optional, UCCD6 / FUCCD dissipation coefficient

New implementations are added with :func:`register` and picked by name;
:meth:`Scheme.validate` rejects inconsistent combinations at start-up
instead of mid-run. A convection entry is a factory ``(face, cell, **options)``
returning an object with ``tendency(u, uf)`` (method-of-lines steppers such as
:class:`~basilisk_cupy.time.imex_bdf2.IMEXBDF2NS`); ``bcg`` means the stepper's
built-in BCG and builds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discretization.base import CellDerivative, FaceComplex
from .discretization.ccd import CCDCellDerivative, FCCDFaceComplex
from .discretization.fd2 import FD2CellDerivative, FD2FaceComplex
from .grid.cartesian import CartesianGrid
from .terms.convection.central import CentralFluxConvection
from .terms.convection.fuccd import FUCCDConvection
from .terms.convection.uccd6 import UCCD6Convection


def _fccd_of(face):
    return face if isinstance(face, FCCDFaceComplex) else FCCDFaceComplex()


def _ccd_of(face, cell):
    if isinstance(cell, CCDCellDerivative):
        return cell
    return face.ccd if isinstance(face, FCCDFaceComplex) else CCDCellDerivative()

_REGISTRY: dict[str, dict[str, type]] = {
    "face": {"fd2": FD2FaceComplex, "fccd": FCCDFaceComplex},
    "cell": {"fd2": FD2CellDerivative, "ccd": CCDCellDerivative},
    "convection": {
        "bcg": lambda face, cell, **kw: None,
        "central": lambda face, cell, **kw: CentralFluxConvection(face),
        "uccd6": lambda face, cell, **kw: UCCD6Convection(_ccd_of(face, cell), **kw),
        "fuccd4": lambda face, cell, **kw: FUCCDConvection(_fccd_of(face), r=4, **kw),
        "fuccd1": lambda face, cell, **kw: FUCCDConvection(_fccd_of(face), r=1, **kw),
    },
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
    convection: Any = None  # tendency(u, uf) object, or None for the stepper's own (BCG)

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
    tcfg = dict(config.get("terms", {}))
    conv_name = tcfg.pop("convection", "bcg")
    try:
        factory = _REGISTRY["convection"][conv_name]
    except KeyError:
        raise ValueError(f"unknown 'convection' implementation {conv_name!r}; "
                         f"available: {available('convection')}") from None
    convection = factory(face, cell, **tcfg)
    scheme = Scheme(grid=grid, face=face, cell=cell, config=config, convection=convection)
    scheme.validate()
    return scheme


def load_scheme(path: str | Path) -> Scheme:
    import yaml  # noqa: PLC0415

    with open(path, encoding="utf-8") as fh:
        return build_scheme(yaml.safe_load(fh) or {})

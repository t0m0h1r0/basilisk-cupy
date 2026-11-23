"""One-dimensional grid mappings ``x(xi)`` for non-uniform (stretched) Cartesian grids.

A non-uniform axis is the image of the uniform computational axis
``xi in [0, L]`` (cell faces at ``xi = k L/n``, centres half way) under a
smooth, increasing map ``x = origin + L m(xi/L)`` (TwoPhaseFlow paper §10:
``10_grid.tex`` §grid_coord, §grid_gen). The operators work in ``xi`` and
convert with the metric ``J = dxi/dx`` (see :mod:`basilisk_cupy.grid.cartesian`).
Because the grid is a map, every multigrid level is the same map sampled at
its own ``xi``: coarse faces are fine faces, coarse centres are fine faces.

``m`` maps ``s in [0, 1]`` onto ``[0, 1]``. Outside ``[0, 1]`` (ghost cells) a
wall axis is extended by reflection about the wall, ``m(-s) = -m(s)`` and
``m(1+s) = 2 - m(1-s)``, so that a wall lies half way between a ghost centre
and its mirror centre in physical space as well; a periodic axis by
translation, ``m(s+1) = m(s) + 1``.

The paper builds the coordinates of a monitor (density) map with the
trapezoidal rule and notes that this caps the grid convergence at
``O(dxi^2)``; :class:`MonitorMapping` instead integrates ``s(x)`` with the
composite Boole rule on a fine grid (``O(dx^6)``) and inverts it by cubic
Hermite interpolation with the exact slope ``dx/ds = W/omega`` (error near
round-off for a smooth ``omega``), so that the map itself does not limit the
order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class Mapping:
    """Base class: ``unit(s)`` on ``[0, 1]``; ``periodic`` selects the ghost extension."""

    periodic: bool = False

    def unit(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def derivative(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """``dm/ds`` on ``[0, 1]`` (exact; used for the wall metric of Neumann data)."""
        raise NotImplementedError

    @property
    def is_uniform(self) -> bool:
        return False

    def extended(self, s) -> np.ndarray:
        """``m(s)`` for any real ``s`` (ghosts included), host float64."""
        s = np.asarray(s, dtype=np.float64)
        if self.periodic:
            k = np.floor(s)
            return self.unit(s - k) + k
        out = np.empty_like(s)
        lo, hi = s < 0.0, s > 1.0
        mid = ~(lo | hi)
        out[mid] = self.unit(s[mid])
        out[lo] = -self.unit(-s[lo])
        out[hi] = 2.0 - self.unit(2.0 - s[hi])
        return out


@dataclass(frozen=True)
class UniformMapping(Mapping):
    periodic: bool = False

    def unit(self, s):
        return np.asarray(s, dtype=np.float64)

    def derivative(self, s):
        return np.ones_like(np.asarray(s, dtype=np.float64))

    @property
    def is_uniform(self) -> bool:
        return True


@dataclass(frozen=True)
class TanhMapping(Mapping):
    """Wall clustering ``m(s) = (1 + tanh(beta (2s - 1)) / tanh(beta)) / 2`` (both walls),
    or one-sided ``1 + tanh(beta (s - 1)) / tanh(beta)`` (``side="lower"``) and its mirror."""

    beta: float = 2.0
    side: str = "both"
    periodic: bool = False

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be > 0")
        if self.side not in ("both", "lower", "upper"):
            raise ValueError("side must be 'both', 'lower' or 'upper'")
        if self.periodic:
            raise ValueError("TanhMapping clusters at walls; use SineMapping for a periodic axis")

    def unit(self, s):
        s = np.asarray(s, dtype=np.float64)
        b, tb = self.beta, math.tanh(self.beta)
        if self.side == "both":
            return 0.5 * (1.0 + np.tanh(b * (2.0 * s - 1.0)) / tb)
        if self.side == "lower":
            return 1.0 + np.tanh(b * (s - 1.0)) / tb
        return np.tanh(b * s) / tb

    def derivative(self, s):
        s = np.asarray(s, dtype=np.float64)
        b, tb = self.beta, math.tanh(self.beta)
        if self.side == "both":
            return b / tb / np.cosh(b * (2.0 * s - 1.0)) ** 2
        if self.side == "lower":
            return b / tb / np.cosh(b * (s - 1.0)) ** 2
        return b / tb / np.cosh(b * s) ** 2


@dataclass(frozen=True)
class SineMapping(Mapping):
    """``m(s) = s - a sin(2 pi (s - s0)) / (2 pi)``: smooth, periodic-compatible, cells
    smallest near ``s0`` (``0 <= a < 1`` keeps it increasing; ``dx/ds`` ranges over ``[1-a, 1+a]``)."""

    amplitude: float = 0.5
    s0: float = 0.0
    periodic: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.amplitude < 1.0:
            raise ValueError("amplitude must be in [0, 1)")

    def unit(self, s):
        s = np.asarray(s, dtype=np.float64)
        a = self.amplitude
        return s - a * (np.sin(2 * math.pi * (s - self.s0)) - math.sin(-2 * math.pi * self.s0)) / (2 * math.pi)

    def derivative(self, s):
        return 1.0 - self.amplitude * np.cos(2 * math.pi * (np.asarray(s, dtype=np.float64) - self.s0))


@dataclass(frozen=True)
class MonitorMapping(Mapping):
    """Equidistribution of a monitor (density) function: ``int_0^x omega = s int_0^1 omega``.

    ``omega(x) > 0`` is given on ``x in [0, 1]`` (the unit axis; a density of
    ``alpha`` means cells ``alpha`` times smaller). TwoPhaseFlow's interface
    monitor is ``omega = 1 + (alpha - 1) exp(-(phi/eps)^2)`` (eq. ``grid_delta``),
    its wall term ``1 + A_W exp(-(d/eps_W)^2)`` (eq. ``grid_composite_monitor``).
    """

    omega: Callable  # compared by identity: two maps are equal only with the same function
    periodic: bool = False
    resolution: int = 1 << 14  # Hermite table intervals (each one Boole panel of 4 sub-intervals)
    _table: tuple = field(init=False, repr=False, compare=False, default=None)
    _total: float = field(init=False, repr=False, compare=False, default=0.0)

    def __post_init__(self) -> None:
        m = self.resolution
        omega = self.omega
        # s(x) = int_0^x omega / W with W = int_0^1 omega, then x(s) by Hermite inversion
        xs = np.linspace(0.0, 1.0, 4 * m + 1)
        w = np.asarray(omega(xs), dtype=np.float64)
        if np.any(w <= 0.0):
            raise ValueError("the monitor function must be positive")
        # composite Boole rule on the fine x grid for s(x) (O(dx^6)); cumulative over panels of 4 sub-intervals
        hx = xs[1] - xs[0]
        panels = (7 * w[:-4:4] + 32 * w[1:-3:4] + 12 * w[2:-2:4] + 32 * w[3:-1:4] + 7 * w[4::4]) * (2 * hx / 45)
        s_nodes = np.concatenate([[0.0], np.cumsum(panels)])
        total = s_nodes[-1]
        x_nodes = xs[::4]
        s_nodes = s_nodes / total
        dxds = total / np.asarray(omega(x_nodes), dtype=np.float64)  # dx/ds at the table nodes
        object.__setattr__(self, "_table", (s_nodes, x_nodes, dxds))
        object.__setattr__(self, "_total", total)

    def derivative(self, s):
        """``dx/ds = W / omega(x(s))``."""
        return self._total / np.asarray(self.omega(self.unit(s)), dtype=np.float64)

    def unit(self, s):
        """Inverse of ``s(x)`` by cubic Hermite interpolation with the exact slope ``dx/ds``."""
        s = np.asarray(s, dtype=np.float64)
        s_nodes, x_nodes, dxds = self._table
        i = np.clip(np.searchsorted(s_nodes, s, side="right") - 1, 0, len(s_nodes) - 2)
        s0, s1 = s_nodes[i], s_nodes[i + 1]
        d = s1 - s0
        t = (s - s0) / d
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t * t * (3 - 2 * t)
        h11 = t * t * (t - 1)
        return h00 * x_nodes[i] + h10 * d * dxds[i] + h01 * x_nodes[i + 1] + h11 * d * dxds[i + 1]


def uniform() -> UniformMapping:
    return UniformMapping()

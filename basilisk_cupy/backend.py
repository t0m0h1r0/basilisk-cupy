"""Array backend selection (numpy or cupy).

This module is the only place that imports cupy. Every other module receives
the array namespace ``xp`` from a :class:`~basilisk_cupy.grid.CartesianGrid`,
so the same code runs on CPU (numpy) for testing and on GPU (cupy).
"""

from __future__ import annotations

import numpy as np

_BACKENDS = ("numpy", "cupy")


def get_xp(name: str = "numpy"):
    """Return the array module for ``name`` ("numpy" or "cupy")."""
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy as cp  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on machine
            raise RuntimeError("backend 'cupy' requested but cupy is not installed") from exc
        return cp
    raise ValueError(f"unknown backend {name!r}; expected one of {_BACKENDS}")


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def array_module(array):
    """Array namespace (numpy or cupy) that owns ``array``; numpy for scalars and host arrays."""
    if type(array).__module__.startswith("cupy"):
        import cupy as cp  # noqa: PLC0415

        return cp
    return np


def to_numpy(array):
    """Copy an array (numpy or cupy) to host memory as a numpy array."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def to_float(value) -> float:
    """Read a 0-d device scalar back to the host (the only device->host sync point)."""
    return float(value)

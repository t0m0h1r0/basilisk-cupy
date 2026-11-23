"""Time-step control (Basilisk ``timestep.h`` and ``dtnext()`` in ``grid/events.h``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backend import to_float
from ..fields import FaceField

TEPS = 1e-9


@dataclass
class Clock:
    t: float = 0.0
    i: int = 0
    dt: float = 1.0
    tnext: float = float("inf")
    #: Basilisk keeps the previous CFL step in a static variable of timestep().
    previous: float = 0.0
    events: list[float] = field(default_factory=list)  # future times that must be hit exactly


def timestep(uf: FaceField, dtmax: float, cfl: float, clock: Clock) -> float:
    """CFL-limited step with Basilisk's smoothing of increases.

    ``dtmax`` is reduced to ``CFL * min(Delta/|uf|)``. An increase over the
    previous value is damped: ``dt = (previous + 0.1 dt)/1.1``.
    """
    xp = uf.grid.xp
    if clock.t == 0.0:
        clock.previous = 0.0
    dtmax /= cfl
    for ax in range(uf.grid.ndim):
        speed = to_float(xp.max(xp.abs(uf[ax])))
        if speed > 0.0:
            dtmax = min(dtmax, uf.grid.h[ax] / speed)
    dtmax *= cfl
    if dtmax > clock.previous:
        dtmax = (clock.previous + 0.1 * dtmax) / 1.1
    clock.previous = dtmax
    return dtmax


def dtnext(dt: float, clock: Clock) -> float:
    """Adjust ``dt`` so that the next event time is hit exactly (Basilisk ``dtnext``)."""
    upcoming = [te for te in clock.events if te > clock.t + TEPS * max(1.0, abs(clock.t))]
    tnext = min(upcoming) if upcoming else float("inf")
    if tnext != float("inf") and tnext > clock.t:
        n = int((tnext - clock.t) / dt)
        if n == 0:
            dt = tnext - clock.t
        else:
            dt1 = (tnext - clock.t) / n
            if dt1 > dt * (1.0 + TEPS):
                dt = (tnext - clock.t) / (n + 1)
            elif dt1 < dt:
                dt = dt1
    clock.tnext = clock.t + dt
    return dt

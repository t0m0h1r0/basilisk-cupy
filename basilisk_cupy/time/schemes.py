"""Time-integration schemes as data: multistep coefficients from Lagrange polynomials, explicit Runge--Kutta tables.

Multistep IMEX schemes (:class:`MultistepScheme`) advance ``du/dt = E(u) + V(u)`` (``E`` explicit, ``V``
implicit) as

    sum_{j>=0} alpha_j u^{n+1-j} = dt [ sum_{j>=0} beta_j E^{n-j} + sum_{j>=0} gamma_j V^{n+1-j} ]

with the coefficients computed every step from the actual time levels (variable steps are exact for the
Lagrange-based families), so that one stepper covers

* implicit families: ``bdf1`` ... ``bdf4`` (``alpha`` from the derivative of the interpolant at
  ``t^{n+1}``, ``gamma = (1)``), ``am2`` ... ``am4`` (Adams--Moulton: ``alpha = (1, -1)``, ``gamma`` the
  integral of the interpolant through ``t^{n+1}, t^n, ...`` over the step; ``am2`` is Crank--Nicolson),
  ``theta<value>`` (``gamma = (theta, 1 - theta)``, e.g. ``theta0.5``; ``theta1`` is backward Euler);
* explicit families: ``ext1`` ... ``ext4`` (extrapolation to ``t^{n+1}``, the partner of BDF),
  ``ab1`` ... ``ab4`` (Adams--Bashforth: the integral of the extrapolant over the step, the partner of AM
  and theta);
* fixed-coefficient pairs (constant step only): ``mcnab2`` (modified CNAB2 of Ascher, Ruuth & Wetton 1995:
  ``gamma = (9/16, 3/8, 1/16)``, ``beta = (3/2, -1/2)``), ``cnlf`` (Crank--Nicolson leap-frog:
  ``alpha = (1/2, 0, -1/2)``, ``beta = (1)``, ``gamma = (1/2, 0, 1/2)``).

Named pairs: ``sbdf1`` ... ``sbdf4`` (``bdf<k>`` + ``ext<k>``), ``cnab2`` (``am2`` + ``ab2``), ``mcnab2``,
``cnlf``. A pair is any ``(implicit, explicit)``; mismatched families (e.g. ``bdf2`` + ``ab2``) are allowed
but first order. With too little history the orders are lowered to what is available (the first step of
``sbdf3`` is ``sbdf1``, the second ``sbdf2``; ``mcnab2`` / ``cnlf`` start with ``cnab2`` / ``sbdf1``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# -- Lagrange weights ---------------------------------------------------------------------------------


def _weights(nodes, moments) -> np.ndarray:
    """``w`` with ``sum_j w_j x_j^m = moments[m]`` (``m = 0 .. len(nodes) - 1``)."""
    x = np.asarray(nodes, dtype=float)
    V = np.vander(x, len(x), increasing=True).T
    return np.linalg.solve(V, np.asarray(moments, dtype=float))


def bdf_alpha(k: int, offsets) -> np.ndarray:
    """BDF-k: ``alpha_j = dt l_j'(0)`` on the nodes ``0, -s_1, -s_2, ...`` (in units of the current step)."""
    nodes = [0.0] + [-s for s in offsets[:k]]
    return _weights(nodes, [0.0, 1.0] + [0.0] * (k - 1))


def ext_beta(k: int, offsets) -> np.ndarray:
    """Extrapolation to ``t^{n+1}`` (node 0) from ``-s_1, ..., -s_k``."""
    nodes = [-s for s in offsets[:k]]
    return _weights(nodes, [1.0] + [0.0] * (k - 1))


def _step_integral(k: int) -> list:
    """``int_{-1}^{0} x^m dx`` for ``m < k``."""
    return [(0.0 - (-1.0) ** (m + 1)) / (m + 1) for m in range(k)]


def ab_beta(k: int, offsets) -> np.ndarray:
    """Adams--Bashforth: mean over the step ``[-1, 0]`` of the interpolant through ``-s_1, ..., -s_k``."""
    nodes = [-s for s in offsets[:k]]
    return _weights(nodes, _step_integral(k))


def am_gamma(k: int, offsets) -> np.ndarray:
    """Adams--Moulton of order ``k`` (``k`` nodes ``0, -s_1, ..., -s_{k-1}``): ``am2`` is the trapezoidal rule."""
    nodes = [0.0] + [-s for s in offsets[:k - 1]]
    return _weights(nodes, _step_integral(k))


@dataclass(frozen=True)
class Coefficients:
    alpha: tuple  # on u^{n+1}, u^n, u^{n-1}, ...
    beta: tuple   # on E^n, E^{n-1}, ...
    gamma: tuple  # on V^{n+1}, V^n, ...

    @property
    def depth(self) -> int:
        """Number of past levels used (``u^n`` counts as one)."""
        return max(len(self.alpha) - 1, len(self.beta), len(self.gamma) - 1)


_FIXED = {
    "mcnab2": Coefficients((1.0, -1.0), (1.5, -0.5), (9.0 / 16.0, 3.0 / 8.0, 1.0 / 16.0)),
    "cnlf": Coefficients((0.5, 0.0, -0.5), (1.0,), (0.5, 0.0, 0.5)),
}
_STARTUP = {"mcnab2": "cnab2", "cnlf": "sbdf1"}
NAMED = {**{f"sbdf{k}": (f"bdf{k}", f"ext{k}") for k in range(1, 5)}, "cnab2": ("am2", "ab2"),
         "cnab1": ("am2", "ab1")}


def _parse(name: str):
    m = re.fullmatch(r"(bdf|ext|ab|am)(\d)", name)
    if m:
        return m.group(1), int(m.group(2))
    m = re.fullmatch(r"theta([0-9.]+)", name)
    if m:
        return "theta", float(m.group(1))
    raise ValueError(f"unknown multistep component {name!r}")


class MultistepScheme:
    """``MultistepScheme("sbdf2")``, ``MultistepScheme(("am2", "ab2"))``, ``MultistepScheme("mcnab2")``."""

    def __init__(self, spec) -> None:
        if isinstance(spec, str) and spec in _FIXED:
            self.name, self.fixed = spec, _FIXED[spec]
            self.implicit = self.explicit = None
            return
        self.fixed = None
        if isinstance(spec, str):
            if spec not in NAMED:
                raise ValueError(f"unknown multistep scheme {spec!r}; named: {sorted(NAMED) + sorted(_FIXED)}")
            impl, expl = NAMED[spec]
            self.name = spec
        else:
            impl, expl = spec
            self.name = f"{impl}+{expl}"
        self.implicit, self.explicit = _parse(impl), _parse(expl)
        fam, k = self.implicit
        if fam not in ("bdf", "am", "theta"):
            raise ValueError(f"{impl!r} is not an implicit family (bdf<k>, am<k>, theta<value>)")
        if fam == "theta" and not 0.0 < k <= 1.0:
            raise ValueError("theta must be in (0, 1] (the implicit part may not vanish)")
        if self.explicit[0] not in ("ext", "ab"):
            raise ValueError(f"{expl!r} is not an explicit family (ext<k>, ab<k>)")
        if fam in ("bdf", "am") and not 1 <= k <= 4 or self.explicit[1] < 1 or self.explicit[1] > 4:
            raise ValueError("orders 1 ... 4 are available")

    @property
    def max_depth(self) -> int:
        if self.fixed is not None:
            return self.fixed.depth
        fam, k = self.implicit
        d_impl = {"bdf": k, "am": max(k - 1, 1), "theta": 1}[fam]
        return max(d_impl, self.explicit[1])

    def coefficients(self, steps, v_levels: int | None = None) -> Coefficients:
        """``steps``: ``dt_n, dt_{n-1}, ...`` (current step first), one per available level ``u^n, u^{n-1}, ...``
        (``E`` is known on the same levels); ``v_levels``: number of known past implicit tendencies ``V^n,
        V^{n-1}, ...`` (default: as many as ``u`` levels). Orders are lowered to what the history allows."""
        hist = len(steps)  # u^n plus len(steps) - 1 earlier levels
        v_levels = hist if v_levels is None else int(v_levels)
        if self.fixed is not None:
            if hist >= self.fixed.depth and v_levels >= len(self.fixed.gamma) - 1:
                if any(abs(s - steps[0]) > 1e-9 * steps[0] for s in steps[1:self.fixed.depth]):
                    raise ValueError(f"{self.name} has constant-step coefficients; use a fixed step")
                return self.fixed
            return MultistepScheme(_STARTUP[self.name]).coefficients(steps, v_levels)
        dt = steps[0]
        offsets = [sum(steps[:i + 1]) / dt for i in range(len(steps))]  # t^{n+1} - t^{n+1-j}, j >= 1, in dt
        fam, k = self.implicit
        if fam == "bdf":
            kk = min(k, hist)
            alpha, gamma = bdf_alpha(kk, offsets), np.array([1.0])
        elif fam == "am":
            kk = min(k, hist + 1, v_levels + 1)
            alpha, gamma = np.array([1.0, -1.0]), am_gamma(kk, offsets)
        elif v_levels >= 1 or k == 1.0:
            alpha, gamma = np.array([1.0, -1.0]), np.array([k, 1.0 - k])
        else:  # theta without V^n: backward Euler for the first step
            alpha, gamma = np.array([1.0, -1.0]), np.array([1.0])
        efam, ek = self.explicit
        ek = min(ek, hist)
        # explicit nodes: t^n, t^{n-1}, ... at offsets 1, 1 + s, ... (relative to t^{n+1})
        beta = (ext_beta if efam == "ext" else ab_beta)(ek, offsets)
        return Coefficients(tuple(float(a) for a in alpha), tuple(float(b) for b in beta),
                            tuple(float(g) for g in gamma))


# -- explicit Runge--Kutta tables (Butcher form) ----------------------------------------------------------

EXPLICIT_RK = {
    "euler": ([[0.0]], [1.0], [0.0]),
    "ssprk2": ([[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5], [0.0, 1.0]),
    "ssprk3": ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]], [1 / 6, 1 / 6, 2 / 3], [0.0, 1.0, 0.5]),
    "rk4": ([[0.0] * 4, [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.0, 0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6],
            [0.0, 0.5, 0.5, 1.0]),
    "heun3": ([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]], [0.25, 0.0, 0.75], [0.0, 1 / 3, 2 / 3]),
}


def explicit_rk(name: str):
    """``(A, b, c)`` of an explicit Runge--Kutta method."""
    try:
        return EXPLICIT_RK[name]
    except KeyError:
        raise ValueError(f"unknown explicit Runge-Kutta method {name!r}; available: {sorted(EXPLICIT_RK)}") from None


__all__ = ["MultistepScheme", "Coefficients", "EXPLICIT_RK", "explicit_rk", "bdf_alpha", "ext_beta", "ab_beta",
           "am_gamma"]


def explicit_real_limit(scheme: "MultistepScheme", r_max: float = 4.0, samples: int = 4001) -> float:
    """Largest ``r dt`` for which the scheme's explicit part is stable on ``u' = -r u`` at constant steps (``u^{n+1}``
    and the history weighted by ``alpha``, the explicit term by ``beta``): EXT2/BDF2 4/3, EXT3/BDF3 0.95, AB2/AM2 1;
    0 when no positive ``r dt`` is stable (CNLF's leapfrog). The explicit hyperviscosity of UCCD6 / FUCCD4 is
    bounded with it (docs/notes/72)."""
    c = scheme.coefficients([1.0] * 4)
    a, b = np.asarray(c.alpha, float), np.asarray(c.beta, float)
    k = max(len(a) - 1, len(b))
    def stable(x):
        poly = np.zeros(k + 1)
        poly[:len(a)] += a
        poly[1:1 + len(b)] += x * b
        return np.abs(np.roots(poly)).max() <= 1.0 + 1e-12

    last = 0.0
    for x in np.linspace(r_max / samples, r_max, samples):
        if not stable(x):
            lo, hi = last, float(x)
            for _ in range(60):  # bisection to round-off (EXT2/BDF2: 4/3 exactly)
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if stable(mid) else (lo, mid)
            return lo
        last = float(x)
    return last

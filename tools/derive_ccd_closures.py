"""Derive the wall closures of CCD / FCCD on the cell-centred grid (exact rationals).

Basilisk puts walls on cell faces, so the first cell centre sits at h/2 from
the wall, unlike the nodal grid of Chu & Fan (1998) used in TwoPhaseFlow. The
interior CCD equations are unchanged; only the rows of the first cell and
the wall-face values need new closures.

Each closure is a linear relation ``sum_j c_j h^{d_j} f^{(d_j)}(x_j) = 0``
between terms (derivative order d_j, position x_j in units of h, wall at 0,
cell k centred at k + 1/2). One coefficient is normalised to 1 and the others
are fixed by Taylor matching about the wall to the highest possible order.

Run ``python tools/derive_ccd_closures.py`` (needs sympy) to regenerate
``basilisk_cupy/discretization/ccd_coefficients.py``.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

NT = 14
HALF = sp.Rational(1, 2)


def centre(k: int):
    return HALF + k


def _taylor(d, x0):
    """Coefficients of h^m f^(m)(0) in h^d f^(d)(x0 h)."""
    return [sp.Integer(0) if m < d else sp.Rational(x0) ** (m - d) / sp.factorial(m - d) for m in range(NT)]


def closure(main, terms):
    """``main`` = (d, x) with coefficient 1, ``terms`` = list of (d, x); returns (coeffs, order).

    ``order`` is the first Taylor power that is not matched (the closure is
    exact for polynomials of degree < order).
    """
    syms = sp.symbols(f"c0:{len(terms)}")
    total = _taylor(*main)
    for s, (d, x) in zip(syms, terms):
        t = _taylor(d, x)
        total = [total[m] + s * t[m] for m in range(NT)]
    sol = sp.solve(total[: len(terms)], syms, dict=True)
    if not sol or len(sol[0]) != len(terms):
        raise ValueError("singular closure")
    sol = sol[0]
    m = len(terms)
    while m < NT and sp.simplify(total[m].subs(sol)) == 0:
        m += 1
    return [sol[s] for s in syms], m


#: Wall datum terms: Dirichlet value f(0) or Neumann gradient h f'(0).
DATUM = {"dirichlet": (0, 0), "neumann": (1, 0)}


#: Implicit terms (derivative order, cell) of the Eq-I and Eq-II rows, and the number of cell values.
#: All rows stay on the first two cells and the wall, the width of the interior
#: three-cell stencil; the accuracy comes from the implicit derivatives (CCD's
#: principle), not from a wider one-sided stencil. A boundary closure one order
#: below the interior (two for a second derivative) keeps the global order
#: (Gustafsson 1975). The earlier wide closures (datum + 4..6 cells, 6th/5th
#: order at the wall) were no more accurate on any Navier--Stokes run and lost
#: digits to round-off at N = 256 (max |D2| h^2 rows of ~20).
STENCILS = {
    # Row I through f''_0, f'_1, f''_1 (5th order f'); row II through f'_0, f''_1
    # (3rd order f''). Row II cannot also take f'_1: with the same implicit
    # unknowns as row I the system is singular.
    "dirichlet": {"I": ([(2, 0), (1, 1), (2, 1)], 2), "II": ([(1, 0), (2, 1)], 2)},
    # Row II through f'_1, f''_1 (the datum is already a first derivative).
    "neumann": {"I": ([(2, 0), (1, 1), (2, 1)], 2), "II": ([(1, 1), (2, 1)], 2)},
    # No datum (derivatives of derivatives, whose wall data are unknown): one
    # explicit term fewer, f' 4th and f'' 2nd order at the wall.
    "extrapolate": {"I": ([(2, 0), (1, 1), (2, 1)], 2), "II": ([(1, 0), (2, 1)], 2)},
}


def ccd_rows(kind: str):
    """First-cell CCD rows.

    Eq-I : f'_0  + sum c h^{d-1} f^(d)_k = [B datum + sum_k r_k f_k] / h
    Eq-II: f''_0 + sum c h^{d-2} f^(d)_k = [B datum + sum_k r_k f_k] / h^2
    """
    rows = {}
    for name, main in (("I", (1, centre(0))), ("II", (2, centre(0)))):
        implicit_terms, npts = STENCILS[kind][name]
        implicit = [(d, centre(k)) for d, k in implicit_terms]
        explicit = ([] if kind == "extrapolate" else [DATUM[kind]]) + [(0, centre(k)) for k in range(npts)]
        coeffs, order = closure(main, implicit + explicit)
        rows[name] = {"terms": implicit_terms, "implicit": coeffs[: len(implicit)],
                      "explicit": [-c for c in coeffs[len(implicit):]], "order": order - main[0]}
    return rows


#: Wall-face FCCD closures: terms (derivative order, cell) of the CCD jet of the
#: first cells (``h^d f^(d)``), after the wall datum.
FCCD_TERMS = {"gradient": ((0, 0), (1, 0), (0, 1), (1, 1)), "value": ((0, 0), (1, 0), (2, 0))}


def fccd_wall_rows():
    """Wall-face FCCD closures and the face-to-cell reconstruction.

    gradient (Dirichlet wall):  h u'(0) = c_w u_w + sum c h^d u^(d)_k   (u_0, h u'_0, u_1, h u'_1)
    value    (Neumann wall):    u(0)    = c_g h g + sum c h^d u^(d)_0   (u_0, h u'_0, h^2 u''_0)
    face_to_cell, first cell:   F(h/2)  = sum_{j<4} c_j F(j h)        (faces 0..3)
    face_to_cell, interior:     F(c_k)  = sum c_j F(faces k-1..k+2)

    The wall faces read the CCD jet of the cells next to the wall, as the
    interior faces read ``u`` and ``u''`` of their two cells: the same width,
    4th order like the interior. (The earlier explicit rows read 4 cells.)
    The face-to-cell rows span 4 faces, the width of the interior row.
    """
    g_terms = [(d, centre(k)) for d, k in FCCD_TERMS["gradient"]]
    v_terms = [(d, centre(k)) for d, k in FCCD_TERMS["value"]]
    grad, og = closure((1, 0), [(0, 0)] + g_terms)
    val, ov = closure((0, 0), [(1, 0)] + v_terms)
    first, of = closure((0, HALF), [(0, j) for j in range(4)])
    inner, oi = closure((0, HALF), [(0, j) for j in range(-1, 3)])
    return {"gradient": ([-c for c in grad], og - 1, FCCD_TERMS["gradient"]),
            "value": ([-c for c in val], ov, FCCD_TERMS["value"]),
            "face_to_cell_wall": ([-c for c in first], of, None), "face_to_cell": ([-c for c in inner], oi, None)}


def _fmt(values):
    return "(" + ", ".join(f"F({sp.numer(v)}, {sp.denom(v)})" for v in values) + ",)"


def main() -> None:
    lines = ['"""Generated by tools/derive_ccd_closures.py -- do not edit."""', "",
             "from fractions import Fraction as F", "", "CCD_WALL = {"]
    for kind in ("dirichlet", "neumann", "extrapolate"):
        rows = ccd_rows(kind)
        lines.append(f'    "{kind}": {{')
        for name, r in rows.items():
            lines.append(f'        "{name}": {{"terms": {tuple(tuple(t) for t in r["terms"])},')
            lines.append(f'               "implicit": {_fmt(r["implicit"])},')
            lines.append(f'               "explicit": {_fmt(r["explicit"])},')
            lines.append(f'               "order": {r["order"]}}},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    w = fccd_wall_rows()
    lines.append("FCCD_WALL = {")
    for name, (c, o, terms) in w.items():
        extra = "" if terms is None else f', "terms": {terms}'
        lines.append(f'    "{name}": {{"coeffs": {_fmt(c)}, "order": {o}{extra}}},')
    lines.append("}")
    out = Path(__file__).resolve().parents[1] / "basilisk_cupy" / "discretization" / "ccd_coefficients.py"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

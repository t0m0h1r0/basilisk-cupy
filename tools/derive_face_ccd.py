"""Derive the wall closures of the CCD on a face line and the Hermite midpoint weights (exact rationals).

Faces along one axis are a nodal line with the walls on its end nodes. The interior rows are
Chu & Fan's CCD; at a wall node, with ``f_k^(d)`` the d-th derivative at node k from the wall
(``h = 1``), the closure rows are

    I:  f'_0 + b0 f''_0 + a1 f'_1 + b1 f''_1 = c0 f_0 + c1 f_1 + c2 f_2      (exact to degree 5)
    II: f''_0 + a2 f''_1 + g1 f'_1 = e0 f_0 + e1 f_1 + e2 f_2                (exact to degree 4)

Both rows stay on nodes 0..2, the width of the interior stencil: the accuracy comes from
the implicit derivatives, not from a wider one-sided stencil. Counting relations on nodes
0..2: the interior rows are exact to degree 7 and span the degree-6 ones, row I is the only
further one exact to degree 5, and row II is exact to degree 4 (``f''`` 3rd order; through
the implicit ``f''_0`` of row I the wall ``f'`` is then 4th order). In ``R_h`` the wall
values enter as ``h f'`` and ``h^2 f''``, so ``R_h`` is 5th order at walls (6th inside). A row II through node 3 (``f''`` 4th order, ``R_h`` 6th) was measured to give
the same Navier--Stokes results (free-slip box Taylor--Green, N = 64: 6.8e-8 both), whose
floor is the 4th-order face operators, and was dropped.

The cell value is the quintic Hermite midpoint ``(f_L + f_R)/2 + 5h/32 (f'_L - f'_R) + h^2/64
(f''_L + f''_R)`` (error ``-h^6 f^(6)/46080``).

Run ``python tools/derive_face_ccd.py`` to print the coefficients used in
``basilisk_cupy/discretization/reconstruction.py``.
"""

import sympy as sp

t = sp.symbols("t")


def d(p, m, at):
    return (sp.diff(p, t, m) if m else p).subs(t, at)


def solve(row, unknowns, degrees):
    sol = sp.solve([sp.expand(row(t ** k)) for k in degrees], unknowns, dict=True)
    assert len(sol) == 1
    return sol[0]


def derive():
    b0, a1, b1, c0, c1, c2 = sp.symbols("b0 a1 b1 c0 c1 c2")
    row1 = lambda p: (d(p, 1, 0) + b0 * d(p, 2, 0) + a1 * d(p, 1, 1) + b1 * d(p, 2, 1)  # noqa: E731
                      - (c0 * d(p, 0, 0) + c1 * d(p, 0, 1) + c2 * d(p, 0, 2)))
    s1 = solve(row1, [b0, a1, b1, c0, c1, c2], range(6))
    a2, g1, e0, e1, e2 = sp.symbols("a2 g1 e0 e1 e2")
    row2 = lambda p: (d(p, 2, 0) + a2 * d(p, 2, 1) + g1 * d(p, 1, 1)  # noqa: E731
                      - (e0 * d(p, 0, 0) + e1 * d(p, 0, 1) + e2 * d(p, 0, 2)))
    s2 = solve(row2, [a2, g1, e0, e1, e2], range(5))
    w0, w1, w2 = sp.symbols("w0 w1 w2")
    mid = lambda p: (p.subs(t, sp.Rational(1, 2)) - (w0 * (d(p, 0, 0) + d(p, 0, 1))  # noqa: E731
                                                    + w1 * (d(p, 1, 0) - d(p, 1, 1)) + w2 * (d(p, 2, 0) + d(p, 2, 1))))
    s3 = solve(mid, [w0, w1, w2], range(6))
    return {
        "I": {"implicit": (s1[b0], s1[a1], s1[b1]), "explicit": (s1[c0], s1[c1], s1[c2])},
        "II": {"implicit": (s2[a2], s2[g1]), "explicit": (s2[e0], s2[e1], s2[e2])},
        "hermite": (s3[w0], s3[w1], s3[w2]),
    }


if __name__ == "__main__":
    for key, value in derive().items():
        print(key, value)

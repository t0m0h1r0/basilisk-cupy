"""Semi-discrete convection ``C(u) = div(u_f u)`` for method-of-lines integrators.

BCG (:mod:`.bcg`) is a space--time scheme tied to a single explicit step, so
multistep integrators such as IMEX-BDF2 need an operator evaluated at one
time level instead. This one is the conservative face-flux form

    C(u)_i = D_f( P_f(u_i) u_f )

with the face value ``P_f`` and divergence ``D_f`` of the shared
:class:`~basilisk_cupy.discretization.FaceComplex` (2nd order with fd2,
4th-order faces with FCCD, as TwoPhaseFlow's FCCD conservative flux,
SP-D Option B). It is non-dissipative; upwind variants (UCCD6/FUCCD) plug in
through the same ``tendency`` interface.
"""

from __future__ import annotations

from ...fields import FaceField, VectorField


class CentralFluxConvection:
    name = "central"

    def __init__(self, ops) -> None:
        self.ops = ops

    def tendency(self, u: VectorField, uf: FaceField) -> list:
        """``-C(u)`` for every component (interior arrays)."""
        grid = u.grid
        u.fill_ghosts()
        out = []
        for comp in u:
            flux = FaceField(grid, [self.ops.interp_f(comp, ax) * uf[ax] for ax in range(grid.ndim)])
            out.append(-self.ops.div_f(flux))
        return out

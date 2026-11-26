"""VOF with the sharp pressure jump (ghost-fluid method), docs/notes/64.

:class:`TwoPhaseVOFJump` is Basilisk's VOF two-phase flow (:class:`.two_phase.TwoPhase`: vof.h, two-phase.h,
tension.h's height-function curvature, reduced.h) with iforce.h's CSF replaced by the pressure jump of
:class:`.cls_two_phase.TwoPhaseCLS` (``pressure_jump=True``):

* cut faces: ``H = [f > 1/2]`` differs across the face;
* the crossing: from the line sum of the volume fraction over the three cells on each side of the face (the height
  function along the face's direction): with phase 1 on the lower side the interface lies at ``S - 2.5`` cell widths
  from the lower cell's centre (``3.5 - S`` with phase 1 on the upper side), ``theta`` clipped to [0, 1]. The
  linear interpolation of ``f`` used for the CLS would misplace it by O(h), as a VOF fraction jumps within a cell;
* GFM coefficient on cut faces ``alpha_f = 1/(theta rho_L + (1 - theta) rho_R)``, ``1/rho_k`` elsewhere;
* jump ``j = sigma kappa + [rho] G.(x - Z)`` at the crossing, from the cells on both sides that have it (the
  potentials are defined in interfacial cells only, NODATA elsewhere): theta-weighted when both do, else the one
  that does; face acceleration ``alpha_f j (H_R - H_L)/Delta``.
"""

from __future__ import annotations

from .fractions import reconstruction
from .heights import NODATA, pad_bc
from .two_phase import TwoPhase

WINDOW = 3  # cells on each side of a face for the crossing


class TwoPhaseVOFJump(TwoPhase):
    def __init__(self, ns, f, *, crossing: str = "sum", cell_coefficient: str = "mean", **kw) -> None:
        """``cell_coefficient``: the coefficient of the cell-centred velocity correction ``g`` on cut faces:
        ``"mean"`` (the face's net force ``a - alpha grad p`` rescaled to two-phase.h's ``1/rho(f_face)``, so the
        cells take it with the density of their own mixture; the static balance is kept since the net force
        vanishes there) or ``"gfm"`` (the GFM coefficient, as for the face velocity). With ``"gfm"`` the capillary
        wave's amplitude grows at a density ratio (docs/notes/65).

        ``crossing``: ``"sum"`` (the line sum of f, above) or ``"plic"`` (the PLIC line of the interfacial cell on
        either side: ``theta = alpha_L / n_ax`` in the lower cell's frame if it falls in [0, 1/2], else
        ``1 + alpha_R / n_ax`` from the upper cell if in [1/2, 1], else the line sum)."""
        if crossing not in ("sum", "plic"):
            raise ValueError("crossing is 'sum' or 'plic'")
        if cell_coefficient not in ("mean", "gfm"):
            raise ValueError("cell_coefficient is 'mean' or 'gfm'")
        self.cell_coefficient = cell_coefficient
        self.crossing = crossing
        self.coefficient = "gfm"  # "mean": two-phase.h's 1/rho(f_face) on cut faces too (diagnostics, notes/64)
        self.jump_weights = "theta"  # "mean": iforce.h's mean of the two cells' potentials (diagnostics)
        super().__init__(ns, f, **kw)

    def _faces(self, cell, ax, width: int = 1):
        """(left, right) values of every face along ``ax`` with ``width`` cells of padding (the face's two cells)."""
        cp = pad_bc(cell, self.f.bc, width)
        n = cell.shape
        lo = [slice(width, width + n[0]), slice(width, width + n[1])]
        lo[ax] = slice(width - 1, width + n[ax])
        hi = list(lo)
        hi[ax] = slice(width, width + n[ax] + 1)
        return cp[tuple(lo)], cp[tuple(hi)]

    def _cut(self, ax):
        xp = self.f.grid.xp
        f = xp.minimum(xp.maximum(self.f.interior, 0.0), 1.0)
        fl, fr = self._faces(f, ax)
        hl, hr = fl > 0.5, fr > 0.5
        cut = hl != hr
        # line sum of f over cells i-3 ... i+2 around face i (between cells i-1 and i)
        w = WINDOW
        fp = pad_bc(f, self.f.bc, w)
        n = f.shape
        S = 0.0
        for k in range(-w, w):
            sl = [slice(w, w + n[0]), slice(w, w + n[1])]
            sl[ax] = slice(w + k, w + k + n[ax] + 1)
            S = S + fp[tuple(sl)]
        theta = xp.where(hl, S - (w - 0.5), (w + 0.5) - S)
        if self.crossing == "plic":
            nx, ny, alpha = reconstruction(pad_bc(f, self.f.bc, 1))
            na = nx if ax == 0 else ny
            inter = (f > 0.0) & (f < 1.0) & (xp.abs(na) > 1e-12)
            t = xp.where(inter, alpha / xp.where(inter, na, 1.0), 2.0)  # crossing in the cell's own frame
            tl, tr = self._faces(t, ax)
            tr = 1.0 + tr
            okl = (tl >= 0.0) & (tl <= 0.5)
            okr = (tr >= 0.5) & (tr <= 1.0)
            theta = xp.where(okl & okr, 0.5 * (tl + tr), xp.where(okl, tl, xp.where(okr, tr, theta)))
        theta = xp.where(cut, xp.minimum(xp.maximum(theta, 0.0), 1.0), 0.0)
        return hl, hr, cut, theta

    def _properties(self, ns) -> None:
        super()._properties(ns)
        alpha_mean = [ns.alpha[ax].copy() for ax in range(2)]  # two-phase.h's 1/rho(f_face)
        ns.g_scale = None
        if self.coefficient == "mean":
            return
        xp = self.f.grid.xp
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            rl = xp.where(hl, self.rho1, self.rho2)
            rr = xp.where(hr, self.rho1, self.rho2)
            ns.alpha[ax] = 1.0 / (theta * rl + (1.0 - theta) * rr)
        if self.cell_coefficient == "mean":
            ns.g_scale = [alpha_mean[ax] / ns.alpha[ax] for ax in range(2)]

    def _interfacial_force(self, ns) -> None:
        if not self.potentials:
            return
        xp = self.f.grid.xp
        phi = None
        for pot in self.potentials:
            v = pot(self)
            phi = v if phi is None else xp.where((phi == NODATA) | (v == NODATA),
                                                 xp.where(phi == NODATA, v, phi), phi + v)
        ns.a_is_constant = False
        h = [float(v) for v in self.f.grid.h]
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            pl, pr = self._faces(phi, ax)
            okl, okr = pl < NODATA, pr < NODATA
            w = theta if self.jump_weights == "theta" else 0.5
            j = xp.where(okl & okr, (1.0 - w) * pl + w * pr, xp.where(okl, pl, xp.where(okr, pr, 0.0)))
            sign = hr.astype(phi.dtype) - hl.astype(phi.dtype)
            ns.a[ax] = ns.a[ax] + xp.where(cut, ns.alpha[ax] * j * sign / h[ax], 0.0)

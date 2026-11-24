"""Curvature from the CLS field ``psi`` (TwoPhaseFlow paper section 3c, eq. ``curvature_psi_2d``).

The paper's standard path differentiates ``psi`` directly (curvature is invariant under the
monotone map ``psi = H_eps(phi)``, theorem ``curvature_invariance``)::

    kappa = -(psi_y^2 psi_xx - 2 psi_x psi_y psi_xy + psi_x^2 psi_yy) / (psi_x^2 + psi_y^2)^(3/2)

with ``psi_x, psi_xx`` from one CCD solve per axis and ``psi_xy`` from the CCD first derivative
of ``psi_x`` along ``y`` (paper section 11b1, stage 4), in the band
``psi_min < psi < 1 - psi_min`` (``psi_min = 0.01``) and ``kappa = 0`` outside. ``kappa > 0``
for a liquid (``psi = 1``) drop, as Basilisk's ``curvature()`` of ``f = 1`` inside.

``source="phi"`` evaluates the same formula on ``phi = H_eps^{-1}(psi)`` (the paper's
"uncorrected phi curvature", for comparison). ``code_variant=True`` adds what TwoPhaseFlow's
``CurvatureCalculatorPsi`` does and the paper does not (its standard path excludes post filters):
the DCCD filter ``g_i + eps_d s_i (g_{i+1} - 2 g_i + g_{i-1})``, ``eps_d = 0.05``,
``s = (2 psi - 1)^2``, on every derivative along every axis, the wall cells left unfiltered.
"""

from __future__ import annotations

from ..discretization.ccd import CCDCellDerivative
from ..fields import CellField
from .cls import phi_from_psi

PSI_MIN = 0.01
DCCD_EPS = 0.05


def _dccd(xp, g, coeff, periodic):
    out = g
    for ax in range(g.ndim):
        if periodic[ax]:
            lap = xp.roll(out, -1, ax) - 2.0 * out + xp.roll(out, 1, ax)
            out = out + coeff * lap
        else:
            c = [slice(None)] * g.ndim
            m = list(c)
            p = list(c)
            c[ax], m[ax], p[ax] = slice(1, -1), slice(0, -2), slice(2, None)
            new = out.copy()
            new[tuple(c)] = out[tuple(c)] + coeff[tuple(c)] * (out[tuple(p)] - 2.0 * out[tuple(c)] + out[tuple(m)])
            out = new
    return out


def psi_curvature(psi: CellField, *, eps: float | None = None, ccd: CCDCellDerivative | None = None,
                  psi_min: float = PSI_MIN, source: str = "psi", code_variant: bool = False):
    """Cell-centred curvature (interior array), zero outside the band ``psi_min < psi < 1 - psi_min``."""
    grid = psi.grid
    if grid.ndim != 2:
        raise NotImplementedError("2-D only")
    xp = grid.xp
    ccd = ccd or CCDCellDerivative()
    q = psi.interior
    periodic = [psi.bc.is_periodic(ax) for ax in range(2)]
    if source == "psi":
        psi.fill_ghosts()
        (gx, gxx), (gy, gyy) = ccd.derivatives(psi, 0), ccd.derivatives(psi, 1)
    elif source == "phi":
        if eps is None:
            raise ValueError("source='phi' needs eps")
        f = phi_from_psi(q, eps)
        gx, gxx = ccd.derivatives_array(f, grid, 0, periodic[0])
        gy, gyy = ccd.derivatives_array(f, grid, 1, periodic[1])
    else:
        raise ValueError("source is 'psi' or 'phi'")
    filt = (lambda g: _dccd(xp, g, DCCD_EPS * (2.0 * q - 1.0) ** 2, periodic)) if code_variant else (lambda g: g)
    gx, gxx, gy, gyy = (filt(g) for g in (gx, gxx, gy, gyy))
    gxy = filt(ccd.d1_array(gx, grid, 1, periodic[1]))
    num = gy * gy * gxx - 2.0 * gx * gy * gxy + gx * gx * gyy
    kappa = -num / (gx * gx + gy * gy + 1e-30) ** 1.5
    band = (q > psi_min) & (q < 1.0 - psi_min)
    return xp.where(band, kappa, 0.0)

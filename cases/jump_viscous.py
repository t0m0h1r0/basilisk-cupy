"""Jump-corrected viscous operator, IIM form (docs/notes/40) -- tried and not adopted.

For equal viscosities the tangential velocity has ``[d_nn u_t] = d_t j / mu`` at the interface (``j`` the GFM
jump), the vorticity cusp of docs/notes/38. With the kink written as ``J = H J+``,
``J+ = perp grad Psi+``, ``Psi+ = -A chi d^3 / 6``, ``A = tau . grad j / mu``, ``chi = exp(-(d/w)^2)`` (``J`` is
divergence-free and its tangential part is ``A d^2/2``), the corrected operator ``L_h(u - J) + L(J)`` adds
``S = mu (H L(J+) - L(H J+))`` to the implicit viscous step: :func:`attach` puts ``dt S / rho`` into its
right-hand side. ``kink="face"`` puts the kink on the face where the GFM jump sits (graph in y only).

Measured (docs/notes/40): it improves the frozen-interface Stokes layer at delta/h = 1.83 and worsens it at
0.92; with the moving interface (capillary wave) it is unstable at N = 32 and 64 and at dt / 16.
"""

from __future__ import annotations

import numpy as np

from basilisk_cupy.discretization.ccd import CCDCellDerivative
from basilisk_cupy.interface.cls import phi_from_psi
from basilisk_cupy.terms.viscous import CCDStress, FD2Stress


def correction(ns, tp, stress, width: float = 2.0, parts: bool = False, form: str = "stream",
               kink: str = "interface"):
    """``S = mu (H L(J+) - L(H J+))``, ``J+ = (tau . grad j / mu) d^2/2 exp(-(d/w)^2) tau`` (cell arrays)."""
    grid = ns.grid
    cls = tp.cls
    psi = cls.psi
    psi.fill_ghosts()
    q = psi.interior
    ccd = CCDCellDerivative()
    gx, gy = ccd.d1(psi, 0), ccd.d1(psi, 1)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-30
    nx, ny = gx / gn, gy / gn
    tx, ty = -ny, nx
    d = phi_from_psi(q, cls.eps)
    if kink == "face":  # the kink where the GFM puts the jump: the face between the cells of different H (graph in y)
        yc_ = np.asarray(grid.centers(1))
        fy = np.asarray(grid.faces(1))
        Hc = q > 0.5
        k = np.argmax(Hc, axis=1)  # first liquid cell from below in each column
        d = yc_[None, :] - fy[k][:, None]
    phi = None
    for pot in tp.potentials:
        v = pot(tp)
        phi = v if phi is None else phi + v
    xc, yc = (np.asarray(grid.centers(a)) for a in range(2))
    jx = np.gradient(phi, xc, axis=0)
    jy = np.gradient(phi, yc, axis=1)
    mu = tp.mu1
    A = (tx * jx + ty * jy) / mu
    h = min(float(np.min(np.diff(np.asarray(grid.faces(a))))) for a in range(2))
    w = width * h
    chi = np.exp(-(d / w) ** 2)
    Q = 0.5 * d * d * chi
    H = (q > 0.5).astype(float)
    if form == "stream":  # divergence-free: J+ = perp grad Psi+, Psi+ = -A chi d^3 / 6 (tangential part -Psi_d tau)
        Psi = -A * chi * d ** 3 / 6.0
        per = [psi.bc.is_periodic(a) for a in range(2)]
        Jp = [ccd.d1_array(Psi, grid, 1, per[1]), -ccd.d1_array(Psi, grid, 0, per[0])]
    else:
        Jp = [A * Q * tx, A * Q * ty]
    tmp = ns.u.copy("jump_tmp")

    def L(vals):
        for c, v in zip(tmp, vals):
            c.interior = v
        tmp.fill_ghosts()
        return stress(tmp, 1.0) if not isinstance(stress, FD2Stress) else stress(tmp, 1.0)

    LJ = L(Jp)
    LHJ = L([H * c for c in Jp])
    S = [mu * (H * a - b) for a, b in zip(LJ, LHJ)]
    if parts:
        return S, {"A": A, "Q": Q, "H": H, "d": d}
    return S


def attach(ns, tp, stress=None, width: float = 2.0, form: str = "stream", scale: float = 1.0,
           kink: str = "interface"):
    """Add ``dt alpha S`` to the right-hand side of the implicit viscous step (hook ``viscous_term``)."""
    stress = stress or FD2Stress()
    log = []

    def hook(_ns):
        S = correction(_ns, tp, stress, width, form=form, kink=kink)
        dt = _ns.clock.dt
        rho = _ns.rho if _ns.rho is not None else 1.0
        for c, s in zip(_ns.u, S):
            c.interior = c.interior + scale * dt / rho * s
        _ns.u.fill_ghosts()
        log.append(max(float(np.max(np.abs(s))) for s in S))

    ns.on("viscous_term", hook)
    return log

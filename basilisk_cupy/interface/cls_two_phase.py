"""Two-phase flow with the CLS interface (TwoPhaseFlow's ``psi``) on Basilisk's centred solver, 2-D.

Same coupling as :class:`~.two_phase.TwoPhase` (Basilisk ``two-phase.h`` + ``iforce.h``) with the
volume fraction replaced by ``psi``, so that only the interface representation differs from the
VOF baseline:

* transport: :meth:`.CLSInterface.advect` with the projected face velocity ``uf`` (stages A, B),
  optionally re-initialised every ``reinit_every`` steps;
* properties: ``rho = rho2 + (rho1 - rho2) psi`` at cells, ``alpha = 1/rho(psi_face)`` and
  ``mu(psi_face)`` with arithmetic face averages (paper 06d: arithmetic averages are the reference
  for the diffuse CLS interface; Basilisk's ``two-phase.h`` does the same with ``f``);
* surface tension: the CSF force ``sigma kappa grad psi`` (paper 02b, 09: balanced force) in
  ``iforce.h``'s face form ``alpha_f sigma kappa_f (psi_R - psi_L)/Delta``, ``kappa`` from
  :meth:`.CLSInterface.curvature` (zero outside the band ``0.01 < psi < 0.99``),
  ``kappa_f`` the face average; tension's time-step bound as in ``tension.h``;
* reduced gravity: ``phi = [rho] G . (x - Z)`` at every cell centre (``grad psi`` localises it;
  ``reduced.h`` evaluates it at the height-function position of the interface cells).

``pressure_jump=True`` replaces the last two by the paper's pressure-jump closure (09b1, 09b3) in
finite-volume form, the ghost-fluid method (Liu, Fedkiw & Kang 2000). The interface is the
``psi = 1/2`` crossing on each face, at the fraction ``theta = (psi_L - 1/2)/(psi_L - psi_R)`` of
the face's centre-to-centre segment:

* jump-corrected face gradient ``G_Gamma(p; j) = G p - B(j)`` with ``B(j) = j (H_R - H_L)/Delta``,
  ``H = [psi > 1/2]``, ``j = sigma kappa + [rho] G . (x - Z)`` interpolated to the crossing; it
  enters Basilisk's projection as the face acceleration ``alpha_f B(j)`` (the same face form as
  ``iforce.h`` with the sharp ``H`` in place of ``f``), so the projection, the velocity
  correction and the centred gradient all use ``G_Gamma``;
* phase-wise constant densities (split PPE): ``alpha_f = 1/rho_k`` on faces inside phase ``k``
  and, on cut faces, the GFM coefficient of ``[alpha d_n p] = 0``,
  ``alpha_f = 1/(theta rho_L + (1 - theta) rho_R)``. One multigrid solve of this operator is the
  two constant-density phase problems coupled by the jump conditions; Basilisk's compact
  two-point face stencils never cross the interface with an unknown jump, so HFE (needed by the
  paper's CCD stencils) has no role here.

Cell densities (momentum, viscosity) and face viscosities stay those of ``psi``.

Capillary time step: tension.h's ``sqrt(rho_m Delta^3 / (pi sigma))`` assumes a curvature whose
response to a grid-scale wiggle is that of 2nd-order differences, ``4/Delta^2``. CCD's second
derivative there is ``48/(5 Delta^2)`` (Nyquist symbol of the Chu--Fan scheme), so the explicit
capillary loop is stiffer by 12/5: the bound is multiplied by ``capillary_dt_factor``, by default
``sqrt(5/12)`` (:data:`CCD_CAPILLARY_DT_FACTOR`); ``1`` is tension.h's bound (unstable at density
ratios 100 and 1000 with equal viscosities, N = 64; docs/notes/19).
"""

from __future__ import annotations

import math

from ..fields import CellField, FaceField
from .cls import CLSInterface
from .two_phase import TwoPhase

#: sqrt(4 / (48/5)): tension.h's bound for a CCD curvature (Nyquist symbol 48/5 against 4).
CCD_CAPILLARY_DT_FACTOR = math.sqrt(5.0 / 12.0)


def _pad_like(xp, q, bc):
    from .heights import pad_bc  # noqa: PLC0415

    return pad_bc(q, bc, 1)


def _cg_face(xp, rhs_u, beta, alpha, ax, per, h, tol=1e-10, maxit=500):
    """Solve ``u / alpha - div(beta grad u) = rhs_u / alpha`` on a face grid by conjugate gradients.

    Along ``ax`` the unknowns are the interior faces (zero beyond: wall faces) unless periodic; along
    the other axis the boundary is a mirror (zero normal derivative) unless periodic. ``beta`` at an
    edge between two unknowns is their average."""
    t = 1 - ax

    def edge(b, axis, periodic):
        nb = xp.roll(b, -1, axis)
        e = 0.5 * (b + nb)
        if not periodic:
            idx = [slice(None)] * 2
            idx[axis] = -1
            e[tuple(idx)] = 0.0  # no edge beyond the last unknown (handled by the boundary terms)
        return e

    ea = edge(beta, ax, per[ax])
    et = edge(beta, t, per[t])
    op = _face_operator(xp, beta, alpha, ax, per, h, ea, et)
    return _cg(xp, op, rhs_u, alpha, tol, maxit)


def _face_operator(xp, beta, alpha, ax, per, h, ea, et):
    t = 1 - ax

    def op(u):
        out = u / alpha
        for axis, e, periodic, hh in ((ax, ea, per[ax], h[ax]), (t, et, per[t], h[t])):
            un = xp.roll(u, -1, axis)
            flux = e * (un - u) / (hh * hh)  # flux to the +neighbour
            if not periodic:
                idx = [slice(None)] * 2
                idx[axis] = -1
                flux[tuple(idx)] = 0.0
            out = out - flux + xp.roll(flux, 1, axis) * _mask_first(xp, u.shape, axis, periodic)
            if axis == ax and not periodic:  # Dirichlet 0 beyond both ends: beta at the end unknowns
                b0 = [slice(None)] * 2
                b0[axis] = slice(0, 1)
                b1 = [slice(None)] * 2
                b1[axis] = slice(-1, None)
                out[tuple(b0)] = out[tuple(b0)] + beta[tuple(b0)] * u[tuple(b0)] / (hh * hh)
                out[tuple(b1)] = out[tuple(b1)] + beta[tuple(b1)] * u[tuple(b1)] / (hh * hh)
        return out

    return op


def _cg(xp, op, rhs_u, alpha, tol, maxit):
    b = rhs_u / alpha
    x = rhs_u.copy()
    r = b - op(x)
    pvec = r.copy()
    rr = float(xp.sum(r * r))
    bnorm = float(xp.sqrt(xp.sum(b * b))) or 1.0
    it = 0
    while rr ** 0.5 > tol * bnorm and it < maxit:
        Ap = op(pvec)
        a = rr / float(xp.sum(pvec * Ap))
        x = x + a * pvec
        r = r - a * Ap
        rr_new = float(xp.sum(r * r))
        pvec = r + (rr_new / rr) * pvec
        rr = rr_new
        it += 1
    return x, it


def _mask_first(xp, shape, axis, periodic):
    """1 everywhere except the first row along ``axis`` when not periodic (no flux from beyond)."""
    if periodic:
        return 1.0
    m = xp.ones(shape)
    idx = [slice(None)] * 2
    idx[axis] = 0
    m[tuple(idx)] = 0.0
    return m


class TwoPhaseCLS(TwoPhase):
    def __init__(self, ns, cls: CLSInterface, *, rho1: float = 1.0, rho2: float = 1.0, mu1: float = 0.0,
                 mu2: float = 0.0, vof_first: bool = False, reinit_every: int = 0,
                 reinit_method: str = "xi_sdf", curvature_kw: dict | None = None,
                 pressure_jump: bool = False, capillary_dt_factor: float | None = None,
                 surface_viscosity: float = 0.0, implicit_tension: float = 0.0, jump_mode: str = "sharp",
                 jump_interp: str = "linear", cell_coefficient: str = "gfm",
                 cell_density: str = "sharp") -> None:
        """``cell_coefficient`` (with ``pressure_jump``): the coefficient of the cell-centred velocity correction on
        the cut faces, ``"gfm"`` (the face's), ``"mean"`` (the net face force rescaled to ``1/rho(psi_face)`` on
        every face, :class:`~.vof_jump.TwoPhaseVOFJump`) or ``"cut"`` (rescaled on the cut faces only),
        docs/notes/65. ``cell_density``: the cell density of the implicit viscous solve, ``"sharp"`` (the default
        since docs/notes/69: ``rho1`` where ``psi > 1/2``, else ``rho2``) or ``"diffuse"`` (two-phase.h's
        ``rho(psi)``, which damps the oscillating drop about four times too much, docs/notes/68)."""
        f: CellField = cls.psi
        if f.grid.ndim != 2:
            raise NotImplementedError("two-phase flows are implemented in 2-D")
        self.ns, self.f, self.cls = ns, f, cls
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2
        self.filtered = False
        self.reinit_every, self.reinit_method = reinit_every, reinit_method
        self.curvature_kw = curvature_kw or {}
        self.potentials = []
        self.pressure_jump = pressure_jump
        self.surface_viscosity = float(surface_viscosity)
        self.implicit_tension = float(implicit_tension)
        if not f.grid.is_uniform and (jump_mode != "sharp" or surface_viscosity or implicit_tension):
            raise NotImplementedError("on a mapped grid TwoPhaseCLS has the sharp jump only (no surface "
                                      "viscosity, no implicit tension)")
        if jump_mode not in ("sharp", "smooth"):
            raise ValueError("jump_mode is 'sharp' (the pressure jumps) or 'smooth' (p - H j_ext is solved for)")
        self.jump_mode = jump_mode
        #: the jump at a face's crossing: ``"linear"`` (the two cell values, weights by theta) or ``"hermite"``
        #: (quintic Hermite with the CCD jet along the line, :func:`.phase_ccd.hermite_at_crossings`; the linear
        #: interpolation error h^2 j'' grows with the density ratio and limits a high-order projection, note 55)
        if jump_interp not in ("linear", "hermite"):
            raise ValueError("jump_interp is 'linear' or 'hermite'")
        self.jump_interp = jump_interp
        if cell_coefficient not in ("gfm", "mean", "cut"):
            raise ValueError("cell_coefficient is 'gfm', 'mean' or 'cut'")
        self.cell_coefficient = cell_coefficient
        if cell_density not in ("diffuse", "sharp"):
            raise ValueError("cell_density is 'diffuse' or 'sharp'")
        self.cell_density = cell_density
        self.j_ext = None
        self.cg_stats = []
        self.capillary_dt_factor = CCD_CAPILLARY_DT_FACTOR if capillary_dt_factor is None else float(capillary_dt_factor)
        self._steps = 0
        ns.cfl = min(ns.cfl, 0.5)
        ns.on("set_dtmax" if vof_first else "vof", self._vof)
        ns.on("properties", self._properties)
        ns.on("acceleration", self._interfacial_force)
        ns.on("end_timestep", self._implicit_capillary)
        if mu1 or mu2 or surface_viscosity:
            ns.mu = FaceField(f.grid, name="mu")

    def _vof(self, ns) -> None:
        dt = ns.clock.dt
        if not dt:
            return
        self.cls.advect(ns.uf, dt)
        self._steps += 1
        if self.reinit_every and self._steps % self.reinit_every == 0:
            self.cls.reinitialize(method=self.reinit_method)

    def add_surface_tension(self, sigma: float) -> None:
        self.sigma = float(sigma)
        self.potentials.append(lambda tp: tp.sigma * tp.cls.curvature(**tp.curvature_kw))
        self.ns.on("stability", self._capillary_dtmax)

    def _capillary_dtmax(self, ns) -> None:
        saved = ns.dtmax
        ns.dtmax = float("inf")
        super()._capillary_dtmax(ns)
        dt = ns.dtmax * self.capillary_dt_factor
        ns.dtmax = min(saved, dt)

    def add_reduced_gravity(self, G=(0.0, 0.0), Z=(0.0, 0.0)) -> None:
        d = self.rho2 - self.rho1
        grid = self.f.grid

        def pot(tp):
            x, y = grid.meshgrid()
            return d * (G[0] * (x - Z[0]) + G[1] * (y - Z[1]))

        self.potentials.append(pot)

    # -- pressure-jump closure (GFM) -------------------------------------------------------------
    def _faces(self, cell, ax):
        """(left, right) cell values of every face along ``ax`` (ghosts from psi's conditions)."""
        from .heights import pad_bc  # noqa: PLC0415

        cp = pad_bc(cell, self.f.bc, 1)
        n = cell.shape
        lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
        lo[ax] = slice(0, n[ax] + 1)
        hi = list(lo)
        hi[ax] = slice(1, n[ax] + 2)
        return cp[tuple(lo)], cp[tuple(hi)]

    def _jump_at_crossings(self, phi, ax, theta):
        """The jump at the crossing of every face along ``ax`` (cell values ``phi``)."""
        if self.jump_interp == "hermite":
            from ..discretization.phase_ccd import hermite_at_crossings  # noqa: PLC0415
            from .cls_curvature import PSI_MIN  # noqa: PLC0415

            q = self.f.interior
            band = (q > PSI_MIN) & (q < 1.0 - PSI_MIN)
            return hermite_at_crossings(phi, theta, ax, float(self.f.grid.h[ax]), self.f.bc.is_periodic(ax), band)
        jl, jr = self._faces(phi, ax)
        return (1.0 - theta) * jl + theta * jr

    def _cut(self, ax):
        xp = self.f.grid.xp
        pl, pr = self._faces(self.f.interior, ax)
        hl, hr = pl > 0.5, pr > 0.5
        cut = hl != hr
        theta = xp.where(cut, (pl - 0.5) / xp.where(cut, pl - pr, 1.0), 0.0)
        return hl, hr, cut, theta

    def _properties(self, ns) -> None:
        super()._properties(ns)
        if self.cell_density == "sharp":
            q = self.f.interior
            ns.rho = self.f.grid.xp.where(q > 0.5, self.rho1, self.rho2)
        if self.surface_viscosity and getattr(self, "sigma", 0.0):
            self._add_surface_viscosity(ns)
        if not self.pressure_jump:
            return
        xp = self.f.grid.xp
        alpha_mean = [ns.alpha[ax].copy() for ax in range(2)] if self.cell_coefficient != "gfm" else None
        scale = []
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            rl = xp.where(hl, self.rho1, self.rho2)
            rr = xp.where(hr, self.rho1, self.rho2)
            ns.alpha[ax] = 1.0 / (theta * rl + (1.0 - theta) * rr)
            if alpha_mean is not None:
                r = alpha_mean[ax] / ns.alpha[ax]
                scale.append(xp.where(cut, r, 1.0) if self.cell_coefficient == "cut" else r)
        ns.g_scale = scale or None

    def _add_surface_viscosity(self, ns) -> None:
        """Surface viscosity ``c dt sigma |grad psi|_f`` added to the implicit viscous solve (the term
        ``dt sigma Delta_s u`` of Hysing 2006 in the predictor). Measured: it does not stabilise the
        capillary loop here, the jump acts after the viscous step (docs/notes/27); kept for comparison."""
        xp = self.f.grid.xp
        q = self.f.interior
        h = [float(v) for v in self.f.grid.h]
        n = q.shape
        coeff = self.surface_viscosity * float(ns.clock.dt) * self.sigma
        if not (self.mu1 or self.mu2):  # two-phase.h leaves mu untouched then: start from zero every step
            for ax in range(2):
                ns.mu[ax] = 0.0 * ns.mu[ax]
        for ax in range(2):
            pl, pr = self._faces(q, ax)
            gn = (pr - pl) / h[ax]
            t = 1 - ax
            qp = _pad_like(xp, q, self.f.bc)
            ct = (xp.roll(qp, -1, t) - xp.roll(qp, 1, t)) / (2.0 * h[t])
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            gt = 0.5 * (ct[tuple(lo)] + ct[tuple(hi)])
            ns.mu[ax] = ns.mu[ax] + coeff * xp.sqrt(gn * gn + gt * gt)

    def _implicit_capillary(self, ns) -> None:
        """Semi-implicit surface tension on the transport velocity (after Hysing 2006, isotropic form).

        After the projection, every component ``u`` of ``uf`` (the velocity that moves ``psi`` at the
        next step) solves

            u / alpha_f - div(beta grad u) = u* / alpha_f,   beta = theta dt^2 sigma |grad psi|

        (``theta = implicit_tension``; the momentum form ``rho (u - u*) = dt^2 sigma Delta_s u delta`` with
        the full gradient for the surface one) on the grid of that component's faces (wall faces stay
        zero, the other walls are mirrors, periodic axes wrap; conjugate gradients, relative residual
        1e-10), and is projected again with a pressure of its own. The cell velocity is not changed.
        Applying the same operator before the projection, to ``uf`` with the jump acceleration in it,
        breaks the balance of the jump and the pressure gradient (parasitic velocities of 0.1, and no
        stabilisation at density ratio 100; docs/notes/27)."""
        if not self.implicit_tension or not getattr(self, "sigma", 0.0):
            return
        xp = self.f.grid.xp
        q = self.f.interior
        h = [float(v) for v in self.f.grid.h]
        n = q.shape
        dt = float(ns.clock.dt)
        theta = self.implicit_tension
        per = [self.f.bc.is_periodic(ax) for ax in range(2)]
        for ax in range(2):
            # |grad psi| on the faces normal to ax
            pl, pr = self._faces(q, ax)
            gn = (pr - pl) / h[ax]
            t = 1 - ax
            qp = _pad_like(xp, q, self.f.bc)
            ct = (xp.roll(qp, -1, t) - xp.roll(qp, 1, t)) / (2.0 * h[t])
            lo = [slice(1, 1 + n[0]), slice(1, 1 + n[1])]
            lo[ax] = slice(0, n[ax] + 1)
            hi = list(lo)
            hi[ax] = slice(1, n[ax] + 2)
            delta = xp.sqrt(gn * gn + (0.5 * (ct[tuple(lo)] + ct[tuple(hi)])) ** 2)
            beta = theta * dt * dt * self.sigma * delta
            u0 = ns.uf[ax]
            alpha = ns.alpha[ax]
            if per[ax]:
                sl = [slice(None)] * 2
                sl[ax] = slice(0, n[ax])
                U, B, A = u0[tuple(sl)], beta[tuple(sl)], alpha[tuple(sl)]
            else:
                sl = [slice(None)] * 2
                sl[ax] = slice(1, n[ax])
                U, B, A = u0[tuple(sl)], beta[tuple(sl)], alpha[tuple(sl)]
            sol, it = _cg_face(xp, U, B, A, ax, per, h)
            self.cg_stats.append(it)
            new = u0.copy()
            new[tuple(sl)] = sol
            if per[ax]:
                last = [slice(None)] * 2
                last[ax] = slice(n[ax], n[ax] + 1)
                first = [slice(None)] * 2
                first[ax] = slice(0, 1)
                new[tuple(last)] = new[tuple(first)]
            ns.uf[ax] = new
        # the smoothed transport velocity is projected again (divergence-free for the psi transport)
        from ..fields import CellField  # noqa: PLC0415
        from ..projection.approximate import project  # noqa: PLC0415

        if not hasattr(self, "_p_si"):
            self._p_si = CellField(self.f.grid, bc=ns.p.bc, name="p_si")
        project(ns.uf, self._p_si, ns.alpha, dt, ops=ns.ops, tolerance=ns.tolerance, basilisk_tolerance=True)
        ns._apply_uf_bc(ns.uf)

    def _interfacial_force(self, ns) -> None:
        if not self.pressure_jump:
            return super()._interfacial_force(ns)
        if not self.potentials:
            return
        xp = self.f.grid.xp
        phi = None
        for pot in self.potentials:
            v = pot(self)
            phi = v if phi is None else phi + v
        ns.a_is_constant = False
        if self.jump_mode == "smooth":
            self._smooth_jump_force(ns, phi)
            return
        from .cls import centre_spacing  # noqa: PLC0415

        for ax in range(2):
            h = centre_spacing(self.f.grid, ax)  # the fd2 face gradient divides by the centre distance
            hl, hr, cut, theta = self._cut(ax)
            j = self._jump_at_crossings(phi, ax, theta)
            sign = hr.astype(phi.dtype) - hl.astype(phi.dtype)
            ns.a[ax] = ns.a[ax] + xp.where(cut, ns.alpha[ax] * j * sign / h, 0.0)

    # -- smooth-pressure form of the jump (paper 09b1: G_Gamma(p; j) = G p - B(j)) ----------------------
    def _crossings(self, phi):
        """Crossing points of every cut face (coordinates) and the jump there (theta-interpolated)."""
        xp = self.f.grid.xp
        grid = self.f.grid
        h = [float(v) for v in grid.h]
        o = [float(v) for v in grid.origin]
        pts, vals = [], []
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            j = self._jump_at_crossings(phi, ax, theta)
            idx = xp.nonzero(cut)
            i, k = idx
            if ax == 0:
                cx = o[0] + (i - 0.5 + theta[cut]) * h[0]
                cy = o[1] + (k + 0.5) * h[1]
            else:
                cx = o[0] + (i + 0.5) * h[0]
                cy = o[1] + (k - 0.5 + theta[cut]) * h[1]
            pts.append(xp.stack([cx, cy], axis=1))
            vals.append(j[cut])
        return xp.concatenate(pts), xp.concatenate(vals)

    def extend_jump(self, phi, chunk: int = 4096):
        """``j_ext``: the jump at the nearest crossing point (closest-point extension, constant along the
        normal), at every cell centre (minimum image on periodic axes)."""
        xp = self.f.grid.xp
        grid = self.f.grid
        pts, vals = self._crossings(phi)
        if pts.shape[0] == 0:
            return xp.zeros(grid.n)
        x, y = grid.meshgrid()
        X = xp.stack([x.ravel(), y.ravel()], axis=1)
        L = [float(v) for v in grid.length]
        per = [self.f.bc.is_periodic(ax) for ax in range(2)]
        out = xp.empty(X.shape[0])
        for s0 in range(0, X.shape[0], chunk):
            d = X[s0:s0 + chunk, None, :] - pts[None, :, :]
            for ax in range(2):
                if per[ax]:
                    d[..., ax] = d[..., ax] - L[ax] * xp.round(d[..., ax] / L[ax])
            k = xp.argmin((d * d).sum(axis=2), axis=1)
            out[s0:s0 + chunk] = vals[k]
        return out.reshape(grid.n)

    def _smooth_jump_force(self, ns, phi) -> None:
        """Face acceleration ``alpha_f (B(j) - D_f(H j_ext))``: the projection then solves for the
        pressure without the jump, ``p~ = p - H j_ext`` (continuous, with a continuous normal derivative
        for equal densities), so that FCCD's face gradient never crosses the jump. ``B(j) = j_Gamma
        (H_R - H_L)/Delta`` on cut faces as in the sharp form; ``D_f(H j_ext) = (H_R j_R - H_L j_L)/Delta``."""
        xp = self.f.grid.xp
        h = float(self.f.grid.h[0])
        jext = self.extend_jump(phi)
        self.j_ext = jext
        for ax in range(2):
            hl, hr, cut, theta = self._cut(ax)
            jg = self._jump_at_crossings(phi, ax, theta)
            el, er = self._faces(jext, ax)
            Hl, Hr = hl.astype(jext.dtype), hr.astype(jext.dtype)
            b = xp.where(cut, jg * (Hr - Hl), 0.0)
            d = Hr * er - Hl * el
            ns.a[ax] = ns.a[ax] + ns.alpha[ax] * (b - d) / h

    def full_pressure(self, ns):
        """The pressure with its jump, ``p~ + H j_ext`` (smooth mode), or ``ns.p`` itself (sharp mode)."""
        if self.jump_mode != "smooth" or self.j_ext is None:
            return ns.p.interior
        return ns.p.interior + (self.f.interior > 0.5) * self.j_ext

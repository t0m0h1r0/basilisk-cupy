"""Conservative level set (CLS) interface, TwoPhaseFlow's method (paper ch. 3, 5; ``src/twophase/levelset``).

The interface indicator is ``psi = H_eps(-phi) = (1 - tanh(phi / 2 eps)) / 2`` (``psi = 1`` in
the liquid, ``phi`` the signed distance, negative in the liquid; here level sets are
positive inside, ``psi = H_eps(Phi)``), ``eps = C_eps Delta`` with ``C_eps = 1.5`` (TwoPhaseFlow's
``epsilon_factor``). Per physical step (paper section 5b, stages A, B):

* **A, transport**: ``d_t psi + div(psi u_f) = 0`` in conservative form, TVD-RK3 (Shu--Osher),
  face flux ``(P_f psi) u_f`` with FCCD's face value ``P_f`` (4th order, central; TwoPhaseFlow's
  ``fccd_flux`` mode), the plain face difference ``D_f`` and ``psi`` clipped to ``[0, 1]`` after
  every stage (TwoPhaseFlow's ``clip_bounds``). ``u_f`` must be divergence-free for ``D_f``: with
  FCCD operators pass ``ops.wall_free_flux(uf, periodic)``.
  ``divergence="fccd"`` corrects the flux by ``-(h^2/24)(psi_xx u + 2 psi_x u_x)`` (face-line CCD
  derivatives of ``psi_f`` and ``u_f``): the plain ``D_f`` is 2nd order on point values and its
  error ``-(h^2/24) psi_xxx`` is a fixed fraction of ``psi_x`` because ``eps = 1.5 Delta``
  (-0.93% at the ``psi = 1/2`` crossing), so the interface lags by that fraction of the distance
  travelled at every N. The ``psi u_xxx`` part is left out, so a ``u_f`` divergence-free for the
  plain ``D_f`` (a projected one) keeps ``psi = 1`` exactly; on walls the correction is zero.
* **B, mass closure**: ``psi += (M - sum psi dV) / sum w dV * w``, ``w = 4 psi (1 - psi)``
  (interface-localised), then clipped (TwoPhaseFlow ``apply_mass_correction``).

Re-initialisation: the paper's standard path is the Ridge--Eikonal distance reconstruction
(stage D, a later step here); it calls the Olsson--Kreiss compression--diffusion equation
``d_tau psi + div(psi (1 - psi) n) = div(eps grad psi)`` (``n = grad psi / |grad psi|``) the
continuous equilibrium only (not adopted), while TwoPhaseFlow's code defaults to it
(``SplitReinitializer``: explicit compression with a dissipative CCD divergence, Crank--Nicolson
ADI diffusion, 4 pseudo-steps, nodal grid). :meth:`CLSInterface.reinitialize` solves the same
equation in conservative finite-volume form on the cell-centred grid, explicitly: face fluxes
``psi(1 - psi) n`` (cell values averaged to the faces, ``n`` from the CCD gradient with
TwoPhaseFlow's floor ``|grad psi| >= 1e-6``) and ``eps`` times FCCD's face gradient, pseudo-step
``dtau = min(Delta^2 / (4 d eps), Delta / 2)`` (TwoPhaseFlow ``compute_dtau``), clipping and the
stage-B closure at the end; its face difference keeps the mass exactly.
"""

from __future__ import annotations

from ..backend import array_module
from ..discretization.ccd import CCDCellDerivative, FCCDFaceComplex
from ..fields import CellField, FaceField

EPSILON_FACTOR = 1.5
#: Pseudo-time step of the ACLS re-initialisations relative to the compression--diffusion one.
ACLS_DTAU_FACTOR = {"acls2008": 1.0, "acls2017": 0.25}


def psi_from_phi(phi, eps: float):
    """``H_eps``: ``(1 + tanh(phi / 2 eps)) / 2`` for a level set ``phi`` positive in the liquid."""
    xp = array_module(phi)
    return 0.5 * (1.0 + xp.tanh(phi / (2.0 * eps)))


def phi_from_psi(psi, eps: float, delta: float = 1e-6):
    """Inverse map (positive in the liquid), ``psi`` limited to ``[delta, 1 - delta]`` first."""
    xp = array_module(psi)
    p = xp.minimum(xp.maximum(psi, delta), 1.0 - delta)
    return 2.0 * eps * xp.arctanh(2.0 * p - 1.0)


def mass_correction(psi, dv, target):
    """Stage B (TwoPhaseFlow ``apply_mass_correction``): interface-weighted, then clipped."""
    xp = array_module(psi)
    m = xp.sum(psi * dv)
    w = 4.0 * psi * (1.0 - psi)
    W = xp.sum(w * dv)
    Ws = xp.where(W > 1e-12, W, 1.0)
    ratio = (xp.asarray(target) - m) / Ws
    gate = xp.where(W > 1e-12, 1.0, 0.0)
    return xp.clip(psi + (gate * ratio) * w, 0.0, 1.0)


def _take_first(a, ax):
    idx = [slice(None)] * a.ndim
    idx[ax] = slice(0, 1)
    return a[tuple(idx)]


class CLSInterface:
    """``psi`` (a :class:`CellField`, boundary conditions of a volume fraction) and its width ``eps``."""

    def __init__(self, psi: CellField, eps: float | None = None, *, ops: FCCDFaceComplex | None = None,
                 mass_closure: bool = True, divergence: str = "plain") -> None:
        grid = psi.grid
        if grid.ndim != 2 or not grid.is_uniform:
            raise NotImplementedError("CLS is implemented on uniform 2-D grids")
        self.psi = psi
        self.eps = EPSILON_FACTOR * float(min(grid.h)) if eps is None else float(eps)
        self.ops = ops or FCCDFaceComplex()
        self.ccd = self.ops.ccd if hasattr(self.ops, "ccd") else CCDCellDerivative()
        self.mass_closure = mass_closure
        if divergence not in ("plain", "fccd"):
            raise ValueError("divergence is 'plain' (the paper's D_f) or 'fccd' (D_f of the corrected flux)")
        self.divergence = divergence
        self._d1_cache: dict = {}
        self.dv = float(grid.h[0] * grid.h[1])

    @classmethod
    def from_level_set(cls, grid, phi_cells, bc, eps: float | None = None, **kw) -> "CLSInterface":
        """``psi = H_eps(phi)`` from cell-centred values of a signed distance positive in the liquid."""
        e = EPSILON_FACTOR * float(min(grid.h)) if eps is None else float(eps)
        return cls(CellField.from_interior(grid, psi_from_phi(phi_cells, e), bc), e, **kw)

    def mass(self) -> float:
        return float(self.psi.interior.sum()) * self.dv

    def curvature(self, method: str = "psi", **kw):
        """Curvature in the interface band: ``method="psi"`` differentiates ``psi`` (the paper,
        :func:`.cls_curvature.psi_curvature`); ``"heights"`` uses height functions of the
        ``psi = 1/2`` crossings (:func:`.cls_heights.height_curvature`, ``method`` there passed as
        ``heights_method="ccd" | "fd2"``)."""
        if method == "heights":
            from .cls_heights import height_curvature  # noqa: PLC0415

            hm = kw.pop("heights_method", "ccd")
            return height_curvature(self.psi, eps=self.eps, method=hm, **kw)
        from .cls_curvature import psi_curvature  # noqa: PLC0415

        return psi_curvature(self.psi, eps=self.eps, ccd=self.ccd, **kw)

    # -- stage A ---------------------------------------------------------------------------
    def _rhs(self, q, uf: FaceField):
        f = CellField.from_interior(self.psi.grid, q, self.psi.bc)
        f.fill_ghosts()
        grid = self.psi.grid
        out = None
        for ax in range(2):
            psi_f = self.ops.interp_f(f, ax)
            flux = psi_f * uf[ax]
            n = grid.n[ax]
            if self.divergence == "fccd":
                flux = flux - self._flux_correction(psi_f, uf[ax], ax)
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
            d = (flux[tuple(hi)] - flux[tuple(lo)]) / float(grid.h[ax])
            out = -d if out is None else out - d
        return out

    def _flux_correction(self, psi_f, un, ax):
        """``(h^2/24)(psi_xx u + 2 psi_x u_x)`` on the faces normal to ``ax`` (zero on walls).

        ``D_f`` of the point flux ``psi_f u_f`` is ``d(psi u)/dx + (h^2/24) d^3(psi u)/dx^3``; the
        terms of ``d^3(psi u)`` with derivatives of ``psi`` grow like ``1/eps^2`` (the profile is
        3 cells wide at every N) and are removed here. ``psi u_xxx`` is kept: the projected ``u_f``
        is divergence-free for the plain ``D_f``, so ``psi = 1`` stays exactly 1. The derivatives
        are face-line CCD derivatives of the face values ``psi_f`` and ``u_f`` (as in
        :meth:`.FCCDFaceComplex.face_flux`, which this equals for a uniform ``u``)."""
        from ..discretization.reconstruction import face_ccd_matrices  # noqa: PLC0415

        grid = self.psi.grid
        xp = grid.xp
        h = float(grid.h[ax])
        n = grid.n[ax]
        periodic = self.psi.bc.is_periodic(ax)
        key = ("cls_d", xp.__name__, n, periodic, str(un.dtype))
        D = self._d1_cache.get(key)
        if D is None:
            D1, D2 = face_ccd_matrices(n, 1.0, periodic)
            D = self._d1_cache[key] = (xp.asarray(D1.T, dtype=un.dtype), xp.asarray(D2.T, dtype=un.dtype))

        def line(v, M):
            moved = xp.moveaxis(v, ax, -1)
            if periodic:
                d = moved[..., :n] @ M
                d = xp.concatenate([d, d[..., :1]], axis=-1)
            else:
                d = moved @ M
            return xp.moveaxis(d, -1, ax)

        corr = (1.0 / 24.0) * (line(psi_f, D[1]) * un + 2.0 * line(psi_f, D[0]) * line(un, D[0]))
        if not periodic:
            w = [slice(None)] * 2
            w[ax] = [0, n]
            corr[tuple(w)] = 0.0
        return corr

    def advect(self, uf: FaceField, dt: float) -> None:
        """One TVD-RK3 step with the face velocity ``uf`` (frozen), clipping each stage, then stage B."""
        xp = self.psi.grid.xp
        q0 = self.psi.interior
        m0 = xp.sum(q0) * self.dv
        q1 = xp.clip(q0 + dt * self._rhs(q0, uf), 0.0, 1.0)
        q2 = xp.clip(0.75 * q0 + 0.25 * (q1 + dt * self._rhs(q1, uf)), 0.0, 1.0)
        q3 = xp.clip(q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * self._rhs(q2, uf)), 0.0, 1.0)
        if self.mass_closure:
            q3 = mass_correction(q3, self.dv, m0)
        self.psi.interior = q3
        self.psi.fill_ghosts()

    # -- re-initialisation ------------------------------------------------------------------------
    def reinitialize(self, n_steps: int = 4, *, method: str = "xi_sdf", code_variant: bool = False,
                     distance: str = "points", acls_options: dict | None = None) -> None:
        """Re-initialise ``psi``: ``method="xi_sdf"`` (the paper's standard stage D on uniform grids) or
        ``"compression"`` (Olsson--Kreiss compression--diffusion, ``n_steps`` pseudo-steps).

        ``"acls2008"`` / ``"acls2017"``: accurate conservative level set re-initialisation with the
        normal of a reconstructed distance (:meth:`_reinit_acls`). Its differences are local on
        purpose: ``phi_map`` amplifies a change of ``psi`` by ``1/(psi(1 - psi))`` towards the
        saturated cells, and CCD's global coupling carries that back to the interface (FCCD/CCD
        derivatives of ``phi_map`` diverged after 700 re-initialisations, N = 128; docs/notes/24).

        ``distance`` (xi-SDF): ``"points"`` is the paper's distance to the nearest crossing point;
        ``"segments"`` the distance to the polyline through the crossings (marching squares on the
        dual cells), which an oblique interface needs (see :func:`_distance_to_crossings`)."""
        if method == "xi_sdf":
            self._reinit_xi_sdf(code_variant, distance)
        elif method == "compression":
            self._reinit_compression(n_steps)
        elif method in ("acls2008", "acls2017"):
            self._reinit_acls(n_steps, method)
        elif method == "acls-fccd":
            self._reinit_acls_fccd(n_steps, **(acls_options or {}))
        else:
            raise ValueError("method is 'xi_sdf', 'compression', 'acls2008', 'acls2017' or 'acls-fccd'")

    def _reinit_xi_sdf(self, code_variant: bool, distance: str = "points") -> None:
        """Ridge--Eikonal distance reconstruction, uniform-grid form (paper 05a, xi-SDF algorithm):

        1. logit ``phi = eps ln(psi / (1 - psi))``, ``psi`` limited to ``[1e-6, 1 - 1e-6]``;
        2. zero crossings between neighbouring cells by linear interpolation (index space);
        3. ``phi`` = sign times the distance to the nearest crossing (no iteration: the crossings stay);
        4. ``psi = H_eps(phi)``;
        5. mass closure by a uniform shift of ``phi`` (``delta = (M - M') / sum psi(1 - psi)/eps dV``,
           repeated until the total is closed to round-off).

        ``code_variant``: also do what TwoPhaseFlow's ``EikonalReinitializer(xi_sdf=True)`` does and the
        paper does not state: clip ``|phi|`` to ``2 eps`` before locating the crossings, and after the
        mass shift keep every cell on its side (``phi`` set to ``+-1e-14``).
        """
        grid = self.psi.grid
        xp = grid.xp
        h = float(grid.h[0])
        eps = self.eps
        q = self.psi.interior
        m0 = xp.sum(q) * self.dv
        phi = phi_from_psi(q, eps)
        sgn = xp.where(phi < 0.0, -1.0, 1.0)
        if code_variant:
            phi = sgn * xp.minimum(xp.abs(phi), 2.0 * eps)
        d = (_distance_to_crossings if distance == "points" else _distance_to_segments)(phi, self.psi.bc)
        if d is None:
            return
        d = d * h
        phi = sgn * d
        new = psi_from_phi(phi, eps)
        if self.mass_closure:
            for _ in range(8):
                w = new * (1.0 - new) / eps
                W = xp.sum(w) * self.dv
                miss = m0 - xp.sum(new) * self.dv
                if float(W) <= 1e-14 or abs(float(miss)) <= 1e-14 * max(float(m0), 1e-300):
                    break
                phi = phi + miss / W
                if code_variant:
                    phi = xp.where(sgn * phi < 0.0, sgn * 1e-14, phi)
                new = psi_from_phi(phi, eps)
        self.psi.interior = new
        self.psi.fill_ghosts()

    # -- re-initialisation (compression--diffusion) --------------------------------------------------
    def _reinit_acls(self, n_steps: int, form: str) -> None:
        """ACLS re-initialisation, ``d_tau psi = div F`` (``F`` in :func:`_acls_flux_divergence`).

        The normal ``n = grad d / |grad d|`` (central differences) comes from a reconstructed signed
        distance ``d``, here the distance to the polyline through the ``psi = 1/2`` crossings (the
        ``distance="segments"`` of the xi-SDF re-initialisation, in place of ACLS's fast marching),
        computed once per call. ``n_steps`` explicit pseudo-steps of the compression--diffusion
        ``dtau`` times :data:`ACLS_DTAU_FACTOR` (the 2017 flux diverged with the full step: across a
        steep cell ``psi (1 - psi)`` on the face exceeds that of the cells, which raises the effective
        diffusion above ``eps``; docs/notes/24), then clipping and the stage-B mass closure."""
        grid = self.psi.grid
        xp = grid.xp
        h = float(min(grid.h))
        q = self.psi.interior
        m0 = xp.sum(q) * self.dv
        phi = phi_from_psi(q, self.eps)
        d = _distance_to_segments(phi, self.psi.bc)
        if d is None:
            return
        d = xp.where(phi < 0.0, -1.0, 1.0) * d * h
        dp = _pad1(d, self.psi.bc)
        gx = (dp[2:, 1:-1] - dp[:-2, 1:-1]) / (2.0 * h)
        gy = (dp[1:-1, 2:] - dp[1:-1, :-2]) / (2.0 * h)
        gn = xp.maximum(xp.sqrt(gx * gx + gy * gy), 1e-12)
        nx, ny = gx / gn, gy / gn
        dtau = min(0.5 * h * h / (2.0 * 2 * self.eps), 0.5 * h) * ACLS_DTAU_FACTOR[form]
        for _ in range(n_steps):
            q = xp.clip(q + dtau * _acls_flux_divergence(q, nx, ny, self.eps, h, form, self.psi.bc), 0.0, 1.0)
        if self.mass_closure:
            q = mass_correction(q, self.dv, m0)
        self.psi.interior = q
        self.psi.fill_ghosts()

    # -- ACLS with FCCD operators (2008 form), explicit or backward Euler + defect correction ----------
    def _acls_normals(self):
        grid = self.psi.grid
        xp = grid.xp
        h = float(min(grid.h))
        q = self.psi.interior
        phi = phi_from_psi(q, self.eps)
        d = _distance_to_segments(phi, self.psi.bc)
        if d is None:
            return None
        d = xp.where(phi < 0.0, -1.0, 1.0) * d * h
        dp = _pad1(d, self.psi.bc)
        gx = (dp[2:, 1:-1] - dp[:-2, 1:-1]) / (2.0 * h)
        gy = (dp[1:-1, 2:] - dp[1:-1, :-2]) / (2.0 * h)
        gn = xp.maximum(xp.sqrt(gx * gx + gy * gy), 1e-12)
        return gx / gn, gy / gn

    def _acls_face_normals(self, nx, ny):
        grid = self.psi.grid
        xp = grid.xp
        cells = []
        for ax, c in enumerate((nx, ny)):
            f = CellField.from_interior(grid, c, _odd_bc(self.psi.bc, ax))
            f.fill_ghosts()
            cells.append(f)
        out = []
        for ax in range(2):
            fx, fy = self.ops.interp_f(cells[0], ax), self.ops.interp_f(cells[1], ax)
            nn = xp.maximum(xp.sqrt(fx * fx + fy * fy), 1e-12)
            out.append((fx / nn, fy / nn))
        return out

    def _acls_fccd_divergence(self, q, fn):
        """div F, F = (eps (grad psi . n) - psi (1 - psi)) n on the faces: FCCD face gradient (normal part),
        FCCD face value of the CCD cell derivative (tangential part) and of psi (1 - psi); zero on walls."""
        grid = self.psi.grid
        f = CellField.from_interior(grid, q, self.psi.bc)
        f.fill_ghosts()
        comp = CellField.from_interior(grid, q * (1.0 - q), self.psi.bc)
        comp.fill_ghosts()
        dcell = []
        for ax in range(2):
            c = CellField.from_interior(grid, self.ccd.d1(f, ax), _odd_bc(self.psi.bc, ax))
            c.fill_ghosts()
            dcell.append(c)
        div = 0.0
        for ax in range(2):
            t = 1 - ax
            fx, fy = fn[ax]
            n_ax, n_t = (fx, fy) if ax == 0 else (fy, fx)
            gn = self.ops.grad_f(f, ax) * n_ax + self.ops.interp_f(dcell[t], ax) * n_t
            flux = (self.eps * gn - self.ops.interp_f(comp, ax)) * n_ax
            flux = _zero_walls(flux, ax, self.psi.bc)
            n = grid.n[ax]
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
            div = div + (flux[tuple(hi)] - flux[tuple(lo)]) / float(grid.h[ax])
        return div

    def _reinit_acls_fccd(self, n_steps: int, *, implicit: bool = False, dtau_factor: float = 1.0,
                          dc_iters: int = 4, stats: list | None = None) -> None:
        """ACLS (2008 form) with FCCD operators, normals from the reconstructed distance, frozen.

        ``implicit=False``: explicit pseudo-steps of the compression--diffusion ``dtau`` times
        ``dtau_factor``. ``implicit=True``: backward Euler ``q - dtau L(q) = q_old`` by defect
        correction, ``dc_iters`` corrections ``(I - dtau L_low) delta = r`` per pseudo-step, ``L_low`` the
        2nd-order operator ``div(eps n_a^2 grad)`` (no cross or compression terms; Basilisk's multigrid),
        ``r`` the residual with the FCCD operator. ``stats``: appended with the residual history."""
        from ..fields import FaceField  # noqa: PLC0415
        from ..linear.poisson import poisson  # noqa: PLC0415

        grid = self.psi.grid
        xp = grid.xp
        h = float(min(grid.h))
        normals = self._acls_normals()
        if normals is None:
            return
        fn = self._acls_face_normals(*normals)
        q = self.psi.interior
        m0 = xp.sum(q) * self.dv
        dtau = min(0.5 * h * h / (2.0 * 2 * self.eps), 0.5 * h) * dtau_factor
        if implicit:
            alpha = FaceField(grid, [self.eps * fn[ax][ax] ** 2 for ax in range(2)])
            for ax in range(2):
                if not self.psi.bc.is_periodic(ax):
                    w = [slice(None)] * 2
                    w[ax] = [0, grid.n[ax]]
                    alpha[ax][tuple(w)] = 0.0
            from ..boundary import BoundaryConditions, Neumann, Periodic  # noqa: PLC0415

            dbc = BoundaryConditions()
            for ax in range(2):
                for side in (0, 1):
                    dbc.set((ax, side), Periodic() if self.psi.bc.get(ax, side).periodic else Neumann(0.0))
        for _ in range(n_steps):
            if not implicit:
                q = xp.clip(q + dtau * self._acls_fccd_divergence(q, fn), 0.0, 1.0)
                continue
            q_old = q
            hist = []
            for _k in range(dc_iters):
                r = q_old - (q - dtau * self._acls_fccd_divergence(q, fn))
                hist.append(float(xp.max(xp.abs(r))))
                delta = CellField(grid, bc=dbc)
                poisson(delta, -r / dtau, alpha, -1.0 / dtau, tolerance=1e-10)
                q = q + delta.interior
            r = q_old - (q - dtau * self._acls_fccd_divergence(q, fn))
            hist.append(float(xp.max(xp.abs(r))))
            if stats is not None:
                stats.append(hist)
            q = xp.clip(q, 0.0, 1.0)
        if self.mass_closure:
            q = mass_correction(q, self.dv, m0)
        self.psi.interior = q
        self.psi.fill_ghosts()

    def _reinit_compression(self, n_steps: int = 4) -> None:
        grid = self.psi.grid
        xp = grid.xp
        h = float(min(grid.h))
        dtau = min(0.5 * h * h / (2.0 * 2 * self.eps), 0.5 * h)
        q = self.psi.interior
        m0 = xp.sum(q) * self.dv
        for _ in range(n_steps):
            f = CellField.from_interior(grid, q, self.psi.bc)
            f.fill_ghosts()
            grads = [self.ccd.d1(f, ax) for ax in range(2)]
            norm = xp.maximum(xp.sqrt(xp.maximum(grads[0] ** 2 + grads[1] ** 2, 1e-12)), 1e-6)
            comp = CellField.from_interior(grid, q * (1.0 - q), self.psi.bc)
            comp.fill_ghosts()
            div = None
            for ax in range(2):
                nf = CellField.from_interior(grid, grads[ax] / norm, _odd_bc(self.psi.bc, ax))
                nf.fill_ghosts()
                flux = (self.ops.interp_f(comp, ax) * self.ops.interp_f(nf, ax)
                        - self.eps * self.ops.grad_f(f, ax))
                flux = _zero_walls(flux, ax, self.psi.bc)
                n = grid.n[ax]
                lo = [slice(None)] * 2
                hi = [slice(None)] * 2
                lo[ax], hi[ax] = slice(0, n), slice(1, n + 1)
                d = (flux[tuple(hi)] - flux[tuple(lo)]) / float(grid.h[ax])
                div = d if div is None else div + d
            q = xp.clip(q - dtau * div, 0.0, 1.0)
        if self.mass_closure:
            q = mass_correction(q, self.dv, m0)
        self.psi.interior = q
        self.psi.fill_ghosts()


def _pad1(a, bc):
    """One ghost layer: periodic wrap or edge copy (zero normal derivative) per axis."""
    xp = array_module(a)
    for ax in range(2):
        mode = "wrap" if bc.get(ax, 0).periodic else "edge"
        width = [(0, 0), (0, 0)]
        width[ax] = (1, 1)
        a = xp.pad(a, width, mode=mode)
    return a


def _acls_flux_divergence(q, nx, ny, eps, h, form, bc, delta=1e-12):
    """Divergence of the ACLS re-initialisation flux (conservative, zero through walls).

    ``form="acls2008"`` (Desjardins, Moureau & Pitsch 2008): ``F = (eps grad psi . n - psi (1 - psi)) n``.
    ``form="acls2017"`` (Chiodi & Desjardins 2017): ``F = psi (1 - psi) (|grad phi_map . n| - 1) n`` with
    ``phi_map = eps ln(psi / (1 - psi))`` and ``psi (1 - psi) = 1 / (4 cosh^2(phi_map / 2 eps))``; the
    face gradient of the mapped distance is exact for a planar profile, so the two terms cancel there.
    On each face the normal derivative is the two-point difference, the tangential one the average of
    the two cells' central differences, ``n`` the normalised average of the cell normals."""
    xp = array_module(q)
    if form == "acls2017":
        qc = xp.minimum(xp.maximum(q, delta), 1.0 - delta)
        g = eps * xp.log(qc / (1.0 - qc))
    else:
        g = q
    gp = _pad1(g, bc)
    np_ = [_pad1(nx, bc), _pad1(ny, bc)]
    n0, n1 = q.shape
    div = 0.0
    for ax in range(2):
        lo = [slice(1, n0 + 1), slice(1, n1 + 1)]
        lo[ax] = slice(0, (n0, n1)[ax] + 1)
        hi = list(lo)
        hi[ax] = slice(1, (n0, n1)[ax] + 2)
        gl, gr = gp[tuple(lo)], gp[tuple(hi)]
        dn = (gr - gl) / h
        t = 1 - ax  # tangential axis: central differences in the two cells, averaged
        cd = (xp.roll(gp, -1, t) - xp.roll(gp, 1, t)) / (2.0 * h)
        dt = 0.5 * (cd[tuple(lo)] + cd[tuple(hi)])
        nfx = 0.5 * (np_[0][tuple(lo)] + np_[0][tuple(hi)])
        nfy = 0.5 * (np_[1][tuple(lo)] + np_[1][tuple(hi)])
        nn = xp.maximum(xp.sqrt(nfx * nfx + nfy * nfy), 1e-12)
        nfx, nfy = nfx / nn, nfy / nn
        n_ax, n_t = (nfx, nfy) if ax == 0 else (nfy, nfx)
        gn = dn * n_ax + dt * n_t
        if form == "acls2017":
            gf = 0.5 * (gl + gr)
            w = 1.0 / (4.0 * xp.cosh(gf / (2.0 * eps)) ** 2)
            flux = w * (xp.abs(gn) - 1.0) * n_ax
        else:
            qf = 0.5 * (gl + gr)
            flux = (eps * gn - qf * (1.0 - qf)) * n_ax
        if not bc.get(ax, 0).periodic:
            w0 = [slice(None)] * 2
            w0[ax] = [0, (n0, n1)[ax]]
            flux[tuple(w0)] = 0.0
        a0 = [slice(None)] * 2
        a1 = [slice(None)] * 2
        a0[ax], a1[ax] = slice(0, (n0, n1)[ax]), slice(1, (n0, n1)[ax] + 1)
        div = div + (flux[tuple(a1)] - flux[tuple(a0)]) / h
    return div


def _odd_bc(bc, axis):
    """Boundary conditions of a normal component: periodic stays, walls normal to ``axis`` antisymmetric."""
    from ..boundary import BoundaryConditions, Dirichlet, Periodic  # noqa: PLC0415

    out = BoundaryConditions()
    for ax in range(2):
        for side in (0, 1):
            cond = bc.get(ax, side)
            out.set((ax, side), Periodic() if cond.periodic else (Dirichlet(0.0) if ax == axis else cond))
    return out


def _zero_walls(flux, axis, bc):
    """No flux through walls (the reinitialisation conserves psi exactly)."""
    if bc.get(axis, 0).periodic:
        return flux
    idx0 = [slice(None)] * 2
    idx1 = [slice(None)] * 2
    idx0[axis], idx1[axis] = 0, -1
    flux = flux.copy()
    flux[tuple(idx0)] = 0.0
    flux[tuple(idx1)] = 0.0
    return flux


def _distance_to_crossings(phi, bc, chunk: int = 4096):
    """Distance (in cells) from every cell centre to the nearest zero crossing of ``phi`` between
    neighbouring cells (linear interpolation; across a periodic boundary too). ``None`` if no crossing."""
    xp = array_module(phi)
    nx, ny = phi.shape
    pts = []
    for axis in range(2):
        a = phi
        if bc.get(axis, 0).periodic:
            b = xp.roll(phi, -1, axis=axis)
        else:
            a = phi[:-1, :] if axis == 0 else phi[:, :-1]
            b = phi[1:, :] if axis == 0 else phi[:, 1:]
        cut = (a * b) < 0.0
        ii, jj = xp.nonzero(cut)
        if ii.size == 0:
            continue
        pa, pb = xp.abs(a[ii, jj]), xp.abs(b[ii, jj])
        t = pa / (pa + pb)
        x = ii.astype(phi.dtype) + (t if axis == 0 else 0.0)
        y = jj.astype(phi.dtype) + (t if axis == 1 else 0.0)
        pts.append(xp.stack([x, y], axis=1))
    if not pts:
        return None
    pts = xp.concatenate(pts, axis=0)
    I, J = xp.meshgrid(xp.arange(nx, dtype=phi.dtype), xp.arange(ny, dtype=phi.dtype), indexing="ij")
    I, J = I.ravel(), J.ravel()
    per = (bc.get(0, 0).periodic, bc.get(1, 0).periodic)
    out = xp.empty(nx * ny, dtype=phi.dtype)
    for s in range(0, nx * ny, chunk):
        dx = I[s:s + chunk, None] - pts[None, :, 0]
        dy = J[s:s + chunk, None] - pts[None, :, 1]
        if per[0]:
            dx = dx - nx * xp.round(dx / nx)
        if per[1]:
            dy = dy - ny * xp.round(dy / ny)
        out[s:s + chunk] = xp.sqrt(xp.min(dx * dx + dy * dy, axis=1))
    return out.reshape(nx, ny)


def _distance_to_segments(phi, bc, chunk: int = 2048):
    """Distance (in cells) from every cell centre to the polyline through the zero crossings: on each
    dual cell (four neighbouring centres) the crossings of its edges are joined by segments (marching
    squares; the ambiguous saddle joins them by the sign of the mean). ``None`` if no crossing."""
    xp = array_module(phi)
    nx, ny = phi.shape
    per = (bc.get(0, 0).periodic, bc.get(1, 0).periodic)
    p = phi
    if per[0]:
        p = xp.concatenate([p, p[:1]], axis=0)
    if per[1]:
        p = xp.concatenate([p, p[:, :1]], axis=1)
    a, b, c, d = p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]  # corners (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    I, J = xp.meshgrid(xp.arange(a.shape[0], dtype=phi.dtype), xp.arange(a.shape[1], dtype=phi.dtype),
                       indexing="ij")

    def cross(u, v):
        return xp.abs(u) / xp.where((xp.abs(u) + xp.abs(v)) > 0, xp.abs(u) + xp.abs(v), 1.0)

    # crossing points on the four edges of each dual cell (bottom, right, top, left)
    e = [((a * b) < 0, I + cross(a, b), J),
         ((b * c) < 0, I + 1.0, J + cross(b, c)),
         ((d * c) < 0, I + cross(d, c), J + 1.0),
         ((a * d) < 0, I, J + cross(a, d))]
    segs = []
    count = sum(m.astype(xp.int32) for m, _, _ in e)
    two = count == 2
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for k, l in pairs:
        m = two & e[k][0] & e[l][0]
        ii, jj = xp.nonzero(m)
        if ii.size:
            segs.append(xp.stack([e[k][1][ii, jj], e[k][2][ii, jj], e[l][1][ii, jj], e[l][2][ii, jj]], axis=1))
    four = count == 4
    if bool(xp.any(four)):
        mean = (a + b + c + d) / 4.0
        same = (mean * a) > 0  # centre on the side of corner a: segments (bottom-left) and (top-right) cut it off
        for (k, l), sel in (((0, 3), ~same), ((1, 2), ~same), ((0, 1), same), ((2, 3), same)):
            m = four & sel
            ii, jj = xp.nonzero(m)
            if ii.size:
                segs.append(xp.stack([e[k][1][ii, jj], e[k][2][ii, jj], e[l][1][ii, jj], e[l][2][ii, jj]], axis=1))
    if not segs:
        return None
    S = xp.concatenate(segs, axis=0)
    ax, ay, bx, by = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    vx, vy = bx - ax, by - ay
    L2 = xp.where(vx * vx + vy * vy > 0, vx * vx + vy * vy, 1.0)
    CI, CJ = xp.meshgrid(xp.arange(nx, dtype=phi.dtype), xp.arange(ny, dtype=phi.dtype), indexing="ij")
    CI, CJ = CI.ravel(), CJ.ravel()
    out = xp.empty(nx * ny, dtype=phi.dtype)
    for s0 in range(0, nx * ny, chunk):
        px = CI[s0:s0 + chunk, None] - ax[None, :]
        py = CJ[s0:s0 + chunk, None] - ay[None, :]
        if per[0]:
            px = px - nx * xp.round(px / nx)
        if per[1]:
            py = py - ny * xp.round(py / ny)
        t = xp.clip((px * vx[None, :] + py * vy[None, :]) / L2[None, :], 0.0, 1.0)
        qx, qy = px - t * vx[None, :], py - t * vy[None, :]
        out[s0:s0 + chunk] = xp.sqrt(xp.min(qx * qx + qy * qy, axis=1))
    return out.reshape(nx, ny)

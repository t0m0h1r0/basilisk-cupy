"""Case C: viscosity, GFM surface tension, projection and the (band-corrected) interface kinematics solved together
in each implicit stage, with first- and second-order time integrators (docs/notes/47). A prototype in ``cases/``.

Stage equation ``X = B + theta dt f(X)`` for ``X = (u, uf, p, psi)``:

* cell:  ``u  = B_u + theta dt (nu L u + f2c(a(psi) - alpha G p))``
* face:  ``uf = B_f + theta dt (interp(nu L u) + a(psi) - alpha G p)``, ``D uf = 0`` (``p`` the stage multiplier)
* psi:   ``psi = B_psi - theta dt T(psi) uf`` (``T`` the band-corrected transport of :mod:`cases.velocity_extension`,
  frozen at the current iterate, no clipping, no mass closure)

solved by fixed-point iterations: GFM force from the current psi, implicit viscosity (``viscous.solve`` with
``theta dt``, the latest pressure gradient inside as CenteredNS does), projection, interface update, until the
change of ``u`` and ``psi`` is below ``tol``. Integrators: ``"be"`` (theta = 1), ``"cn"`` (theta = 1/2, explicit
half ``f(X^n)``), ``"sdirk2"`` (two stages, ``gamma = 1 - 1/sqrt 2``, L-stable).

:class:`CaseCMap` replaces :meth:`cases.capwave_jacobian.CapwaveMap.F` so that the discrete dispersion relation
(:func:`dispersion`) can be measured with the same machinery.
"""

from __future__ import annotations

import math

import numpy as np

from basilisk_cupy.fields import CellField, FaceField
from basilisk_cupy.projection.approximate import project
from cases.capwave_jacobian import CapwaveMap, reduced
from cases.velocity_extension import advective_rhs, corrected_face_velocity

GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


class Stepper:
    def __init__(self, ns, tp, method="cn", tol=1e-13, maxit=200, face_base="interp"):
        self.ns, self.tp, self.cls = ns, tp, tp.cls
        self.method, self.tol, self.maxit = method, tol, maxit
        if face_base not in ("interp", "faces"):
            raise ValueError("face_base is 'interp' (uf rebuilt from the cells, CenteredNS) or 'faces' (uf^n kept)")
        self.face_base = face_base
        self.iterations = []
        self._p = CellField(ns.grid, bc=ns.p.bc, name="p_stage")

    # -- pieces ------------------------------------------------------------------------------
    def face_force(self, psi):
        ns, tp, cls = self.ns, self.tp, self.cls
        cls.psi.interior = psi
        cls.psi.fill_ghosts()
        tp._properties(ns)
        for ax in range(2):
            ns.a[ax] = 0.0 * np.asarray(ns.alpha[ax])
        tp._interfacial_force(ns)
        return [np.asarray(ns.a[ax], dtype=float).copy() for ax in range(2)]

    def nu_lap(self, u):
        """``nu L u`` (cell, per unit density) of the viscous solver's high-order operator."""
        ns = self.ns
        tmp = ns.u.copy("lap_tmp")
        for c, v in zip(tmp, u):
            c.interior = v
        tmp.fill_ghosts()
        from basilisk_cupy.terms.viscous import cell_viscosity

        high = ns.viscous.high(tmp, cell_viscosity(ns.mu), 1.0, 1.0, ns.mu)
        return [v - h for v, h in zip(u, high)]

    def interp(self, u):
        ns = self.ns
        tmp = ns.u.copy("i_tmp")
        for c, v in zip(tmp, u):
            c.interior = v
        tmp.fill_ghosts()
        return [np.asarray(ns.ops.interp_f(tmp[ax], ax)) for ax in range(2)]

    def f2c(self, faces):
        return [np.asarray(self.ns.ops.face_to_cell(faces[ax], ax, periodic=False)) for ax in range(2)]

    def grad_p(self, p):
        ns = self._p
        ns.interior = p
        ns.fill_ghosts()
        return [np.asarray(self.ns.alpha[ax]) * np.asarray(self.ns.ops.grad_f(ns, ax)) for ax in range(2)]

    def psi_rate(self, psi, uf):
        cls = self.cls
        cls.psi.interior = psi
        cls.psi.fill_ghosts()
        uff = FaceField(cls.psi.grid, uf, "uf")
        ufc = corrected_face_velocity(cls, uff)
        delta = FaceField(cls.psi.grid, [ufc[a] - uff[a] for a in range(2)], "delta")
        return np.asarray(cls._rhs(psi, uff) + advective_rhs(cls, psi, delta))

    def wall(self, uf):
        out = [x.copy() for x in uf]
        out[0][[0, -1], :] = 0.0
        out[1][:, [0, -1]] = 0.0
        return out

    # -- one implicit stage ------------------------------------------------------------------
    def stage(self, Bu, Bf, Bpsi, tdt, guess):
        ns = self.ns
        u, uf, p, psi = (np.copy(x) if not isinstance(x, list) else [y.copy() for y in x] for x in guess)
        for it in range(self.maxit):
            a = self.face_force(psi)
            gp = self.grad_p(p)
            g = self.f2c([a[ax] - gp[ax] for ax in range(2)])
            r = [Bu[ax] + tdt * g[ax] for ax in range(2)]
            for c, v in zip(ns.u, r):
                c.interior = v
            ns.u.fill_ghosts()
            ns.viscous.solve(ns.u, ns.mu, ns.rho, tdt, tolerance=1e-14)
            un = [np.asarray(c.interior).copy() for c in ns.u]
            visc = [un[ax] - r[ax] for ax in range(2)]  # tdt nu L u
            ivisc = self.interp(visc)
            ufs = FaceField(ns.grid, self.wall([Bf[ax] + ivisc[ax] + tdt * a[ax] for ax in range(2)]), "uf*")
            self._p.interior = p
            project(ufs, self._p, ns.alpha, tdt, ops=ns.ops, tolerance=1e-14)
            pn = np.asarray(self._p.interior).copy()
            ufn = self.wall([np.asarray(ufs[ax]).copy() for ax in range(2)])
            gpn = self.grad_p(pn)
            gn = self.f2c([a[ax] - gpn[ax] for ax in range(2)])
            un = [Bu[ax] + visc[ax] + tdt * gn[ax] for ax in range(2)]
            psin = Bpsi + tdt * self.psi_rate(psi, ufn)  # psi_rate is psi_t = -T uf
            du = max(float(np.abs(un[ax] - u[ax]).max()) for ax in range(2))
            dpsi = float(np.abs(psin - psi).max())
            scale = max(1e-30, max(float(np.abs(x).max()) for x in un))
            u, uf, p, psi = un, ufn, pn, psin
            if it > 0 and du < self.tol * max(scale, 1.0) and dpsi < self.tol:
                break
        self.iterations.append(it + 1)
        f_u = [(u[ax] - Bu[ax]) / tdt for ax in range(2)]
        f_f = [(uf[ax] - Bf[ax]) / tdt for ax in range(2)]
        f_psi = (psi - Bpsi) / tdt
        return (u, uf, p, psi), (f_u, f_f, f_psi)

    def explicit_f(self, u, uf, p, psi):
        """``f(X)`` at a given state (for CN's explicit half)."""
        a = self.face_force(psi)
        gp = self.grad_p(p)
        lap = self.nu_lap(u)
        f_u = [lap[ax] + self.f2c([a[k] - gp[k] for k in range(2)])[ax] for ax in range(2)]
        il = self.interp(lap)
        f_f = self.wall([il[ax] + a[ax] - gp[ax] for ax in range(2)])
        f_psi = self.psi_rate(psi, uf)
        return f_u, f_f, f_psi

    def step(self, u, uf, p, psi, dt):
        iu = self.interp(u) if self.face_base == "interp" else [x.copy() for x in uf]
        X0 = (u, uf, p, psi)
        if self.method == "be":
            X, _ = self.stage(u, self.wall(iu), psi, dt, X0)
            return X
        if self.method == "cn":
            fu, ff, fpsi = self.explicit_f(u, uf, p, psi)
            Bu = [u[ax] + 0.5 * dt * fu[ax] for ax in range(2)]
            Bf = self.wall([iu[ax] + 0.5 * dt * ff[ax] for ax in range(2)])
            X, _ = self.stage(Bu, Bf, psi + 0.5 * dt * fpsi, 0.5 * dt, X0)
            return X
        if self.method == "sdirk2":
            X1, (fu, ff, fpsi) = self.stage(u, self.wall(iu), psi, GAMMA * dt, X0)
            w = (1.0 - GAMMA) * dt
            Bu = [u[ax] + w * fu[ax] for ax in range(2)]
            Bf = self.wall([iu[ax] + w * ff[ax] for ax in range(2)])
            X, _ = self.stage(Bu, Bf, psi + w * fpsi, GAMMA * dt, X1)
            return X
        raise ValueError(self.method)


class CaseCMap(CapwaveMap):
    """The capillary-wave map of :mod:`cases.capwave_jacobian` stepped by :class:`Stepper`."""

    def __init__(self, n=32, f=1.0, DT=None, method="cn", viscous="fd2", face_base="interp", **kw):
        from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity
        from cases import capillary_wave as cw

        DT = cw.STEP if DT is None else DT
        self._method = method
        self._face_base = face_base
        super().__init__(n, f, DT, curvature=kw.pop("curvature", "exact"), cls_kinematics="extension",
                         viscous=CCDViscosity() if viscous == "ccd" else FD2Viscosity(), **kw)

    def F(self, q):
        if not hasattr(self, "_stepper"):
            self._stepper = Stepper(self.ns, self.tp, self._method, face_base=self._face_base)
        self.set(q)
        ns, cls = self.ns, self.cls
        u = [np.asarray(c.interior).copy() for c in ns.u]
        uf = [np.asarray(ns.uf[ax]).copy() for ax in range(2)]
        p = np.asarray(ns.p.interior).copy()
        psi = np.asarray(cls.psi.interior).copy()
        dt = self.clock[3]
        un, ufn, pn, psin = self._stepper.step(u, uf, p, psi, dt)
        for c, v in zip(ns.u, un):
            c.interior = v
        ns.u.fill_ghosts()
        ns.uf = FaceField(self.grid, ufn, "uf")
        ns.p.interior = pn
        ns.p.fill_ghosts()
        cls.psi.interior = psin
        cls.psi.fill_ghosts()
        # g is not part of this scheme's state; keep it zero
        for c in ns.g:
            c.interior = 0.0 * np.asarray(c.interior)
        ns.g.fill_ghosts()
        ns.clock.dt = dt
        return self.get()


def dispersion(n=32, f=1.0, DT=None, method="cn", viscous="fd2", face_base="interp", **kw):
    import cmath

    from cases.capwave_normal_mode import K, mode

    cm = CaseCMap(n, f, DT, method, viscous, face_base, **kw)
    s, _ = mode(K, 1.0, 1.0, cm.nu)
    M, res = reduced(cm)
    lam, vecs = np.linalg.eig(M)
    tg = cmath.exp(s * cm.dt)
    k = int(np.argmin(np.abs(lam - tg)))
    sh = complex(np.log(lam[k]) / cm.dt)
    return {"s": s, "s_h": sh, "damping_err": sh.real / s.real - 1, "frequency_err": sh.imag / s.imag - 1,
            "residual": res, "dt": cm.dt, "lam": lam, "vector": vecs[:, k],
            "iterations": float(np.mean(cm._stepper.iterations)) if hasattr(cm, "_stepper") else None}

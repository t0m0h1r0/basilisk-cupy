"""Discrete dispersion relation of the capillary wave (docs/notes/41).

The scheme (CenteredNS + TwoPhaseCLS, the set-up of :mod:`cases.capwave_normal_mode`) is linearised about the
flat interface at rest, which is exactly steady. One step's Jacobian ``J`` (central differences on the whole
state ``q = (psi, u, uf, p, g)``, :class:`cases.step_jacobian.StepMap`'s state) is applied matrix-free;
Arnoldi started from the exact normal mode gives the Ritz value ``lambda`` closest to ``exp(s dt)``, and
``s_h = log(lambda) / dt``. Two steps ``dt`` and ``dt/2`` give the semi-discrete value by Richardson.

Variants (``kw``): ``viscous`` (e.g. ``CCDViscosity()``), ``ops_ns`` / ``jump_mode`` (FCCD faces),
``curvature``: ``"psi"`` (default), ``"heights"``, ``"exact"`` (``kappa = -eta''`` from the ``psi = 1/2``
crossings by a cosine series in x, set on every cell of the column: the linear curvature without the
discretisation of psi's derivatives).

    python -m cases.capwave_jacobian
"""

from __future__ import annotations

import cmath
import math
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases import capillary_wave as cw
from cases.capwave_normal_mode import K, mode
from cases.step_jacobian import arnoldi


def crossing_heights(tp):
    """``eta(x)``: the ``psi = 1/2`` crossing height in each column (linear interpolation between centres)."""
    grid = tp.f.grid
    psi = tp.cls.psi.interior
    y = np.asarray(grid.centers(1))
    lo, hi = psi[:, :-1] - 0.5, psi[:, 1:] - 0.5
    cross = (lo * hi <= 0) & (lo != hi)
    t = np.where(cross, lo / np.where(cross, lo - hi, 1.0), 0.0)
    return np.nanmean(np.where(cross, y[None, :-1] + t * np.diff(y)[None, :], np.nan), axis=1)


def sharp_kinematics(tp, where: str = "interface"):
    """Replace the CLS transport by ``eta_t = v`` with ``v`` the face velocity ``uf_y`` interpolated linearly
    in y at ``eta`` (``where="interface"``), or its average over the psi band weighted by ``psi(1 - psi)``
    (``"band"``, what a band transport sees); ``psi`` is rebuilt as ``H_eps(y - eta)``."""
    from basilisk_cupy.interface.cls import psi_from_phi

    grid = tp.f.grid
    cls = tp.cls
    yf = np.asarray(grid.faces(1))
    yc = np.asarray(grid.centers(1))

    from basilisk_cupy.interface.cls import phi_from_psi

    def advect(uf, dt):
        # exact inversion of psi = H_eps(y - eta): eta = y - phi at the two cells nearest the crossing
        q = cls.psi.interior
        phi = phi_from_psi(q, cls.eps)
        j = np.argmin(np.abs(q - 0.5), axis=1)
        i = np.arange(q.shape[0])
        j2 = np.where(q[i, j] > 0.5, j - 1, j + 1)
        eta = 0.5 * ((yc[j] - phi[i, j]) + (yc[j2] - phi[i, j2]))
        vf = np.asarray(uf[1])  # (nx, ny + 1) on the y faces
        if where == "interface":
            v = np.array([np.interp(e, yf, vf[i]) for i, e in enumerate(eta)])
        elif where == "weights":  # the CLS transport's own weights, applied to the face velocity
            v = vf @ tp._kin_w
        else:
            q = cls.psi.interior
            w = q * (1 - q)
            vc = 0.5 * (vf[:, 1:] + vf[:, :-1])
            v = np.sum(w * vc, axis=1) / np.sum(w, axis=1)
        eta = eta + dt * v
        cls.psi.interior = psi_from_phi(yc[None, :] - eta[:, None], cls.eps)
        cls.psi.fill_ghosts()

    cls.advect = advect


def kinematic_weights(tp):
    """Weights ``w_f`` with which the discrete CLS transport moves the ``psi = 1/2`` crossing of the flat
    interface: ``eta_t = sum_f w_f V_f`` for a divergence-free ``cos kx`` mode ``uf_y = V_f cos kx`` (``uf_x``
    from the discrete continuity). Returns ``(y_faces, w)``."""
    grid = tp.f.grid
    cls = tp.cls
    nx, ny = grid.n
    h = float(grid.h[1])
    hx = float(grid.h[0])
    xc = np.asarray(grid.centers(0))
    xf = np.asarray(grid.faces(0))
    yc = np.asarray(grid.centers(1))
    q = cls.psi.interior.copy()
    j0 = int(np.argmin(np.abs(q[0] - 0.5)))
    j1 = j0 + 1 if q[0, j0] < 0.5 else j0 - 1
    lo, hi = min(j0, j1), max(j0, j1)
    w = np.zeros(ny + 1)
    wx = np.cos(K * xc) * 2.0 / nx
    for f in range(1, ny):
        V = np.zeros(ny + 1)
        V[f] = 1.0
        U = -(V[1:] - V[:-1]) / h * hx / (2.0 * np.sin(K * hx / 2.0))  # discrete continuity per row
        ufx = U[None, :] * np.sin(K * xf)[:, None]
        ufy = V[None, :] * np.cos(K * xc)[:, None]
        rhs = cls._rhs(q, FaceField(grid, [ufx, ufy], "uf"))
        pl, ph = q[:, lo], q[:, hi]
        dl, dh = rhs[:, lo], rhs[:, hi]
        # crossing y* = y_lo + h (1/2 - p_lo) / (p_hi - p_lo)
        dp = ph - pl
        ystar_t = h * (-dl * dp - (0.5 - pl) * (dh - dl)) / dp ** 2
        w[f] = float(np.sum(ystar_t * wx))
    return np.asarray(grid.faces(1)), w


def exact_linear_curvature(tp):
    """``kappa(x) = -eta''(x)`` (cosine series, walls at x = 0, L) of the ``psi = 1/2`` crossing heights."""
    grid = tp.f.grid
    psi = tp.cls.psi.interior
    y = np.asarray(grid.centers(1))
    lo, hi = psi[:, :-1] - 0.5, psi[:, 1:] - 0.5
    cross = (lo * hi <= 0) & (lo != hi)
    t = np.where(cross, lo / np.where(cross, lo - hi, 1.0), 0.0)
    yc = np.where(cross, y[None, :-1] + t * np.diff(y)[None, :], np.nan)
    eta = np.nanmean(yc, axis=1)
    nx = eta.size
    L = float(grid.length[0])
    xc = np.asarray(grid.centers(0))
    # cosine series on the cell centres (DCT-II), kappa = sum a_m (m pi / L)^2 cos(m pi x / L)
    m = np.arange(nx)
    C = np.cos(np.pi * m[None, :] * xc[:, None] / L)
    a = np.linalg.solve(C, eta)
    kap = C @ (a * (np.pi * m / L) ** 2)
    # sign: kappa > 0 for a liquid drop (Basilisk); liquid above (psi = 1 for y > eta): a crest is convex
    # towards the gas, kappa = +eta'' there... the potential is sigma * kappa with the code's convention
    return np.broadcast_to(kap[:, None], psi.shape).copy()


class CapwaveMap:
    def __init__(self, n=32, f=1.0, DT=cw.STEP, curvature="psi", tolerance=1e-14, eps=1e-6, **kw):
        warnings.filterwarnings("ignore")
        self.nu = cw.MU * f
        grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
        x, y = grid.meshgrid()
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=tolerance,
                        viscous=kw.pop("viscous", None), ops=kw.pop("ops_ns", None))
        cls = CLSInterface.from_level_set(grid, y + 0.0 * x, BoundaryConditions(), divergence="fccd",
                                          ops=FCCDFaceComplex(order=6))
        ckw = {"method": "heights"} if curvature == "heights" else {}
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=self.nu, mu2=self.nu, vof_first=True, pressure_jump=True,
                         curvature_kw=ckw, capillary_dt_factor=1e9, jump_mode=kw.pop("jump_mode", "sharp"))
        tp.add_surface_tension(1.0)
        if curvature == "exact":
            tp.potentials[0] = lambda t_: t_.sigma * self._sign * exact_linear_curvature(t_)
        hyper = kw.pop("hyper", None)
        if hyper is not None:
            from cases.split_hyperviscosity import attach

            attach(ns, **hyper)
        kin = kw.pop("kinematics", None)
        if kin == "weights":
            tp._kin_w = kinematic_weights(tp)[1]
        if kin is not None:
            sharp_kinematics(tp, kin)
        self._sign = 1.0
        self.grid, self.ns, self.cls, self.tp, self.eps = grid, ns, cls, tp, eps
        ns.DT = DT
        ns.start()
        ns.clock.events = []  # start() adds an event at t + dt0 that would cut the step
        self.clock = (1.0, 1, 1.0 + DT, DT)  # t != 0: Basilisk's CFL smoothing would cut the first step to dt/11
        self.q0 = self.get()
        self.F(self.q0)
        self.dt = float(ns.clock.dt)  # the step actually taken (checked against DT)
        if abs(self.dt - DT) > 1e-12 * DT:
            raise RuntimeError(f"the map steps by {self.dt}, not DT = {DT}")
        if curvature == "exact":  # match the code's sign convention: compare with psi curvature on a wavy interface
            psi0 = cls.psi.interior.copy()
            cls.psi.interior = CLSInterface.from_level_set(grid, y - 0.01 * np.cos(K * x), BoundaryConditions()).psi.interior
            cls.psi.fill_ghosts()
            kp = cls.curvature()
            ke = exact_linear_curvature(tp)
            band = kp != 0
            self._sign = float(np.sign(np.sum(kp[band] * ke[band])))
            cls.psi.interior = psi0
            cls.psi.fill_ghosts()

    def get(self):
        ns, cls = self.ns, self.cls
        parts = [cls.psi.interior, ns.u[0].interior, ns.u[1].interior, ns.uf[0], ns.uf[1], ns.p.interior,
                 ns.g[0].interior, ns.g[1].interior]
        self.shapes = [np.shape(a) for a in parts]
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in parts])

    def set(self, q):
        ns, cls = self.ns, self.cls
        out, k = [], 0
        for s in self.shapes:
            m = int(np.prod(s))
            out.append(q[k:k + m].reshape(s).copy())
            k += m
        cls.psi.interior = out[0]
        cls.psi.fill_ghosts()
        ns.u[0].interior, ns.u[1].interior = out[1], out[2]
        ns.u.fill_ghosts()
        ns.uf = FaceField(self.grid, [out[3], out[4]], "uf")
        ns.p.interior = out[5]
        ns.p.fill_ghosts()
        ns.g[0].interior, ns.g[1].interior = out[6], out[7]
        ns.g.fill_ghosts()

    def F(self, q):
        self.set(q)
        c = self.ns.clock
        c.t, c.i, c.tnext, c.dt = self.clock
        c.events = []
        c.previous = self.clock[3]
        self.ns.step()
        return self.get()

    def deflate(self, v):
        v = v.copy()
        k = 0
        for idx, s in enumerate(self.shapes):
            m = int(np.prod(s))
            if idx == 5:
                v[k:k + m] -= v[k:k + m].mean()
            k += m
        return v

    def jvp(self, v):
        v = self.deflate(v)
        nv = float(np.linalg.norm(v)) or 1.0
        e = self.eps * math.sqrt(v.size) / nv
        return self.deflate((self.F(self.q0 + e * v) - self.F(self.q0 - e * v)) / (2 * e))

    def start_vector(self):
        """The exact normal mode (real part) as a state: psi of eta = a cos kx, u, uf = interp(u), p, g = 0."""
        s, Fm = mode(K, 1.0, 1.0, self.nu)
        x, y = self.grid.meshgrid()
        u, v, p, _, _ = Fm(x, y)
        psi1 = CLSInterface.from_level_set(self.grid, y - 0.01 * np.cos(K * x), BoundaryConditions(),
                                           eps=self.cls.eps).psi.interior - self.cls.psi.interior
        parts = [psi1, u.real, v.real, np.zeros(self.shapes[3]), np.zeros(self.shapes[4]), np.zeros(self.shapes[5]),
                 np.zeros(self.shapes[6]), np.zeros(self.shapes[7])]
        return np.concatenate([np.asarray(a).ravel() for a in parts]), s


def eigen(n=32, f=1.0, DT=cw.STEP, m=30, **kw):
    cm = CapwaveMap(n, f, DT, **kw)
    v0, s = cm.start_vector()
    target = cmath.exp(s * cm.dt)
    lam, X = arnoldi(cm.jvp, cm.q0.size, m=m, power=0, nev=m, seed=0) if False else _arnoldi_from(cm.jvp, v0, m)
    k = int(np.argmin(np.abs(lam - target)))
    sh = complex(np.log(lam[k]) / cm.dt)
    return {"s": s, "s_h": sh, "damping_err": sh.real / s.real - 1, "frequency_err": sh.imag / s.imag - 1,
            "lambda": lam[k], "target": target}


def _arnoldi_from(apply, v0, m):
    size = v0.size
    V = np.zeros((size, m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = v0 / np.linalg.norm(v0)
    for j in range(m):
        w = apply(V[:, j])
        for _ in range(2):
            for i in range(j + 1):
                c = V[:, i] @ w
                H[i, j] += c
                w = w - c * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            m = j + 1
            break
        V[:, j + 1] = w / H[j + 1, j]
    lam = np.linalg.eigvals(H[:m, :m])
    return lam, None


def reduced(cm):
    """The step's Jacobian restricted to the k = 2 pi family: every field is (cos kx or sin kx) x (a y profile);
    returns the dense matrix ``M`` (8 n + 1) and the relative residual of the restriction ``|J B - B M| / |J B|``."""
    grid = cm.grid
    n = grid.n[0]
    xc = np.asarray(grid.centers(0))
    xf = np.asarray(grid.faces(0))
    cosc, sinc, sinf = np.cos(K * xc), np.sin(K * xc), np.sin(K * xf)
    # (field index in the state, x pattern, number of y entries)
    plan = [(0, cosc, n), (1, sinc, n), (2, cosc, n), (3, sinf, n), (4, cosc, n + 1), (5, cosc, n), (6, sinc, n), (7, cosc, n)]
    offsets = np.cumsum([0] + [int(np.prod(sh)) for sh in cm.shapes])
    cols = []
    for idx, pat, ny in plan:
        for j in range(ny):
            arr = np.zeros(cm.shapes[idx])
            arr[:, j] = pat
            v = np.zeros(cm.q0.size)
            v[offsets[idx]:offsets[idx + 1]] = arr.ravel()
            cols.append(v / np.linalg.norm(v))
    B = np.array(cols).T
    JB = np.array([cm.jvp(B[:, i]) for i in range(B.shape[1])]).T
    M = B.T @ JB
    res = float(np.linalg.norm(JB - B @ M) / np.linalg.norm(JB))
    return M, res


def dispersion(n=32, f=1.0, DT=cw.STEP, **kw):
    """``s_h`` from the reduced Jacobian (the eigenvalue closest to ``exp(s dt)``)."""
    cm = CapwaveMap(n, f, DT, **kw)
    s, _ = mode(K, 1.0, 1.0, cm.nu)
    M, res = reduced(cm)
    lam = np.linalg.eigvals(M)
    tg = cmath.exp(s * cm.dt)
    k = int(np.argmin(np.abs(lam - tg)))
    sh = complex(np.log(lam[k]) / cm.dt)
    return {"s": s, "s_h": sh, "damping_err": sh.real / s.real - 1, "frequency_err": sh.imag / s.imag - 1,
            "residual": res, "dt": cm.dt, "lam": lam}


def semi_discrete(n=32, f=1.0, DT=cw.STEP, **kw):
    """Richardson from DT and DT/2 (first order in time for the split step)."""
    a = dispersion(n, f, DT, **kw)
    b = dispersion(n, f, DT / 2, **kw)
    sh = 2 * b["s_h"] - a["s_h"]
    s = a["s"]
    return {"s": s, "dt": a, "dt2": b, "s_h": sh, "damping_err": sh.real / s.real - 1, "frequency_err": sh.imag / s.imag - 1}


def main() -> None:
    r = eigen(32, 1.0)
    print(r)


if __name__ == "__main__":
    main()

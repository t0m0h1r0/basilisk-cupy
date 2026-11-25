"""Capillary wave started on the exact normal mode (docs/notes/38).

Two identical viscous fluids (rho = 1, sigma = 1, k = 2 pi, the domain and walls of
:mod:`cases.capillary_wave`), started with the velocity of the standing normal mode
(:func:`mode`: ``u = U(y) i sin kx``, ``v = V(y) cos kx``, complex in time) and the interface
``eta = a cos kx``. Without the start-up transient of the initial-value problem the interface
amplitude is a single damped cosine; ``s`` is fitted to it by least squares (:func:`fit`). The amplitude is
the ``cos kx`` Fourier coefficient of the ``psi = 1/2`` crossings.

    python -m cases.capwave_normal_mode
"""

from __future__ import annotations

import cmath
import math
import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases import capillary_wave as cw
from cases.capwave_modes import growth

K = 2.0 * math.pi
W0 = math.sqrt(K ** 3 / 2.0)


def mode(k: float, sigma: float, rho: float, nu: float, a: float = 0.01):
    """``(s, fields)``: ``fields(x, y) -> (u, v, p, lap u, lap v)`` of the standing mode with
    ``eta = a cos kx`` at t = 0 (complex amplitudes; the physical field is the real part)."""
    s = growth(k, sigma, rho, nu)
    mu = rho * nu
    m = cmath.sqrt(k * k + s / nu)
    m = m if m.real > 0 else -m
    i = 1j
    M = np.array([[i * k, -m, -i * k, -m], [-k, -i * k, -k, i * k],
                  [-2 * i * k * k, m * m + k * k, -2 * i * k * k, -(m * m + k * k)],
                  [rho * s + 2 * mu * k * k + sigma * k ** 3 / s, 2 * i * mu * k * m + i * sigma * k ** 3 / s,
                   -(rho * s + 2 * mu * k * k), 2 * i * mu * k * m]], dtype=complex)
    A, B, C, D = np.linalg.svd(M)[2][-1].conj()
    scale = a * s / (-k * A - i * k * B)  # s eta = v(0)
    A, B, C, D = A * scale, B * scale, C * scale, D * scale

    def fields(x, y):
        up = y >= 0
        sx, cx = 1j * np.sin(k * x), np.cos(k * x)
        Ek, Em, Fk, Fm = np.exp(-k * y), np.exp(-m * y), np.exp(k * y), np.exp(m * y)
        q = m * m - k * k
        u = np.where(up, i * k * A * Ek - m * B * Em, i * k * C * Fk + m * D * Fm) * sx
        v = np.where(up, -k * A * Ek - i * k * B * Em, k * C * Fk - i * k * D * Fm) * cx
        p = np.where(up, -rho * s * A * Ek, -rho * s * C * Fk) * cx
        lu = np.where(up, -m * B * Em * q, m * D * Fm * q) * sx
        lv = np.where(up, -i * k * B * Em * q, -i * k * D * Fm * q) * cx
        return u, v, p, lu, lv

    return s, fields


def fit(t, a, s0: complex) -> complex:
    """Least-squares damped cosine ``a = Re(A e^{s t})`` (Gauss-Newton on ``Re s, Im s`` with the complex
    ``A`` solved linearly at every iterate), started from ``s0``."""
    t, a = np.asarray(t), np.asarray(a)

    def resid(g, w):
        e = np.exp(g * t)
        X = np.stack([e * np.cos(w * t), -e * np.sin(w * t)], axis=1)
        c = np.linalg.lstsq(X, a, rcond=None)[0]
        return X @ c - a

    g, w = s0.real, s0.imag
    for _ in range(50):
        r0 = resid(g, w)
        J = np.stack([(resid(g + 1e-6, w) - r0) / 1e-6, (resid(g, w + 1e-6) - r0) / 1e-6], axis=1)
        dg, dw = np.linalg.lstsq(J, -r0, rcond=None)[0]
        g, w = g + dg, w + dw
        if abs(dg) + abs(dw) < 1e-10:
            break
    return complex(g, w)


def run(n, f: float = 1.0, periods: float = 2.0, a: float = 0.01, viscous=None, ops_ns=None,
        jump_mode: str = "sharp", DT: float = cw.STEP, profile: bool = False, budget: bool = False,
        eps_factor: float | None = None, cls_kw: dict | None = None, pressure_jump: bool = True,
        vorticity: bool = False, mapping_y=None, stokes: bool = False, plain: bool | None = None):
    warnings.filterwarnings("ignore")
    nu = cw.MU * f
    s, F = mode(K, 1.0, 1.0, nu, a)
    nx, ny = (n, n) if np.isscalar(n) else n  # (nx, ny): anisotropic cells
    grid = CartesianGrid((nx, ny), length=(2.0, 2.0), origin=(0.0, -1.0),
                         mapping=None if mapping_y is None else (None, mapping_y))
    walls = ("left", "right", "bottom", "top")
    if viscous is None and not grid.is_uniform:
        from basilisk_cupy.terms.viscous import FD2Viscosity

        viscous = FD2Viscosity()  # Basilisk's viscosity() needs square uniform cells
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-10, viscous=viscous, ops=ops_ns,
                    stokes=stokes)
    x, y = grid.meshgrid()
    u, v, _, _, _ = F(x, y)
    eta = a * np.cos(K * x)
    dist = (y - eta) / np.sqrt(1.0 + (a * K * np.sin(K * x)) ** 2)
    from basilisk_cupy.interface.cls import min_width

    hmin = min_width(grid)
    plain = (not grid.is_uniform) if plain is None else plain  # the mapped CLS has the plain divergence only
    cls = CLSInterface.from_level_set(grid, dist, BoundaryConditions(),
                                      divergence="plain" if plain else "fccd",
                                      eps=None if eps_factor is None else eps_factor * hmin,
                                      ops=FCCDFaceComplex(order=4 if plain else 6))
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=nu, mu2=nu, vof_first=True, pressure_jump=pressure_jump,
                     jump_mode=jump_mode, **(cls_kw or {}))
    tp.add_surface_tension(1.0)
    ns.u[0].interior = u.real
    ns.u[1].interior = v.real
    ns.u.fill_ghosts()
    ns.DT = DT
    xc = np.asarray(grid.centers(0))
    w = np.cos(K * xc) * 2.0 / nx

    def amplitude():
        psi = cls.psi.interior
        lo, hi = psi[:, :-1] - 0.5, psi[:, 1:] - 0.5
        cross = (lo * hi <= 0) & (lo != hi)
        t = np.where(cross, lo / np.where(cross, lo - hi, 1.0), 0.0)
        yy = np.broadcast_to(y, psi.shape)
        yc = np.where(cross, yy[:, :-1] + t * (yy[:, 1:] - yy[:, :-1]), np.nan)
        return float(np.sum(w * np.nanmean(yc, axis=1)))

    T = 2 * math.pi / s.imag
    times = [i * DT for i in range(int(periods * T / DT) + 1)]
    dA = grid.cell_volumes()

    def kinetic():
        return 0.5 * sum(float(np.sum(c.interior ** 2 * dA)) for c in ns.u)

    def surface():  # excess surface energy of eta = a cos kx: sigma/2 int eta_x^2 dx = sigma k^2 a^2 L_x / 4
        return K * K * amplitude() ** 2 * 2.0 / 4.0

    def face_area(ax):  # volume attached to a face: width across x centre distance along ax
        from basilisk_cupy.interface.cls import cell_widths, centre_spacing

        return centre_spacing(grid, ax) * cell_widths(grid, 1 - ax) * np.ones(grid.face_shape(ax))

    stages = {"advection": 0.0, "viscous": 0.0, "projection": 0.0, "surface": 0.0, "force_work": 0.0,
              "pressure_work": 0.0, "quadratic": 0.0}
    state = {}
    if budget:
        def mark(name):
            def hook(_ns):
                e = kinetic()
                if name == "viscous_term":
                    stages["advection"] += e - state["k"]
                elif name == "acceleration":
                    stages["viscous"] += e - state["k"]
                elif name == "projection":
                    _ns.u.fill_ghosts()
                    state["Iu"] = [_ns.ops.interp_f(_ns.u[ax], ax).copy() for ax in range(2)]
                    state["a"] = [np.asarray(_ns.a[ax]).copy() * np.ones_like(state["Iu"][ax]) for ax in range(2)]
                    return
                elif name == "end_timestep":
                    stages["projection"] += e - state["k"]
                    dt_ = _ns.clock.dt
                    _ns.p.fill_ghosts()
                    for ax in range(2):  # dt <I u, a - alpha G p> over the faces (interior faces carry the work)
                        Gp = _ns.alpha[ax] * _ns.ops.grad_f(_ns.p, ax)
                        sl = [slice(None)] * 2
                        sl[ax] = slice(1, -1)
                        sl = tuple(sl)
                        fa = face_area(ax)[sl]
                        stages["force_work"] += dt_ * float(np.sum(state["Iu"][ax][sl] * state["a"][ax][sl] * fa))
                        stages["pressure_work"] -= dt_ * float(np.sum(state["Iu"][ax][sl] * Gp[sl] * fa))
                    stages["quadratic"] += 0.5 * dt_ ** 2 * sum(float(np.sum(c.interior ** 2 * dA)) for c in _ns.g)
                    sf = surface()
                    stages["surface"] += sf - state["s"]
                    state["s"] = sf
                    state["Eint"] += 0.5 * (state["E"][-1] + e + sf) * _ns.clock.dt
                    state["E"].append(e + sf)
                state["k"] = e
            return hook
        ns.on("viscous_term", mark("viscous_term"))
        ns.on("acceleration", mark("acceleration"))
        ns.on("projection", mark("projection"))
        ns.on("end_timestep", mark("end_timestep"))
        state.update(k=kinetic(), s=surface(), E=[], Eint=0.0)
        state["E"].append(state["k"] + state["s"])
    ns.start(events=times[1:])
    amps = [amplitude()]
    wsx = np.sin(K * xc)[:, None] * 2.0 / nx
    vort = []

    def omega_k():  # sin kx coefficient of omega = v_x - u_y (cell values, centred differences)
        uu, vv = ns.u[0].interior, ns.u[1].interior
        om = np.gradient(vv, grid.centers(0), axis=0) - np.gradient(uu, grid.centers(1), axis=1)
        return np.sum(om * wsx, axis=0)

    while len(amps) < len(times):
        ns.step()
        if abs(ns.clock.t - times[len(amps)]) <= 1e-9:
            amps.append(amplitude())
            if vorticity:
                vort.append((ns.clock.t, omega_k()))
    sh = fit(times, amps, s)
    out = {"s": s, "s_h": sh, "damping_err": sh.real / s.real - 1, "frequency_err": sh.imag / s.imag - 1,
           "delta_h": math.sqrt(2 * nu / W0) / (2.0 / ny), "amplitudes": amps, "t": times}
    if vorticity:
        out["vorticity"] = vort
        out["y"] = y[0]
    if budget:  # damping rate of each stage: gamma = -sum(dE) / (2 int E dt); the exact rate is -Re s
        Eint = state["Eint"]
        out["budget"] = {k: -v / (2.0 * Eint) for k, v in stages.items()}
        out["budget"]["total"] = -(state["E"][-1] - state["E"][0]) / (2.0 * Eint)
        out["energy"] = state["E"]
    if profile:
        t_end = ns.clock.t
        yy = y[0]
        sx = np.sin(K * xc)[:, None] * 2.0 / nx
        cx = np.cos(K * xc)[:, None] * 2.0 / nx
        ue, ve, _, _, _ = F(x, y)
        e = cmath.exp(s * t_end)
        out["profile"] = {"y": yy, "u": np.sum(ns.u[0].interior * sx, axis=0), "v": np.sum(ns.u[1].interior * cx, axis=0),
                          "u_exact": np.sum((ue * e).real * sx, axis=0), "v_exact": np.sum((ve * e).real * cx, axis=0)}
    return out


def main() -> None:
    from basilisk_cupy.terms.viscous import CCDViscosity

    for n, f in ((32, 0.5), (32, 1.0), (64, 1.0)):
        for name, kw in (("fd2", {}), ("ccd", {"viscous": CCDViscosity()}),
                         ("ccd + fccd, smooth", {"viscous": CCDViscosity(), "ops_ns": FCCDFaceComplex(),
                                                 "jump_mode": "smooth"})):
            r = run(n, f, **kw)
            print(f"N={n} f={f} delta/h {r['delta_h']:.2f} {name:20s} damping err {r['damping_err']:+.4f} "
                  f"frequency err {r['frequency_err']:+.4f}", flush=True)


if __name__ == "__main__":
    main()

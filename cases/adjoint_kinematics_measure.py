"""Measurements for docs/notes/45: work mismatch and split of Delta F = F_adj - F_GFM.

    python -m cases.adjoint_kinematics_measure
"""

from __future__ import annotations

import warnings

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.discretization.ccd import FCCDFaceComplex
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS
from cases import capillary_wave as cw
from cases.adjoint_kinematics import adjoint_force, from_face_field, gfm_face_force, split, to_face_field, transport_matrix
from cases.capwave_normal_mode import K, mode


def capwave_state(n, f=1.0, a=0.01, curvature="psi"):
    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-12)
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, (y - a * np.cos(K * x)) / np.sqrt(1 + (a * K * np.sin(K * x)) ** 2),
                                      BoundaryConditions(), divergence="fccd", ops=FCCDFaceComplex(order=6))
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=cw.MU * f, mu2=cw.MU * f, pressure_jump=True)
    tp.add_surface_tension(1.0)
    if curvature == "exact":
        from cases.capwave_jacobian import exact_linear_curvature

        kp = cls.curvature()
        ke = exact_linear_curvature(tp)
        band = kp != 0
        sgn = float(np.sign(np.sum(kp[band] * ke[band])))
        tp.potentials[0] = lambda t_: t_.sigma * sgn * exact_linear_curvature(t_)
    s, F = mode(K, 1.0, 1.0, cw.MU * f, a)
    u, v, _, _, _ = F(x, y)
    # the phase with the largest interface velocity: imaginary part; faces by averaging, then projected
    ns.u[0].interior, ns.u[1].interior = u.imag, v.imag
    ns.u.fill_ghosts()
    uf = ns.face_velocity()
    from basilisk_cupy import CellField
    from basilisk_cupy.projection.approximate import project

    p = CellField(grid, bc=ns.p.bc, name="p")
    alpha = FaceField(grid, [np.ones(grid.face_shape(k)) for k in range(2)], "alpha")
    project(uf, p, alpha, 1.0, tolerance=1e-13)
    return grid, cls, tp, uf


def work_split(grid, cls, tp, uf, kinematics):
    T, faces = transport_matrix(cls, kinematics)
    Fa, _ = adjoint_force(cls, T)
    Fg = gfm_face_force(tp, faces)
    u = from_face_field(uf, faces)
    Vf = float(grid.h[0] * grid.h[1])
    Pa, Pg = u @ Fa * Vf, u @ Fg * Vf
    dF = Fa - Fg
    grad, sol = split(dF, grid, faces)
    gg, gs = split(Fg, grid, faces)
    return {"P_adj": Pa, "P_gfm": Pg, "mismatch": Pg / Pa - 1, "dF": np.linalg.norm(dF) / np.linalg.norm(Fg),
            "dF_sol": np.linalg.norm(sol) / np.linalg.norm(Fg), "dF_grad": np.linalg.norm(grad) / np.linalg.norm(Fg),
            "work_sol": u @ sol * Vf / Pa, "gfm_sol": np.linalg.norm(gs) / np.linalg.norm(Fg),
            "adj_sol": np.linalg.norm(split(Fa, grid, faces)[1]) / np.linalg.norm(Fa)}


def drop_state(n, R=0.25):
    warnings.filterwarnings("ignore")
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-12)
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, R - np.sqrt(x * x + y * y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1, pressure_jump=True)
    tp.add_surface_tension(1.0)
    return grid, cls, tp


def main() -> None:
    for n in (32, 64):
        for curv in ("psi", "exact"):
            grid, cls, tp, uf = capwave_state(n, curvature=curv)
            for kin in ("transport", "extension"):
                r = work_split(grid, cls, tp, uf, kin)
                print(f"capwave N={n} {curv:5s} {kin:9s}: work GFM / adjoint - 1 = {r['mismatch']:+.4f}  |dF|/|F_GFM| {r['dF']:.3f} "
                      f"(grad {r['dF_grad']:.3f}, div-free {r['dF_sol']:.3f}); work of the div-free part / adjoint work "
                      f"{r['work_sol']:+.4f}", flush=True)
    for n in (32, 64):
        grid, cls, tp = drop_state(n)
        from cases.adjoint_kinematics import adjoint_force, gfm_face_force, split, transport_matrix

        for kin in ("transport", "extension"):
            T, faces = transport_matrix(cls, kin)
            Fa, _ = adjoint_force(cls, T)
            Fg = gfm_face_force(tp, faces)
            print(f"static drop N={n} {kin:9s}: div-free share |P F|/|F|: GFM {np.linalg.norm(split(Fg, grid, faces)[1]) / np.linalg.norm(Fg):.2e}"
                  f"  adjoint {np.linalg.norm(split(Fa, grid, faces)[1]) / np.linalg.norm(Fa):.2e}", flush=True)


if __name__ == "__main__":
    main()


def damping_prediction(n, f=1.0, curvature="psi", kinematics="transport", a=0.01, energy="area"):
    """``delta gamma / gamma`` predicted from the work mismatch: the cycle average of
    ``B(u, eta) = sum uf (F_GFM - F_adj) V`` is ``B(Re u_hat) / 2`` (eta_hat = a real), the mean energy is twice the
    mean kinetic energy, ``delta gamma = -<B> / (2 E)``. Also the reactive ratio ``B(Im u_hat) / P_adj(Im u_hat)``."""
    out = {}
    for phase in ("re", "im"):
        warnings.filterwarnings("ignore")
        grid = CartesianGrid((n, n), length=(2.0, 2.0), origin=(0.0, -1.0))
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-12)
        x, y = grid.meshgrid()
        cls = CLSInterface.from_level_set(grid, (y - a * np.cos(K * x)) / np.sqrt(1 + (a * K * np.sin(K * x)) ** 2),
                                          BoundaryConditions(), divergence="fccd", ops=FCCDFaceComplex(order=6))
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=cw.MU * f, mu2=cw.MU * f, pressure_jump=True)
        tp.add_surface_tension(1.0)
        if curvature == "exact":
            from cases.capwave_jacobian import exact_linear_curvature

            kp = cls.curvature()
            ke = exact_linear_curvature(tp)
            band = kp != 0
            sgn = float(np.sign(np.sum(kp[band] * ke[band])))
            tp.potentials[0] = lambda t_: t_.sigma * sgn * exact_linear_curvature(t_)
        s, F = mode(K, 1.0, 1.0, cw.MU * f, a)
        u, v, _, _, _ = F(x, y)
        part = np.real if phase == "re" else np.imag
        ns.u[0].interior, ns.u[1].interior = part(u), part(v)
        ns.u.fill_ghosts()
        uf = ns.face_velocity()
        from basilisk_cupy import CellField
        from basilisk_cupy.projection.approximate import project

        p = CellField(grid, bc=ns.p.bc, name="p")
        alpha = FaceField(grid, [np.ones(grid.face_shape(k)) for k in range(2)], "alpha")
        project(uf, p, alpha, 1.0, tolerance=1e-13)
        T, faces = transport_matrix(cls, kinematics)
        if energy == "sharp":
            from cases.adjoint_kinematics import sharp_energy_force

            Fa, _, _ = sharp_energy_force(cls, T)
        else:
            Fa, _ = adjoint_force(cls, T)
        Fg = gfm_face_force(tp, faces)
        uu = from_face_field(uf, faces)
        Vf = float(grid.h[0] * grid.h[1])
        out[phase] = (uu @ (Fg - Fa) * Vf, uu @ Fa * Vf)
        if phase == "re":
            Ekin = 0.25 * float(np.sum(np.abs(u) ** 2 + np.abs(v) ** 2)) * Vf  # time-mean kinetic energy
    gamma = -s.real
    B_re = out["re"][0]
    dgamma = -0.5 * B_re / (2.0 * 2.0 * Ekin)  # <B> = B_re / 2, E_mean = 2 E_kin_mean
    return {"dgamma_over_gamma": dgamma / gamma, "reactive": out["im"][0] / out["im"][1]}


def main_prediction(energy="area", curvs=("psi", "exact")) -> None:
    for n in (32, 64):
        for curv in curvs:
            for kin in ("transport", "extension"):
                r = damping_prediction(n, curvature=curv, kinematics=kin, energy=energy)
                print(f"N={n} {curv:5s} {kin:9s}: predicted damping change from the work mismatch {r['dgamma_over_gamma']:+.4f}"
                      f"   reactive mismatch {r['reactive']:+.4f}", flush=True)

"""TwoPhaseIMEX: the paper's standard two-phase step as one combination (docs/notes/50)."""

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX
from cases.two_phase_imex_check import capwave, drop


@pytest.mark.parametrize("rho2,jump,u_end,u_max", [
    (1.0, 4.00977, 1.074e-4, 3.674e-4),
    (0.001, 4.01030, 1.171e-4, 6.758e-4),  # density ratio 1000 at tension.h's step x sqrt(5/12)
])
def test_static_drop(rho2, jump, u_end, u_max):
    """R = 0.25, sigma = 1, mu = 0.1, N = 32, t = 0.1 (60 / 81 steps): Laplace jump 4, mass to round-off."""
    r = drop(32, rho2, "imex", t_end=0.1)
    assert r["t"] == pytest.approx(0.1)
    assert r["jump"] == pytest.approx(jump, abs=1e-4)
    assert r["u_end"] == pytest.approx(u_end, rel=0.03) and r["u_max"] == pytest.approx(u_max, rel=0.03)
    assert abs(r["mass"]) < 1e-13


@pytest.mark.parametrize("interface,rho2,jump,u_end,u_max,steps", [
    ("vof", 1.0, 4.04118, 2.789e-4, 4.475e-3, 42),        # tension.h's CSF (height-function curvature)
    ("vof", 0.001, 4.04111, 1.106e-4, 5.751e-3, 56),
    ("vof-gfm", 1.0, 4.04090, 2.908e-4, 4.820e-3, 42),    # the same curvature as a sharp jump (docs/notes/64)
    ("vof-gfm", 0.001, 4.04072, 2.189e-4, 7.325e-3, 56),
    ("vof-gfm-plic", 1.0, 4.04093, 2.662e-4, 4.790e-3, 42),  # crossings from the PLIC lines
    ("vof-gfm-plic", 0.001, 4.04059, 1.941e-4, 6.910e-3, 56),
    ("vof-gfm-pccd", 1.0, 4.04098, 2.936e-4, 5.286e-3, 42),  # phase-wise CCD faces, no Taylor terms (notes/67)
    ("vof-gfm-pccd", 0.001, 4.04372, 8.297e-4, 8.361e-3, 56),
])
def test_static_drop_vof(interface, rho2, jump, u_end, u_max, steps):
    """The VOF interface on the IMEX stepper (fd2 viscosity): the jump is the heights' curvature at N = 32 (1 %),
    mass to round-off."""
    kw = {"vof_crossing": "plic"} if interface.endswith("plic") else {}
    if interface.endswith("pccd"):
        kw = {"faces": "phase-ccd"}
    r = drop(32, rho2, "imex", t_end=0.1, interface=interface.removesuffix("-plic").removesuffix("-pccd"), **kw)
    assert r["t"] == pytest.approx(0.1) and r["steps"] == steps
    assert r["jump"] == pytest.approx(jump, abs=1e-4)
    assert r["u_end"] == pytest.approx(u_end, rel=0.05) and r["u_max"] == pytest.approx(u_max, rel=0.05)
    # the phase-ccd projection leaves an fd2 divergence that the transport's reprojection removes to its tolerance
    assert abs(r["mass"]) < (1e-10 if interface.endswith("pccd") else 1e-13)


@pytest.mark.parametrize("transport,error,damping", [
    ("projected", 0.20320, -0.3601),  # the paper's stage 1: psi moved by u_f^n, first order in time
    ("ab2", 0.19070, -0.3011),        # midpoint velocity (1 + w/2) u_f^n - (w/2) u_f^{n-1}
])
def test_capillary_wave(transport, error, damping):
    """capwave.c, 8 points per wavelength (CenteredNS + CLS/GFM: RMS error 0.1866, damping -28.7 %)."""
    r = capwave(16, "imex", transport=transport)
    assert r["error"] == pytest.approx(error, rel=1e-3)
    assert r["damping"] == pytest.approx(damping, abs=1e-3)
    assert r["steps"] == 738


def test_options():
    grid = CartesianGrid((8, 8), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    phi = 0.25 - np.hypot(x, y)
    with pytest.raises(NotImplementedError):
        TwoPhaseIMEX(grid, phi, rho2=0.1, faces="fccd")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, faces="ccd")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, convection="weno")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, transport="rk2")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, interface="level-set")
    with pytest.raises(NotImplementedError):  # phase-ccd's segments end at the sharp jump's cut faces
        TwoPhaseIMEX(grid, phi, interface="vof", faces="phase-ccd")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, interface="vof-gfm", vof_crossing="linear")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, interface="vof-gfm", cell_coefficient="harmonic")
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, cell_coefficient="harmonic")
    assert TwoPhaseIMEX(grid, phi, interface="vof-gfm").tp.cell_coefficient == "mean"
    assert TwoPhaseIMEX(grid, phi).tp.cell_coefficient == "gfm"
    ns = TwoPhaseIMEX(grid, phi, rho1=1.0, rho2=0.5, sigma=1.0, periodic=(0,))
    assert ns.uf_normal == {"bottom": 0.0, "top": 0.0}
    assert ns.cls.psi.bc.is_periodic(0) and not ns.cls.psi.bc.is_periodic(1)


def test_vof_phase_ccd_transport_velocity():
    """With phase-wise CCD faces the projected u_f is free of FCCD's divergence, not of fd2's; the VOF's transport
    velocity is projected once more with fd2's GFM operator (docs/notes/67)."""
    from basilisk_cupy.discretization.fd2 import FD2FaceComplex

    grid = CartesianGrid((32, 32), origin=(-0.5, -0.5))
    ns = TwoPhaseIMEX(grid, lambda x, y: 0.0625 - x * x - y * y, rho1=1.0, rho2=0.001, mu1=0.1, mu2=0.1, sigma=1.0,
                      interface="vof-gfm", faces="phase-ccd", transport="ab2", tolerance=1e-8)
    seen = []
    project = ns._fd2_project

    def spy(uf):
        project(uf)
        seen.append(float(np.abs(np.asarray(FD2FaceComplex().div_f(uf))).max()))

    ns._fd2_project = spy
    ns.start()
    for _ in range(4):
        ns.step()
    raw = float(np.abs(np.asarray(FD2FaceComplex().div_f(ns.uf))).max())
    assert raw > 1e-6                    # the momentum's u_f: fd2 divergence left by the phase-ccd projection
    assert seen and max(seen) < 1e-3 * raw  # the transport velocity's

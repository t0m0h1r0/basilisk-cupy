"""Two-phase Navier--Stokes with the CLS interface (TwoPhaseCLS) against the VOF baseline."""

import math

import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import CCD_CAPILLARY_DT_FACTOR, TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS


def _static_drop(n, backend, steps=10, rho2=1.0, **kw):
    kw.setdefault("capillary_dt_factor", 1.0)  # the numbers of notes 15-18 (tension.h's bound)
    grid = CartesianGrid((n, n), origin=(-0.5, -0.5), backend=backend)
    xp = grid.xp
    walls = ("left", "right", "bottom", "top")
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in walls}, tolerance=1e-8)
    x, y = grid.meshgrid()
    r = xp.sqrt(x * x + y * y)
    cls = CLSInterface.from_level_set(grid, 0.25 - r, BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=0.1, mu2=0.1, **kw)
    tp.add_surface_tension(1.0)
    m0 = cls.mass()
    ns.start()
    for _ in range(steps):
        ns.step()
    jump = float(xp.mean(ns.p.interior[r < 0.1]) - xp.mean(ns.p.interior[r > 0.4]))
    umax = max(float(xp.max(xp.abs(c.interior))) for c in ns.u)
    return jump, umax, cls.mass() / m0 - 1


@pytest.mark.parametrize("kw,jump,umax", [
    ({}, 4.0230, 3.159e-4),
    ({"curvature_kw": {"psi_min": 1e-4}}, 4.1299, 5.168e-5),
    ({"reinit_every": 5, "reinit_method": "xi_sdf"}, None, None),
])
def test_static_drop(backend, kw, jump, umax):
    """R = 0.25, sigma = 1, mu = 0.1, N = 64, 10 steps. The jump sigma/R = 4 (VOF: 4.0138, parasitic
    velocity 4.2e-3). With the paper's band psi_min = 0.01 the CSF jump keeps 98% of sum grad psi:
    4.0230 / 3.9410 / 3.9228 at N = 64 / 128 / 256; psi_min = 1e-4: 4.1299 / 4.0298 / 4.0067."""
    j, u, dm = _static_drop(64, backend, **kw)
    assert abs(dm) < 1e-12 and math.isfinite(u)
    if jump is not None:
        assert j == pytest.approx(jump, rel=1e-3) and u == pytest.approx(umax, rel=0.05)
    else:
        assert j == pytest.approx(4.0, rel=0.02)


def test_density_ratio(backend):
    """rho2 = 0.1 (light outer phase): the jump is unchanged (no gravity)."""
    j, u, _ = _static_drop(64, backend, rho2=0.1)
    assert j == pytest.approx(4.0230, rel=2e-3) and u < 1e-2


def test_capillary_wave(backend):
    """Basilisk's capwave.c with the CLS interface, 8 points per wavelength: relative RMS error 0.355
    (VOF: 0.1386). 16 / 32 points: 0.315 / 0.236 (VOF 0.0242 / 0.0064). The CSF force spread over
    eps = 1.5 Delta slows the wave: half period 4.79 / 4.05 / 3.70 against Prosperetti's 3.456 (in
    omega_0 t) at 8 / 16 / 32 points per wavelength."""
    from cases import capillary_wave as cw

    r = cw.run(16, backend, interface="cls", capillary_dt_factor=1.0)
    assert r["error"] == pytest.approx(0.35541, rel=1e-3)


@pytest.mark.parametrize("n,jump,umax", [(32, 4.0091, 3.330e-4), (64, 4.0027, 4.873e-5), (128, 4.0010, 8.574e-6)])
def test_static_drop_pressure_jump(backend, n, jump, umax):
    """GFM pressure jump (pressure_jump=True), 10 steps: the jump converges to sigma/R = 4 (CSF:
    4.3014 / 4.0230 / 3.9410) and the parasitic velocity falls like h^2.5 (CSF 3.6e-4 / 3.2e-4 / 1.7e-4)."""
    j, u, dm = _static_drop(n, backend, pressure_jump=True)
    assert j == pytest.approx(jump, abs=2e-4) and u == pytest.approx(umax, rel=0.05) and abs(dm) < 1e-12


def test_static_drop_pressure_jump_reaches_rest(backend):
    """N = 64, 400 steps: parasitic velocity 1.1e-8 (CSF 2.4e-4; VOF 6.1e-8); density ratio 10 at 400 steps: 1.6e-7."""
    assert _static_drop(64, backend, steps=400, pressure_jump=True)[1] < 2e-8
    assert _static_drop(64, backend, steps=400, rho2=0.1, pressure_jump=True)[1] < 3e-7


def test_capillary_wave_pressure_jump(backend):
    """capwave.c with the GFM jump, 8 / 16 / 32 / 64 points per wavelength: 0.201 / 0.0933 / 0.0279 /
    0.0130 (CSF 0.355 / 0.315 / 0.236; VOF 0.139 / 0.0242 / 0.0064 / 0.0023); half period 3.629 / 3.507
    at 16 / 32 points (Prosperetti 3.456; CSF 4.05 / 3.70)."""
    from cases import capillary_wave as cw

    r = cw.run(16, backend, interface="cls", pressure_jump=True, capillary_dt_factor=1.0)
    assert r["error"] == pytest.approx(0.20122, rel=1e-3)


def test_capillary_wave_pressure_jump_corrected_divergence(backend):
    """GFM jump + corrected divergence + 6th-order face values, 8 / 16 / 32 / 64 points per wavelength:
    0.188 / 0.0639 / 0.0109 / 0.00209 (plain divergence 0.201 / 0.0933 / 0.0279 / 0.0130; VOF 0.139 /
    0.0242 / 0.0064 / 0.0023). 4th-order face values: 0.188 / 0.0654 / 0.0126 / 0.00348."""
    from cases import capillary_wave as cw

    r = cw.run(32, backend, interface="cls", pressure_jump=True, divergence="fccd", face_order=6, capillary_dt_factor=1.0)
    assert r["error"] == pytest.approx(0.06389, rel=1e-3)


@pytest.mark.parametrize("rho2,factor,bound", [(0.01, 1.0, None), (0.01, None, 1e-6), (0.001, None, 1e-6)])
def test_static_drop_high_density_ratio(backend, rho2, factor, bound):
    """GFM jump, equal viscosities 0.1, N = 64, 400 steps. With tension.h's time step the capillary loop
    grows at density ratios 100 and 1000 (8.7e-4 and 2.7 here): the CCD curvature answers a grid-scale
    wiggle with 48/5 instead of 4 per Delta^2. capillary_dt_factor = sqrt(5/12): 4.4e-7 (ratio 1000)."""
    f = CCD_CAPILLARY_DT_FACTOR if factor is None else factor
    _, u, dm = _static_drop(64, backend, steps=400, rho2=rho2, pressure_jump=True, capillary_dt_factor=f)
    assert abs(dm) < 1e-12
    if bound is None:
        assert not (u < 1e-4)  # grows (or overflows)
    else:
        assert u < bound


def test_default_capillary_factor(backend):
    """TwoPhaseCLS uses sqrt(5/12) of tension.h's capillary bound unless told otherwise."""
    grid = CartesianGrid((16, 16), origin=(-0.5, -0.5), backend=backend)
    ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")})
    x, y = grid.meshgrid()
    cls = CLSInterface.from_level_set(grid, 0.25 - grid.xp.sqrt(x * x + y * y), BoundaryConditions())
    tp = TwoPhaseCLS(ns, cls, pressure_jump=True)
    assert tp.capillary_dt_factor == pytest.approx(math.sqrt(5.0 / 12.0), rel=1e-15)

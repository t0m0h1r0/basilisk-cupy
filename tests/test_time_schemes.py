"""Multistep IMEX schemes as data and the generic stepper (docs/notes/51)."""

import math

import numpy as np
import pytest

from basilisk_cupy import CartesianGrid
from basilisk_cupy.time.imex_bdf2 import IMEXBDF2NS
from basilisk_cupy.time.imex_multistep import IMEXMultistepNS
from basilisk_cupy.time.schemes import MultistepScheme, ab_beta, am_gamma, bdf_alpha, explicit_rk, ext_beta
from basilisk_cupy.time.two_phase_imex import TwoPhaseIMEX


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_lagrange_weights_are_exact_on_polynomials(k):
    """Variable offsets (steps 1, 0.7, 1.3, 0.9): every family is exact for polynomials of its degree."""
    steps = [1.0, 0.7, 1.3, 0.9]
    offsets = [sum(steps[:i + 1]) for i in range(4)]
    nodes = np.array([0.0] + [-s for s in offsets])
    for m in range(k + 1):
        f = nodes ** m
        assert np.dot(bdf_alpha(k, offsets), f[:k + 1]) == pytest.approx(m * 0.0 ** max(m - 1, 0) if m else 0.0,
                                                                         abs=1e-11)
    for m in range(k):
        f = nodes ** m
        integral = (0.0 - (-1.0) ** (m + 1)) / (m + 1)
        assert np.dot(ext_beta(k, offsets), f[1:k + 1]) == pytest.approx(1.0 if m == 0 else 0.0, abs=1e-11)
        assert np.dot(ab_beta(k, offsets), f[1:k + 1]) == pytest.approx(integral, abs=1e-11)
        assert np.dot(am_gamma(k, offsets), f[:k]) == pytest.approx(integral, abs=1e-11)


def test_named_schemes_and_startup():
    c = MultistepScheme("sbdf2").coefficients([1.0, 1.0])
    assert c.alpha == pytest.approx((1.5, -2.0, 0.5)) and c.beta == pytest.approx((2.0, -1.0))
    w = 0.5  # variable step: IMEXBDF2NS's gamma = (1 + w)/(1 + 2w), EXT2 (1 + w, -w)
    c = MultistepScheme("sbdf2").coefficients([1.0, 1.0 / w])
    assert 1.0 / c.alpha[0] == pytest.approx((1 + w) / (1 + 2 * w)) and c.beta == pytest.approx((1 + w, -w))
    assert MultistepScheme("sbdf3").coefficients([1.0]).alpha == pytest.approx((1.0, -1.0))  # first step: sbdf1
    assert MultistepScheme("sbdf3").coefficients([1.0, 1.0]).alpha == pytest.approx((1.5, -2.0, 0.5))
    c = MultistepScheme("cnab2").coefficients([1.0, 1.0])
    assert c.gamma == pytest.approx((0.5, 0.5)) and c.beta == pytest.approx((1.5, -0.5))
    assert MultistepScheme("cnab2").coefficients([1.0], v_levels=0).gamma == pytest.approx((1.0,))
    assert MultistepScheme("mcnab2").coefficients([1.0, 1.0], 2).gamma == pytest.approx((9 / 16, 3 / 8, 1 / 16))
    assert MultistepScheme("mcnab2").coefficients([1.0, 1.0], 1).gamma == pytest.approx((0.5, 0.5))  # needs V^{n-1}
    assert MultistepScheme("mcnab2").coefficients([1.0], 0).gamma == pytest.approx((1.0,))  # startup
    with pytest.raises(ValueError):
        MultistepScheme("mcnab2").coefficients([1.0, 0.5], 2)  # constant-step coefficients only
    for bad in ("sbdf5", ("ext2", "bdf2"), ("bdf2", "am2"), ("theta0", "ab1"), "rk4"):
        with pytest.raises(ValueError):
            MultistepScheme(bad)
    with pytest.raises(ValueError):
        explicit_rk("rk7")


def _tg(cls, **kw):
    grid = CartesianGrid((16, 16), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    ns = cls(grid, periodic=(0, 1), mu=0.01, tolerance=1e-12, **kw)
    ns.u[0].interior = -np.cos(2 * math.pi * x) * np.sin(4 * math.pi * y)
    ns.u[1].interior = 0.5 * np.sin(2 * math.pi * x) * np.cos(4 * math.pi * y)
    ns.start(events=[0.1])
    ns.run(0.1)
    return np.stack([c.interior for c in ns.u])


def test_sbdf2_reproduces_imex_bdf2():
    """Adaptive (CFL) variable steps included."""
    a, b = _tg(IMEXBDF2NS), _tg(IMEXMultistepNS, scheme="sbdf2")
    assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("scheme,rate", [("sbdf1", 1.0), ("sbdf2", 2.0), ("cnab2", 2.0), ("mcnab2", 2.0),
                                         ("sbdf3", 2.0)])
def test_temporal_order_stokes(scheme, rate):
    """Stokes, N = 32, kx != ky: every scheme at most 2nd order (sbdf3: the incremental pressure correction's
    splitting error; its constant is 6.5 times sbdf2's). Rates measured 0.99 / 2.06 / 1.95 / 1.94 / 1.97."""
    from cases.time_order import run

    ref = run(scheme, n=32, steps=320, stokes=True)
    e = [float(np.abs(run(scheme, n=32, steps=k, stokes=True) - ref).max()) for k in (20, 40)]
    assert math.log2(e[0] / e[1]) == pytest.approx(rate, abs=0.15)


def test_two_phase_transport_options():
    grid = CartesianGrid((16, 16), origin=(-0.5, -0.5))
    x, y = grid.meshgrid()
    phi = 0.25 - np.hypot(x, y) + 0.02 * np.cos(2 * np.arctan2(y, x))
    out = {}
    for tr, integ in (("frozen", "ssprk3"), ("ab2", "ssprk3"), ("ab3", "ssprk3"), ("pc", "ssprk3"), ("frozen", "rk4"),
                      ("frozen", "euler")):
        ns = TwoPhaseIMEX(grid, phi, mu1=0.1, mu2=0.1, sigma=1.0, transport=tr, psi_integrator=integ,
                          scheme="cnab2", fixed_dt=2e-3)
        m0 = ns.mass()
        ns.start()
        for _ in range(6):
            ns.step()
        assert ns.mass() == pytest.approx(m0, rel=1e-13) and ns.clock.i == 6
        out[(tr, integ)] = ns.psi.copy()
    base = out[("frozen", "ssprk3")]
    for key, psi in out.items():  # every option moves the interface differently, by a small amount
        if key != ("frozen", "ssprk3"):
            assert 0.0 < np.abs(psi - base).max() < 1e-2
    with pytest.raises(ValueError):
        TwoPhaseIMEX(grid, phi, psi_integrator="rk7")

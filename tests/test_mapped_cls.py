"""CLS and the sharp GFM jump on a grid stretched in y (docs/notes/39)."""

import sys

import numpy as np
import pytest

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.grid.mapping import MonitorMapping
from basilisk_cupy.interface.cls import CLSInterface, cell_widths, centre_spacing, min_width
from cases.capwave_normal_mode import run as wave
from cases.mapped_static_drop import band, run as drop

pytestmark = pytest.mark.xdist_group("mapped_cls")


def test_identity_map_reproduces_the_uniform_grid():
    """A monitor of 1 is a uniform grid sent through the mapped code paths (widths, centre spacings, volumes)."""
    flat = MonitorMapping(lambda s: np.ones_like(s))
    a = wave(16, 4.0, periods=0.3, stokes=True, plain=True)
    b = wave(16, 4.0, periods=0.3, stokes=True, plain=True, mapping_y=flat)
    # identical up to the solver tolerances (Basilisk viscosity vs FD2 defect correction, point vs line multigrid)
    assert np.max(np.abs(np.subtract(a["amplitudes"], b["amplitudes"]))) < 1e-7 * np.max(np.abs(a["amplitudes"]))  # 1.1e-8


def test_geometry_helpers():
    g = CartesianGrid((16, 16), origin=(-0.5, -0.5), mapping=(None, band(4, 0.5)))
    w = cell_widths(g, 1).ravel()
    assert w.sum() == pytest.approx(1.0) and min_width(g) == pytest.approx(w.min())
    d = centre_spacing(g, 1).ravel()
    c = np.asarray(g.centers(1))
    assert np.allclose(d[1:-1], np.diff(c)) and d[0] == pytest.approx(2 * (c[0] + 0.5))  # mirrored ghost
    assert cell_widths(CartesianGrid((8, 8)), 0) == pytest.approx(1 / 8)


def test_mass_is_kept_and_the_jump_is_right_on_a_stretched_grid():
    r = drop(24, 2, steps=20)
    assert r["mass"] < 1e-12
    assert r["jump"] == pytest.approx(4.0, rel=0.02)
    assert max(r["umax"]) < 2e-3


def test_refining_the_layer_reduces_the_damping_error():
    """Same 24 x 24 cells: cells 4 times finer about the interface (y = 0) cut the error (delta/h 1.4 -> 3.4)."""
    poisson = sys.modules["basilisk_cupy.linear.poisson"]
    saved = poisson.LINES_MAPPED_ONLY
    poisson.LINES_MAPPED_ONLY = True
    try:
        u = wave((24, 24), 2.0, periods=0.6, stokes=True, plain=True)
        m = wave((24, 24), 2.0, periods=0.6, stokes=True, plain=True, mapping_y=band(4, 0.5))
    finally:
        poisson.LINES_MAPPED_ONLY = saved
    assert abs(m["damping_err"]) < 0.5 * abs(u["damping_err"])


def test_unported_features_refuse_mapped_grids():
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.time.basilisk_centered import CenteredNS
    from basilisk_cupy.terms.viscous import FD2Viscosity

    g = CartesianGrid((16, 16), origin=(-0.5, -0.5), mapping=(None, band(2, 0.5)))
    x, y = g.meshgrid()
    with pytest.raises(NotImplementedError):
        CLSInterface.from_level_set(g, 0.25 - np.hypot(x, y), BoundaryConditions(), divergence="fccd")
    cls = CLSInterface.from_level_set(g, 0.25 - np.hypot(x, y), BoundaryConditions())
    with pytest.raises(NotImplementedError):
        cls.reinitialize()
    ns = CenteredNS(g, viscous=FD2Viscosity(), stokes=True)
    with pytest.raises(NotImplementedError):
        TwoPhaseCLS(ns, cls, pressure_jump=True, jump_mode="smooth")

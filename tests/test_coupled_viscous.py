"""CenteredNS(coupled_iterations=K): viscosity, new force and projection solved together (docs/notes/34)."""

import pytest

from cases.capillary_dt_scan import growth


@pytest.mark.parametrize("K,stable", [(0, False), (1, True)])
def test_viscosity_stabilises_with_the_coupled_iteration(K, stable):
    """Static drop (GFM, equal densities, mu = 0.1, N = 64) at 1.6 times tension.h's step: Basilisk's
    splitting grows (limit 1.15-1.20), one coupled iteration decays to 1.4e-8 (limit 2.09-2.31)."""
    r = growth(64, 1.0, 1.6, steps=400, mu=0.1, ns_kw={"coupled_iterations": K})
    assert r["stable"] == stable


def test_well_balanced():
    """Static drop at tension.h's step, 10 steps: jump 4.00281 and max |u| 4.19e-5 (K = 0: 4.00269, 4.87e-5)."""
    exec_ns = {}
    src = open("tests/test_cls_two_phase.py").read().split("@pytest")[0]
    exec(src, exec_ns)
    import basilisk_cupy.time.basilisk_centered as bc

    orig = bc.CenteredNS.__init__

    def init(self, *a, **k):
        k.setdefault("coupled_iterations", 1)
        orig(self, *a, **k)

    bc.CenteredNS.__init__ = init
    try:
        j, u, dm = exec_ns["_static_drop"](64, "numpy", pressure_jump=True, capillary_dt_factor=1.0)
    finally:
        bc.CenteredNS.__init__ = orig
    assert j == pytest.approx(4.00281, abs=2e-5) and u == pytest.approx(4.188e-5, rel=0.03)

"""docs/notes/110: transport with moving phase-aware control volumes (1-D, exact translation)."""

from cases.mac_transport import run, scenarios


def _scn(N, ratio, name):
    return next(s for s in scenarios(N, ratio) if s[0] == name)


def test_mass_transfer_free_stream_and_tangential_equivalence_across_the_switch():
    N, r = 32, 1e6
    _, a, b, dl = _scn(N, r, "cross the switch")
    o = run(N, r, "D0m", "phase", a, b, dl)
    for c in ("t", "n"):
        x = o[c]
        assert x["mass_row"] < 1e-12 and x["mass_col"] < 1e-12            # m_ij balances, including the re-assignment
        assert x["free_gal"] < 1e-12 and x["free_rem"] < 1e-12            # free stream
        assert x["amp_gal"] < 1 + 1e-9 and x["prod_gal"] < 1 + 1e-9       # theorem 4
    assert o["t"]["mass_jump"] > 1e11                                     # the control-volume mass changes by ~ r^2
    # tangential: Galerkin and the momentum Remap coincide (the transform is an isometry)
    assert o["t"]["defect_gal"] < 1e-8 and o["t"]["amp_rem"] < 1 + 1e-9
    # normal: they do not; the Remap amplification stays bounded
    assert o["n"]["defect_gal"] > 0.05 and o["n"]["amp_rem"] < 1.6


def test_remap_amplification_does_not_compound_with_matching_volumes_but_explodes_with_dual_cells():
    N, r = 32, 1e3
    _, a, b, dl = _scn(N, r, "translate CFL 0.37")
    good = run(N, r, "D0m", "phase", a, b, dl)
    assert good["n"]["prod_rem"] < 1.4 and good["t"]["prod_rem"] < 1 + 1e-9
    assert good["n"]["E_rem_max"] < 1 + 1e-4 and good["n"]["mom_rem"] < 1e-10       # 1 + 2e-6 measured
    bad = run(N, r, "D0m", "dual", a, b, dl)
    assert bad["t"]["prod_rem"] > 100 and bad["t"]["E_rem_max"] > 1.5     # the old dual cells: real energy growth

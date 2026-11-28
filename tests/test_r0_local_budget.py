"""A-8 (#173), docs/notes/122: the local-budget verifier, from flux arrays; its detection power on the known defects
(#156's c = 1/2 rule, #162's D1) and the clean budgets of the library and of A-3's configuration."""

import numpy as np
import pytest

from cases.r0_local_budget import _run, summarise, verify

pytestmark = pytest.mark.xdist_group("r0_local_budget")


def test_verifier_on_synthetic_fluxes():
    n = 8
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 0.2, n + 1)
    F[-1] = F[0]                                                  # periodic
    un = F + 0.1
    c0 = rng.uniform(0, 1, n)
    c1 = c0 + F[:-1] - F[1:]
    ff = [np.full(n + 1, 2.0)]
    Fq = [2.0 * F]
    q0 = [2.0 * c0]
    q1 = [q0[0] + Fq[0][:-1] - Fq[0][1:]]
    sw = {"axis": 0, "periodic": (True,), "un": un, "F": F, "c0": c0, "c1": c1, "Sc": 0 * c0, "Sdiv": 0 * c0,
          "Fq": Fq, "ff": ff, "phase": [0], "q0": q0, "q1": q1, "Sq": [0 * c0], "conc": [np.full(n, 2.0)]}
    v = verify(sw)
    assert v["L1"]["max"] < 1e-15 and v["L2"][0]["max"] < 1e-15 and v["L3"]["max"] == 0 and v["L4"][0]["max"] < 1e-15
    sw["Fq"] = [Fq[0].copy()]
    sw["Fq"][0][3] += 1e-6                                        # a broken face: L2 and L4 point at it
    v = verify(sw)
    assert v["L4"][0]["at"] == [3] and v["L2"][0]["max"] > 1e-7


@pytest.mark.parametrize("config", ["vortex", "free-stream"])
def test_library_and_full_configuration_budgets(config):
    for conf in ("base", "full"):
        cap, scale = _run(config, conf)
        r = summarise(cap, scale_q=scale, lin=conf == "full")
        assert max(r["worst"][k] for k in ("L1", "L2", "L3", "L4", "L5")) < 1e-14
    if config == "free-stream":
        assert summarise(_run(config, "base")[0])["zero_momentum_flux_faces"] > 0           # D1 seen in the base
        assert summarise(_run(config, "full")[0], lin=True)["zero_momentum_flux_faces"] == 0


def test_rule_before_160_is_detected_at_half_cells():
    r = summarise(_run("half, rule before #160", "base")[0], scale_q=2.5)
    assert r["L5_special"]["c=1/2"] > 1e-8
    assert summarise(_run("half", "base")[0], scale_q=2.5)["L5_special"]["c=1/2"] < 1e-15


def test_l6_is_measured_from_the_density_the_solver_used():
    # audit on #173: L6 compares the transported mass with the density actually used to recover the velocity
    base = summarise(_run("free-stream", "base")[0])["worst"]["L6"]
    full = summarise(_run("free-stream", "full")[0])["worst"]["L6"]
    assert base > 1e-6 and 0 < full < 1e-12

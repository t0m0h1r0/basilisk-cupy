"""docs/notes/104 (Gate 1-A): the pressure projection on PCCD cell averages."""

import math

import numpy as np

from cases.projection_gate import _y_operators, layered, projection_accuracy, schur_ratios, structure_1d


def test_high_order_correction_removes_the_single_phase_checkerboard():
    r = {x["alpha"]: x for x in schur_ratios(alphas=(0.0, 1.2), n=513)["pccd"]}
    assert r[0.0]["1d_min"] < 1e-12 and r[0.0]["2d_min"] < 1e-12
    assert abs(r[1.2]["1d_min"] - 0.472) < 2e-3 and abs(r[1.2]["2d_min"] - 0.382) < 2e-3
    s0, s1 = structure_1d(32, 0.0), structure_1d(32, 1.2)
    assert s0["null_B"] == 2 and s1["null_B"] == 1 and s1["null_L"] == 1
    for s in (s0, s1):          # G-orthogonal projection: idempotent, divergence-free, no energy growth, momentum
        assert s["idempotent"] < 1e-12 and s["B_Pi"] < 1e-12 and s["energy_max"] < 1e-12 and s["momentum"] < 1e-12
    assert s1["Pi_u_constant"] < 1e-12 < s0["Pi_u_constant"]


def test_correction_costs_one_order_of_projection_accuracy():
    for al, order in ((0.0, 6), (1.2, 5)):
        a, b = projection_accuracy(al, thetas=(0.2, 0.1))
        assert abs(math.log2(a["div_free_err"] / b["div_free_err"]) - order) < 0.3


def test_two_phase_split_pccd_faces_ignore_the_mixed_cell():
    N, ratio, c = 32, 1e3, 0.5
    a = (N // 4 + 1 - c) / N
    G, I, m, jump = _y_operators(N, ratio, a, 0.7137)
    assert np.abs(I[:, 8]).max() < 1e-12 and np.abs(I[:, 22]).max() < 1e-12   # the two mixed cells
    o = layered(N, ratio, a, ntheta=3)
    assert o["alphas"][0.0]["null_By"] == 3                  # constant + one per mixed cell
    assert o["alphas"][1.2]["null_By"] == 1 and o["alphas"][1.2]["min"] < 1e-3
